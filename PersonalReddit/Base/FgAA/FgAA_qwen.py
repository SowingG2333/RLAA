import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# 匿名化LLM提示词
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not. 
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not. 
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""
PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user as well as [Personal Inferences] made about the user that can be made from the [Comment]. Your task is to anonymize the [Comment] to prevent these [Personal Inferences], while keeping the rest of the text intact.

[Comment]:
{user_response}

[Personal Inferences]:
{feedback}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
**Then, ONLY return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".**
"""
# 攻击者LLM提示词
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response must follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, current_city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "current_city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
**JSON Value Rules (CRITICAL):**
1. age: must be an accurate integer (e.g., 35, not "35" or "30-40").
2. sex: must be one of the following exact strings: "male" or "female".
3. relationship_status: must be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced" or "widowed".
4. income_level: must be one of the following exact strings: "low", "middle", "high" or "very high".
5. current_city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China").
6. education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner").
"""

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline"""
    if dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype = dtype.lower()
        if dtype == "fp16": torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"): torch_dtype = torch.bfloat16
        else: torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {"device_map": "auto"} if device is None else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa", **kwargs)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device else None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))
    return gen_pipe, tokenizer

# --- [NEW] ---
# 从您上一个（Arbitrator）脚本中引入的更健壮的 JSON 提取器
def extract_first_json(text: str) -> Optional[Any]:
    """
    从可能包含额外文本的字符串中提取第一个有效的 JSON 对象 ( { } ) 或列表 ( [ ] )
    """
    i = 0
    while i < len(text):
        first_brace = text.find("{", i)
        first_bracket = text.find("[", i)
        
        start_index = -1
        start_char = ''
        end_char = ''

        if first_brace == -1 and first_bracket == -1:
            logging.debug("No JSON object or list found in text.")
            return None 
        
        if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
            start_index = first_brace
            start_char = '{'
            end_char = '}'
        else:
            start_index = first_bracket
            start_char = '['
            end_char = ']'

        depth = 0
        for j in range(start_index, len(text)):
            if text[j] == start_char:
                depth += 1
            elif text[j] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_index : j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Found matching brackets but invalid JSON: {e}. Text: {candidate[:100]}... Continuing search.")
                        i = j + 1 
                        break
        else:
            logging.warning("Found JSON start but no matching end bracket/brace. Continuing search.")
            i = start_index + 1
    
    logging.debug("No valid JSON found after searching entire text.")
    return None

# --- [NEW] ---
# Qwen3 响应解析器
def extract_qwen3_response(raw_text: str) -> str:
    """
    Extracts the final response from a Qwen3 output that may contain <think>...</think> blocks.
    It finds the last '</think>' tag and returns everything after it.
    """
    think_tag = "</think>"
    index = raw_text.rfind(think_tag)
    
    if index != -1:
        # Found the tag, return the text after it
        return raw_text[index + len(think_tag):].strip()
    else:
        # No think tag found, return the original text
        return raw_text.strip()

# --- [NEW] ---
# 新的早停条件检查器
def check_all_unknown(guess_json: Dict[str, Any]) -> bool:
    """
    检查 Attacker 的猜测是否所有属性都为 'unknown'。
    """
    if not guess_json or "error" in guess_json:
        return False # JSON 解析失败不是成功

    # 这些 key 必须与 PROMPT_ATTACKER_USER 中的 'Guess:' 部分完全匹配
    ATTRIBUTES_TO_CHECK = [
        "age", "sex", "city_country", "birth_city_country", 
        "education", "occupation", "income_level", "relationship_status"
    ]
    
    for key in ATTRIBUTES_TO_CHECK:
        value = guess_json.get(key)
        # 检查值是否 *不是* 字符串 "unknown"
        if not (isinstance(value, str) and value.strip().lower() == "unknown"):
            # 只要有一个值不是 "unknown"，就返回 False
            logging.debug(f"[check_all_unknown] Failed: '{key}' was '{value}'")
            return False
    
    # 如果循环完成，说明所有值都是 "unknown"
    logging.debug("[check_all_unknown] Success: All attributes are 'unknown'.")
    return True

# --- [MODIFIED] ---
# 使用了来自您上一个脚本的更健壮的解析器
def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析 *非结构化的推理* 和 猜测(JSON)。"""
    inference_block = ""
    guess_json = {}
    try:
        inference_block_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if inference_block_match:
            inference_block = inference_block_match.group(1).strip()
        else:
            logging.warning("Could not find 'Inference:' block in attacker output.")
            inference_block = "No Inference block found."
        
        # 使用更健壮的 JSON 提取器
        guess_json = extract_first_json(response_text)
        if guess_json is None:
             guess_json = {"error": "No JSON object found in attacker output"}
             logging.warning(f"No JSON found: {response_text[:100]}...")
        if not isinstance(guess_json, dict):
             guess_json = {"error": "Extracted JSON is not a dictionary"}
             logging.warning(f"Extracted JSON is not a dict: {str(guess_json)[:100]}...")

    except Exception as e:
        logging.error(f"Error parsing attacker output: {e}\nOutput:\n{response_text[:500]}...", exc_info=True)
        return {"inference": str(e), "guess_json": {}}
    
    return {"inference": inference_block, "guess_json": guess_json}

# --- [MODIFIED] ---
def call_anonymizer(pipe, user_response: str, feedback: str, terminator_ids: List[int], record_id: int) -> str:
    user_content = PROMPT_ANONYMIZER_USER.format(
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    
    # 启用 thinking
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
    
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT (Raw) ---\n{out}\n--- End ---")
    
    # 移除 thinking 块
    out = extract_qwen3_response(out)
    
    parts = out.split('#', 1)
    explanation = parts[0].strip() if len(parts) == 2 else "No explanation provided (LLM failed to follow format)."
    final_response = parts[1].strip().strip('"').strip() if len(parts) == 2 else out.strip().strip('"').strip()
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION (post-think) ---\n{explanation}\n--- End ---")
    return final_response

# --- [MODIFIED] ---
def call_attacker(pipe, user_response: str, terminator_ids: List[int], record_id: int) -> Dict[str, Any]:
    user_content = PROMPT_ATTACKER_USER.format(
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    
    # 启用 thinking
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
    
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ATTACKER FULL OUTPUT (Raw) ---\n{out}\n--- End ---")
    
    # 移除 thinking 块
    out = extract_qwen3_response(out)
    
    parsed_data = parse_attacker_output(out)
    logging.debug(f"[Record {record_id}] --- ATTACKER PARSED (post-think) ---\n{parsed_data}\n--- End ---")
    return parsed_data

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles"""
    leaked_attributes: List[str] = []
    def _log(key, guess, true): return f"{key} (guessed: '{guess}', true: '{true}')"
    true_age = true_profile.get("age"); guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3: leaked_attributes.append(_log("age", guessed_age, true_age))
    simple_keys = ["sex", "relationship_status"]
    for key in simple_keys:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            if true_val.strip().lower() == guess_val.strip().lower(): leaked_attributes.append(_log(key, guess_val, true_val))
    complex_keys = ["city_country", "birth_city_country", "education", "occupation", "income_level"]
    for key in complex_keys:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
            if len(guess_s) < 3: continue
            if (guess_s in true_s) or (true_s in guess_s): leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- [MODIFIED] ---
def adversarial_anonymization(
    pipe,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 5,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""
    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        parsed_attack = call_attacker(pipe, original_response, terminator_ids, record_id)
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
        
        if "error" in attacker_guess:
             raise ValueError(f"Attacker guess JSON error: {attacker_guess.get('error')}")

    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- [MODIFIED] 步骤 0.5: 初始裁判 (新的早停条件) ---
    # 仍然运行 compare_profiles 来记录初始泄露了多少
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    
    # 【新的早停条件】
    if check_all_unknown(attacker_guess):
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). Attacker guessed 'unknown' for all attributes.")
        meta["status"] = "success_on_original"
        meta["final_leaked_attributes"] = [] # 成功了，所以最终泄露为空
        return current_anonymized_response, meta
    
    # 如果没成功，记录初始泄露（即使是空列表）
    logging.info(f"[Record {record_id}] Initial attack did not guess 'unknown' for all. Leaked (vs true): {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer(
                pipe, current_anonymized_response, feedback, terminator_ids, record_id
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta # 返回 *上一次* 成功的匿名化结果

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker(pipe, current_anonymized_response, terminator_ids, record_id)
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess

            if "error" in attacker_guess:
                raise ValueError(f"Attacker guess JSON error: {attacker_guess.get('error')}")
                
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            continue # 跳过本轮的评判，进入下一轮（使用旧的 feedback）

        # --- [MODIFIED] 步骤 3: 评判泄露 (新的早停条件) ---
        logging.info(f"{iteration_log_prefix} Calling Judge...")
        
        # 【新的早停条件】
        if check_all_unknown(attacker_guess):
            logging.info(f"{iteration_log_prefix} Success! Attacker guessed 'unknown' for all attributes.")
            meta["status"] = "success"
            meta["final_leaked_attributes"] = [] # 成功了，泄露清零
            return current_anonymized_response, meta
        
        # 如果没成功，记录一下（与真实 PII 相比）泄露了什么
        leaked = compare_profiles(true_personality, attacker_guess)
        logging.info(f"{iteration_log_prefix} Failed. Attacker still guessing. Leaked (vs true): {leaked}")
        meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked (vs true): {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    try:
        personality = data.get("personality")
        response = str(data.get("response"))
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not all([personality, response]):
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data

    if not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        original_response=response,
        true_personality=personality,
        terminator_ids=terminator_ids,
        max_iterations=max_iterations,
        record_id=record_id
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的回答进行匿名化")
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新 token 数（全局）")
    parser.add_argument("--log_file", type=str, default="anonymizer_local.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    
    records_to_process = []
    for i, line in enumerate(lines):
        if line.strip():
            try:
                records_to_process.append((i, json.loads(line))) # [MODIFIED] 传递原始索引
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON on line {i+1}. Skipping.")
                
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 ---
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    # --- 串行处理 ---
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "unknown_fail": 0}
    
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)
    
    for rec_tuple in tqdm(records_to_process, desc="Anonymizing profiles"): 
        rec_idx, rec_data = rec_tuple # [MODIFIED] 解包
        try:
            processed_record = _task(rec_idx, rec_data) # [MODIFIED] 传递 rec_idx
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] = counters.get(status, 0) + 1
        except Exception as exc:
            logging.error(f"[Record {rec_idx}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1

    # --- 结果写入 ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            status = result.get("anonymization_meta", {}).get("status")
            if status in ("success", "success_on_original"): success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # --- 记录最终摘要 ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in counters.items():
        if count > 0: logging.info(f"    {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()