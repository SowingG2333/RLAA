import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict # [修复] 导入 defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- [新] PROMPT 1: Attacker (Health Specific) ---
PROMPT_ATTACKER_HEALTH_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_HEALTH_USER = """
Analyze the [Comments] below to guess the author's **health_issue**.

[Comments]:
{user_response}

Provide your response in this exact format:

Reasoning: [Your step-by-step reasoning based ONLY on the Comments, focusing on diagnoses, medications, or symptoms]
Guess: [Your single best guess for the *main* health issue (e.g., "psoriasis", "anxiety", "antabuse"). Use "unknown" if unable to infer.]
Certainty: [A number from 1 (low certainty/bias) to 5 (high certainty/evidence)]

Output ONLY the Reasoning, Guess, and Certainty lines in the specified format.
"""

# --- [新] PROMPT 2: Anonymizer (Health Specific) ---
PROMPT_ANONYMIZER_HEALTH_SYSTEM = """
You are a privacy assistant designed to rewrite any given text to hide the author’s real identity while keeping the relevant and non-private information in the text unchanged.
"""
PROMPT_ANONYMIZER_HEALTH_USER = """
Rewrite the [Original Comments] below. The goal is to make it seem like the author matches the [Target Persona], NOT the [Real Profile].
Use the [Clues to Fix] to identify parts of the original comments that revealed the real profile.
Change ONLY the parts of the text related to health to match the [Target Persona]. Keep the rewritten text natural and coherent.

[Target Persona]:
{target_attributes_dict_str}

[Original Comments]:
{user_response}

[Real Profile]: (Attributes to Hide)
{true_attributes_dict_str}

[Clues to Fix]: (Reasoning why the original comments revealed the real profile)
{combined_feedback}

Output MUST be in this exact JSON format:
{{
    "explanation": "<Your brief explanation of what health terms you changed>",
    "anonymized_comment": "<Your rewritten comment>"
}}
"""

# --- (来自 Incogni_local.py) ---
def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline"""
    if dtype is None:
        if torch.cuda.is_available(): torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available(): torch_dtype = torch.float16
        else: torch_dtype = torch.float32
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

# --- (来自 Incogni_local.py) ---
def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取第一个完整 JSON 对象"""
    start = text.find("{")
    if start == -1: return None
    depth = 0; end = -1
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end != -1:
        candidate = text[start:end+1]
        try: return json.loads(candidate)
        except Exception: pass
    try: # Fallback
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{"); end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start: return json.loads(cleaned[start:end+1])
    except Exception: return None
    return None

# --- [新] 解析 Health Attacker 的输出 ---
def parse_attacker_output_health(response_text: str, record_id: int) -> Dict[str, Any]:
    """从 Health Attacker 的自由格式输出中解析"""
    reasoning = "Parsing failed."; first_guess = "unknown"; certainty = 0
    log_prefix = f"[Record-{record_id}]"
    try:
        reasoning_match = re.search(r"Reasoning:(.*?)(Guess:|Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match: reasoning = reasoning_match.group(1).strip()
        
        guess_match = re.search(r"Guess:(.*?)(Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if guess_match:
            first_guess_str = guess_match.group(1).strip().split(';')[0].strip()
            if first_guess_str:
                first_guess = first_guess_str
            else:
                first_guess = "unknown"
        else: 
            logging.warning(f"{log_prefix} No 'Guess:' line found in attacker output.")
            first_guess = "unknown"
            
        certainty_match = re.search(r"Certainty:.*?(\d)", response_text, re.DOTALL | re.IGNORECASE)
        if certainty_match:
            try: certainty = int(certainty_match.group(1))
            except ValueError: certainty = 0

    except Exception as e: 
        logging.error(f"{log_prefix} Error parsing attacker output: {e}\nOutput:\n{response_text[:500]}...", exc_info=True)
    
    return {"reasoning": reasoning, "guess": first_guess, "certainty": certainty}

# --- [新] 调用 Health Attacker (本地) ---
def call_attacker_local_health(
    pipe, 
    user_response: str, 
    terminator_ids: List[int], 
    record_id: int
) -> Dict[str, Any]:
    """为 health_issue 属性调用本地 Attacker LLM"""
    iteration_log_prefix = f"[Record-{record_id}] Attacking 'health_issue'"
    logging.debug(f"{iteration_log_prefix}...")
    user_content = PROMPT_ATTACKER_HEALTH_USER.format(
        user_response=user_response
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_HEALTH_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Attacker PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1; do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Attacker FULL OUTPUT ---\n{response_text}\n--- End ---")
        return parse_attacker_output_health(response_text, record_id)
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local attack.", exc_info=True)
        return {"reasoning": f"Local LLM Error: {e}", "guess": "unknown", "certainty": 0, "error": True}

# --- [新] 调用 Health Anonymizer (本地) ---
def call_anonymizer_local_health(
    pipe, 
    user_response: str, 
    target_attributes: Dict[str, Any], 
    combined_feedback: str, 
    true_attributes: Dict[str, Any], 
    terminator_ids: List[int], 
    record_id: int
) -> str:
    """调用多属性本地 Anonymizer LLM"""
    iteration_log_prefix = f"[Record-{record_id}] Anonymizing..."
    logging.debug(f"{iteration_log_prefix} with target: {target_attributes}")
    
    target_attributes_dict_str = json.dumps(target_attributes, indent=2, ensure_ascii=False)
    true_attributes_dict_str = json.dumps(true_attributes, indent=2, ensure_ascii=False)
    
    user_content = PROMPT_ANONYMIZER_HEALTH_USER.format(
        target_attributes_dict_str=target_attributes_dict_str,
        user_response=user_response,
        true_attributes_dict_str=true_attributes_dict_str,
        combined_feedback=combined_feedback if combined_feedback else "No specific cues identified. Focus on reflecting the target persona."
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_HEALTH_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Anonymizer PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5; do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Anonymizer FULL OUTPUT ---\n{response_text}\n--- End ---")
        anon_output_json = extract_first_json_object(response_text)
        
        if anon_output_json and isinstance(anon_output_json, dict) and "anonymized_comment" in anon_output_json:
             explanation = anon_output_json.get("explanation", "No explanation parsed.")
             logging.debug(f"{iteration_log_prefix} Anonymizer explanation: {explanation}")
             return anon_output_json["anonymized_comment"]
        else:
            logging.warning(f"{iteration_log_prefix} Anonymizer failed to produce valid JSON or missing 'anonymized_comment'. Returning original text.")
            return user_response
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local anonymization.", exc_info=True)
        return user_response # 发生错误时返回原始（或上一次的）文本

# --- [新] 专门用于 Health 的比较逻辑 ---
def compare_profiles_health(
    true_profile: Dict[str, Any], 
    guessed_profile_dict: Dict[str, Any],
    min_string_match_len: int = 3
) -> List[str]:
    """
    比较 health profiles。
    'true_profile' 包含一个 'health_issue' 列表。
    'guessed_profile_dict' 包含一个 'health_issue' 字符串。
    """
    leaked_attributes: List[str] = []
    
    true_val_list = true_profile.get("health_issue")
    guess_val_str = guessed_profile_dict.get("health_issue") # 键是 'health_issue'

    if not (isinstance(true_val_list, list) and true_val_list and 
            isinstance(guess_val_str, str) and guess_val_str and 
            guess_val_str.lower() != "unknown"):
        return [] 

    guess_s = guess_val_str.strip().lower()
    if len(guess_s) < min_string_match_len:
        return []

    for true_item in true_val_list:
        if isinstance(true_item, str):
            true_s = true_item.strip().lower()
            # 模糊匹配：任一方向的子字符串
            if (guess_s in true_s) or (true_s in guess_s):
                leaked_attributes.append(f"health_issue (guessed: '{guess_s}', true: '{true_s}')")
                break # 找到匹配项
                
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- [新] 专门用于 Health 的对抗循环 (本地) ---
def adversarial_anonymization_health_local(
    pipe, # transformers pipeline
    original_response: str,
    true_personality: Dict[str, Any],
    target_persona_dict: Dict[str, Any], # 接收 Target Persona
    terminator_ids: List[int],
    max_iterations: int = 3,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    '''执行 IncogniText (本地版)，专门针对 Health。'''
    record_log_prefix = f"[Record-{record_id}]"
    current_anonymized_response = original_response
    
    target_health_attributes = target_persona_dict.get("health_issue")
    if not (target_health_attributes and isinstance(target_health_attributes, list)):
        meta = {"status": "skipped_no_valid_targets_in_input", "error": "Target persona dict is missing a valid 'health_issue' list"}
        logging.error(f"{record_log_prefix} {meta['error']}. Skipping.")
        return original_response, meta

    true_health_attributes = {"health_issue": true_personality.get("health_issue", [])}
    if not true_health_attributes["health_issue"]:
         meta = {"status": "skipped_no_real_health_issue", "error": "Real personality dict is missing 'health_issue'"}
         logging.warning(f"{record_log_prefix} {meta['error']}. Proceeding, but judging may fail.")
         
    meta = {
        "status": "max_iterations_reached", "iterations_used": 0, 
        "final_leaked_attributes": [], "final_attacker_guesses": {}, 
        "target_attributes": target_persona_dict,
        "attributes_processed": ["health_issue"]
    }
    
    logging.info(f"{record_log_prefix} Starting IncogniText for [health_issue]. Target: {target_health_attributes}")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"{record_log_prefix} Starting initial attack (Round 0)...")
    try:
        attack_result = call_attacker_local_health(
            pipe, original_response, terminator_ids, record_id
        )
        if attack_result.get("error"): raise Exception(f"Attacker LLM Error: {attack_result.get('reasoning')}")
        
        attacker_guess_str = attack_result.get("guess")
        feedback = attack_result.get("reasoning", "No inference provided by attacker.")
        attacker_guess_dict = {"health_issue": attacker_guess_str} 
        meta["final_attacker_guesses"] = attacker_guess_dict

    except Exception as e:
        logging.error(f"{record_log_prefix}[Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "model_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始裁判 ---
    leaked = compare_profiles_health(true_health_attributes, attacker_guess_dict)
    meta["final_leaked_attributes"] = leaked
    
    if not leaked or attacker_guess_str.lower() == "unknown":
        logging.info(f"{record_log_prefix} Success on original text (0 iterations). No leaks found or guess was 'unknown'.")
        meta["status"] = "success_on_original"
        return current_anonymized_response, meta
    
    logging.info(f"{record_log_prefix} Initial attack leaked: {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"{record_log_prefix} [Round {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer_local_health(
                pipe, current_anonymized_response, 
                target_persona_dict, # Target = {"health_issue": ["..."]}
                feedback, 
                true_health_attributes, # Real = {"health_issue": ["..."]}
                terminator_ids, record_id
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "model_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta 

        # 2) 攻击者推断
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            attack_result = call_attacker_local_health(
                pipe, current_anonymized_response, terminator_ids, record_id
            )
            if attack_result.get("error"): raise Exception(f"Attacker LLM Error: {attack_result.get('reasoning')}")

            attacker_guess_str = attack_result.get("guess")
            feedback = attack_result.get("reasoning", "No inference provided by attacker.")
            attacker_guess_dict = {"health_issue": attacker_guess_str}
            meta["final_attacker_guesses"] = attacker_guess_dict
            
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Attacker failed: {e}", exc_info=True)
            meta["status"] = "model_error"; meta["error"] = f"attacker_error_round_{i}: {e}"
            return current_anonymized_response, meta 

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles_health)...")
        leaked = compare_profiles_health(true_health_attributes, attacker_guess_dict)
        meta["final_leaked_attributes"] = leaked

        if not leaked or attacker_guess_str.lower() == "unknown":
            logging.info(f"{iteration_log_prefix} Success! No leaks found (or guess was 'unknown').")
            meta["status"] = "success" 
            meta["final_leaked_attributes"] = [] 
            return current_anonymized_response, meta

        logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
        # 循环继续...

    logging.warning(f"{record_log_prefix} Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- [新] 专门用于 Health 的记录处理器 ---
def process_record_health_local(
    pipe, 
    data: Dict[str, Any], 
    max_iterations: int, 
    record_id: int, 
    terminator_ids: List[int]
) -> Dict[str, Any]:
    """处理单条记录 (IncogniText Local Health version - 读取 Persona)"""
    record_log_prefix = f"[Record-{record_id}]"
    logging.debug(f"{record_log_prefix} Starting processing.")
    try:
        data["record_id"] = record_id # 确保
        personality = data.get("personality")
        response = str(data.get("response"))
        target_persona = data.get("target_persona") # 由阶段 1 生成
        
        if not isinstance(personality, dict):
            raise ValueError("'personality' key is missing or not a dictionary.")
        if not isinstance(target_persona, dict):
            raise ValueError("'target_persona' key is missing or not a dictionary.")
        if not response:
            raise ValueError("'response' key is missing or empty.")
            
    except Exception as e:
        status = "skipped_data_read_error"; logging.warning(f"{record_log_prefix} Skipped: Data read error: {e}")
        data["anonymization_meta"] = {"status": status, "error": str(e)}; return data

    try:
        anonymized_response, meta = adversarial_anonymization_health_local(
            pipe=pipe,
            original_response=response,
            true_personality=personality,
            target_persona_dict=target_persona, # 传入 Persona
            terminator_ids=terminator_ids,
            max_iterations=max_iterations,
            record_id=record_id
        )
        data["anonymized_response"] = anonymized_response
        data["anonymization_meta"] = meta
    
    except Exception as e:
        # 捕获来自模型调用的致命错误
        logging.error(f"{record_log_prefix} Generated an unhandled exception: {e}", exc_info=True)
        data["anonymization_meta"] = {"status": "unknown_fail", "error": str(e)}
        
    return data

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Run IncogniText Anonymization for Health (Local Model)")
    # --- 输入/输出参数 ---
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path (data + target_persona)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL path (final anonymized data)")
    # --- 模型参数 ---
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", help="Hugging Face model name")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu/mps/cuda:0. Default auto")
    parser.add_argument("--dtype", type=str, default=None, help="Tensor dtype: fp32/fp16/bf16. Default auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory hint")
    parser.add_argument("--terminator_tokens", type=str, nargs='+', default=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"], help="用于提前停止的额外 terminator token 列表")
    # --- IncogniText 参数 ---
    parser.add_argument("--max_iterations", type=int, default=3, help="Max IncogniText iterations per record")
    # --- 控制参数 ---
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--success_file", type=str, default=None, help="Optional path for successful records")
    parser.add_argument("--failed_file", type=str, default=None, help="Optional path for failed/skipped records")
    # --- 日志参数 ---
    parser.add_argument("--log_file", type=str, default="run_incognitext_health_local.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file, 'w', 'utf-8')]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("transformers").setLevel(logging.ERROR); logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}.")
    logging.info(f"Run arguments: {vars(args)}") # 记录所有参数

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: logging.error(f"Input file not found: '{args.input_file}'"); sys.exit(1)
    
    records_to_process = []
    for i, line in enumerate(lines):
        if line.strip():
            try:
                records_to_process.append((i, json.loads(line)))
            except json.JSONDecodeError:
                logging.warning(f"Skipping line {i+1} due to invalid JSON.")
    
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 ---
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception as e: logging.error(f"Failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in args.terminator_tokens] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    # --- IncogniText 处理 (串行) ---
    logging.info(f"Starting IncogniText (Health, Local) for {len(records_to_process)} records...")
    results = []; counters = defaultdict(int)
    
    # [修改] 这是一个串行循环，与 Incogni_local.py 保持一致
    for rec_idx, rec in tqdm(records_to_process, desc="Running IncogniText (Local)"):
        try:
            processed_record = process_record_health_local(
                gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids
            )
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] += 1
        except Exception as exc:
            logging.error(f"[Record {rec_idx}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1

    # --- 结果写入 ---
    logging.info(f"\nAnonymization complete. Writing {len(results)} results...")
    
    # [修改] 按 record_id 排序
    results.sort(key=lambda x: x.get("record_id", float('inf')))

    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            status = result.get("anonymization_meta", {}).get("status")
            if status in ("success", "success_on_original"): 
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: 
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # --- 记录最终摘要 ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in sorted(counters.items()):
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Output saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful records saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped records saved to '{args.failed_file}'.")
    logging.info(f"Logs saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()