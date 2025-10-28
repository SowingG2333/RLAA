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

# --- 提示词 ---
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
**Then, ONLY return the anonymized text without any extra preamble like "Here is the "anonymized text:" or suffix like "Note:".**
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
**JSON Value Rules (CRITICAL):**
1.  age: MUST be an accurate integer (e.g., 35, not "35" or "30-40") or string "unknown".
2.  sex: MUST be one of the following exact strings: "male", "female", or "unknown".
3.  relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced", "widowed", or "unknown".
4.  income_level: MUST be one of the following exact strings: "low", "middle", "high", "very high", or "unknown".
5.  city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China") if the city is unknown, or "unknown".
6.  education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner") or "unknown".
7.  **If you cannot reasonably infer an attribute with high confidence, you MUST use the string value "unknown" instead of randomly guessing an answer.**
"""

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件为字典列表"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try: data.append(json.loads(line))
                except json.JSONDecodeError as e: logging.warning(f"Skipping line {i+1} due to JSON decode error: {e}")
        logging.info(f"Loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError: logging.error(f"Input file not found: '{filepath}'"); sys.exit(1)
    except Exception as e: logging.error(f"Error loading file '{filepath}': {e}", exc_info=True); sys.exit(1)

def extract_anonymized_text(anonymizer_output: str) -> str:
    """从 Anonymizer 的输出中提取文本部分"""
    parts = anonymizer_output.split('#', 1)
    if len(parts) == 2:
        anonymized_text = parts[1].strip().strip('"').strip()
        anonymized_text = anonymized_text.split("<|eot_id|>")[0].strip()
        anonymized_text = anonymized_text.split("<|end_of_text|>")[0].strip()
        return anonymized_text
    logging.warning("Anonymizer output did not contain '#'. Using raw.")
    raw_output = anonymizer_output.strip().strip('"').strip()
    raw_output = raw_output.split("<|eot_id|>")[0].strip()
    raw_output = raw_output.split("<|end_of_text|>")[0].strip()
    return raw_output

def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析推理和猜测(JSON)"""
    inference = ""
    guess_json = {}
    inference_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if inference_match: inference = inference_match.group(1).strip()
    json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try: guess_json = json.loads(json_str)
        except json.JSONDecodeError: logging.warning(f"Failed to parse JSON: {json_str[:100]}..."); guess_json = {"error": "Failed to parse JSON"}
    else: logging.warning(f"No JSON found: {response_text[:100]}..."); guess_json = {"error": "No JSON found"}

    return {"inference": inference, "guess_json": guess_json}

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

def build_sft_pipeline(model_path: str, device: Optional[str] = None, dtype: Optional[str] = None):
    """加载 SFT 微调后的模型和 Tokenizer 创建 Pipeline"""
    logging.info(f"Loading SFT model and tokenizer from: {model_path}")
    try:
        if dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        elif dtype.lower() in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype.lower() == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        logging.info(f"Using dtype: {torch_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer does not have a pad token set. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        pipe_device = device if "auto" not in str(model.hf_device_map) else None
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipe_device
        )
        logging.info(f"Pipeline created successfully for {model_path}. Device map: {model.hf_device_map}")
        return gen_pipe
    except Exception as e:
        logging.error(f"Error building pipeline for {model_path}: {e}", exc_info=True)
        raise

def call_anonymizer_sft(
    pipe, 
    user_response: str, 
    feedback: str, 
    terminator_ids: List[int], 
    record_id: int
) -> str:
    """使用 SFT Anonymizer Pipeline 调用匿名器"""
    anonymizer_messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
        {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
            user_response=str(user_response),
            feedback=str(feedback)
        )}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        anonymizer_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{prompt}\n--- End ---")
    
    temperature = 0.5; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, 
                   do_sample=do_sample, temperature=temperature if do_sample else None, 
                   top_p=0.9 if do_sample else None, 
                   pad_token_id=pipe.tokenizer.eos_token_id,
                   return_full_text=False)
    
    anonymizer_full_output = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT ---\n{anonymizer_full_output}\n--- End ---")
    
    parts = anonymizer_full_output.split('#', 1)
    explanation = parts[0].strip() if len(parts) == 2 else "No explanation provided (LLM failed to follow format)."
    final_response = parts[1].strip().strip('"').strip() if len(parts) == 2 else anonymizer_full_output.strip().strip('"').strip()
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION ---\n{explanation}\n--- End ---")
    
    return final_response

def call_attacker_sft(
    pipe, 
    user_response: str, 
    terminator_ids: List[int], 
    record_id: int
) -> Dict[str, Any]:
    """使用 SFT Attacker Pipeline 调用攻击器"""
    attacker_messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(
            user_response=str(user_response)
        )}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        attacker_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{prompt}\n--- End ---")

    temperature = 0.1; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, 
                   do_sample=do_sample, temperature=temperature if do_sample else None, 
                   top_p=0.9 if do_sample else None, 
                   pad_token_id=pipe.tokenizer.eos_token_id,
                   return_full_text=False)
    
    attacker_full_output = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ATTACKER FULL OUTPUT ---\n{attacker_full_output}\n--- End ---")
    
    parsed_data = parse_attacker_output(attacker_full_output)
    logging.debug(f"[Record {record_id}] --- ATTACKER PARSED ---\n{parsed_data}\n--- End ---")
    return parsed_data

# --- FgAA Core Logic ---
def adversarial_anonymization_sft(
    attacker_pipe,
    anonymizer_pipe,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 3,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程"""
    
    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        parsed_attack = call_attacker_sft(attacker_pipe, original_response, terminator_ids, record_id)
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {e}" # 状态码与 FgAA_local.py 一致
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始裁判 ---
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        return current_anonymized_response, meta
    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        previous_response = current_anonymized_response
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer_sft(
                anonymizer_pipe, current_anonymized_response, feedback, terminator_ids, record_id
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error_iter_{i+1}: {e}"
            return previous_response, meta # 返回上一轮的结果

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker_sft(attacker_pipe, current_anonymized_response, terminator_ids, record_id)
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            # FgAA_local.py 的逻辑是在 Attacker 失败时继续循环，而不是返回
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            continue # 继续到下一次迭代

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(true_personality, attacker_guess)
        if not leaked:
            logging.info(f"{iteration_log_prefix} Success! No attributes leaked.")
            meta["status"] = "success"
            return current_anonymized_response, meta
        logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
        meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Run FgAA using SFT-finetuned Llama models (Arbitrator Logic).")
    # SFT 特有参数
    parser.add_argument("--attacker_model_path", type=str, required=True, help="Path to the SFT Attacker model")
    parser.add_argument("--anonymizer_model_path", type=str, required=True, help="Path to the SFT Anonymizer model")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (e.g., 'cuda:0'). Default: auto")
    parser.add_argument("--dtype", type=str, default=None, help="Data type (e.g., 'bf16', 'fp16'). Default: auto (bf16 if supported)")
    
    # 从 FgAA_local.py 移植的参数
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=5, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--log_file", type=str, default="fgaa_sft_arbitrator.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    args = parser.parse_args()

    # --- Logger Setup (来自 FgAA_local.py) ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")

    # --- 数据加载 (来自 FgAA_local.py) ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 (SFT 版本) ---
    try:
        attacker_pipe = build_sft_pipeline(args.attacker_model_path, args.device, args.dtype)
        anonymizer_pipe = build_sft_pipeline(args.anonymizer_model_path, args.device, args.dtype)
    except Exception as e:
        logging.error(f"Failed to build pipelines: {e}", exc_info=True)
        sys.exit(1)

    # --- 终止符 (来自 FgAA_local.py) ---
    # 假设两个模型的 tokenizer 相同
    tokenizer = attacker_pipe.tokenizer
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    # --- 串行处理 (来自 FgAA_local.py) ---
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with SFT models...")
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "unknown_fail": 0}

    for i, record in enumerate(tqdm(records_to_process, desc="Anonymizing profiles")):
        record_id = i
        status = "unknown_fail" # 默认状态
        try:
            # 数据读取和验证
            try:
                personality = record.get("personality")
                response = str(record.get("response"))
            except Exception as e:
                logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
                status = "skipped_data_read_error"
                record["anonymization_meta"] = {"status": status}
                results.append(record)
                counters[status] += 1
                continue

            if not all([personality, response]):
                logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
                status = "skipped_incomplete_data"
                record["anonymization_meta"] = {"status": status}
                results.append(record)
                counters[status] += 1
                continue

            if not isinstance(personality, dict):
                logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
                status = "skipped_invalid_personality"
                record["anonymization_meta"] = {"status": status}
                results.append(record)
                counters[status] += 1
                continue
            
            # 核心处理
            anonymized_response, meta = adversarial_anonymization_sft(
                attacker_pipe=attacker_pipe,
                anonymizer_pipe=anonymizer_pipe,
                original_response=response,
                true_personality=personality,
                terminator_ids=terminator_ids,
                max_iterations=args.max_iterations,
                record_id=record_id
            )
            record["anonymized_response"] = anonymized_response
            record["anonymization_meta"] = meta
            results.append(record)
            status = meta.get('status', 'unknown_fail')
            counters[status] = counters.get(status, 0) + 1
            logging.info(f"[Record {record_id}] Finished processing. Status: {status}")

        except Exception as exc:
            logging.error(f"[Record {i}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1
            # 确保即使有未处理异常，原始记录也被添加
            if "anonymization_meta" not in record:
                record["anonymization_meta"] = {"status": "unknown_fail", "error": str(exc)}
                results.append(record)


    # --- 结果写入 (来自 FgAA_local.py) ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
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

    # --- 记录最终摘要 (来自 FgAA_local.py) ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in counters.items():
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()