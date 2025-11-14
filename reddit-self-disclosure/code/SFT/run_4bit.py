import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import torch
# 导入 BNBConfig 和 PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# --- 提示词 (与 FgAA_API.py 同步) ---
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

In a first line, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
In the next line, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
"""
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
CORE_PII_ATTRIBUTES_ALL = {"age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status", "health_issue"}

def generate_attacker_user_prompt(user_response: str, attributes_to_protect: List[str]) -> str:
    attributes_list_str = ", ".join(attributes_to_protect)
    json_keys_str = ", ".join([f'"{attr}"' for attr in attributes_to_protect])
    return f"""
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - **{attributes_list_str}**.

Guess: A single JSON object containing your best guess for the following key: **{json_keys_str}**.
"""

# --- 辅助函数 ---

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
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
    inference = ""
    guess_json = {}
    inference_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if inference_match: inference = inference_match.group(1).strip()
    json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try: 
            guess_json = json.loads(json_str)
        except json.JSONDecodeError: 
            logging.warning(f"Failed to parse JSON: {json_str[:100]}..."); 
            guess_json = {"error": "Failed to parse JSON"}
    else: 
        try:
            guess_json = json.loads(response_text)
            logging.debug("Parsed response directly as JSON."); inference = "No inference text found (parsed directly as JSON)."
        except json.JSONDecodeError:
            logging.warning(f"No JSON found: {response_text[:100]}..."); 
            guess_json = {"error": "No JSON found"}
    return {"inference": inference, "guess_json": guess_json}

# Judge Logic (与 FgAA_API.py 同步)
def compare_profiles(
    true_profile: Dict[str, Any], 
    guessed_profile: Dict[str, Any], 
    attributes_to_protect: List[str],
    age_tolerance: int,
    min_string_match_len: int
) -> List[str]:
    leaked_attributes: List[str] = []
    def _log(key, guess, true): return f"{key} (guessed: '{guess}', true: '{true}')"
    for key in attributes_to_protect:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        if guess_val is None or (isinstance(guess_val, str) and guess_val.strip().lower() == "unknown"):
            continue
        if key == "age":
            if isinstance(true_val, int) and isinstance(guess_val, int):
                if abs(true_val - guess_val) <= age_tolerance: 
                    leaked_attributes.append(_log("age", guess_val, true_val))
        elif key in ["sex", "relationship_status", "income_level"]:
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                if true_val.strip().lower() == guess_val.strip().lower(): 
                    leaked_attributes.append(_log(key, guess_val, true_val))
        elif key == "health_issue":
            if (isinstance(true_val, list) and true_val and isinstance(guess_val, str) and guess_val):
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= min_string_match_len:
                    for true_item in true_val:
                        if isinstance(true_item, str):
                            true_s = true_item.strip().lower()
                            if (guess_s in true_s) or (true_s in guess_s):
                                leaked_attributes.append(_log(key, guess_val, true_val)); break 
            elif (isinstance(true_val, str) and true_val and isinstance(guess_val, str) and guess_val):
                true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
                if len(guess_s) >= min_string_match_len:
                    if (guess_s in true_s) or (true_s in guess_s): 
                        leaked_attributes.append(_log(key, guess_val, true_val))
        elif key in ["city_country", "birth_city_country", "education", "occupation"]:
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
                if len(guess_s) < min_string_match_len: continue 
                if (guess_s in true_s) or (true_s in guess_s): 
                    leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison (subset: {attributes_to_protect}) complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- SFT Pipeline 调用函数 (适配器切换) ---

def call_anonymizer_sft(
    pipe, # 传入的 pipe 是共享的
    user_response: str, 
    feedback: str, 
    terminator_ids: List[int], 
    record_id: int
) -> str:
    """使用 SFT Anonymizer Pipeline 调用匿名器"""
    
    # 切换到 Anonymizer 适配器
    try:
        pipe.model.set_adapter("anonymizer")
        logging.debug(f"[Record {record_id}] Set active adapter to 'anonymizer'")
    except Exception as e:
        logging.error(f"[Record {record_id}] FAILED to set adapter 'anonymizer'. Is it loaded? Error: {e}")
        raise

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
    final_response = extract_anonymized_text(anonymizer_full_output)
    return final_response

def call_attacker_sft(
    pipe, # 传入的 pipe 是共享的
    user_response: str, 
    attributes_to_protect: List[str],
    terminator_ids: List[int], 
    record_id: int
) -> Dict[str, Any]:
    """使用 SFT Attacker Pipeline 调用攻击器"""
    
    # 切换到 Attacker 适配器
    try:
        pipe.model.set_adapter("attacker")
        logging.debug(f"[Record {record_id}] Set active adapter to 'attacker'")
    except Exception as e:
        logging.error(f"[Record {record_id}] FAILED to set adapter 'attacker'. Is it loaded? Error: {e}")
        raise

    user_content = generate_attacker_user_prompt(
        user_response=str(user_response),
        attributes_to_protect=attributes_to_protect
    )
    attacker_messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": user_content}
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

# --- FgAA 核心逻辑 (适配器切换版) ---

def adversarial_anonymization_sft(
    pipe, # 单个共享的 pipeline
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int,
    record_id: int,
    protect_attribute: Optional[str],
    age_tolerance: int,
    min_string_match_len: int
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程 (SFT 适配器切换版本)"""

    attributes_to_protect: List[str]
    if protect_attribute:
        if protect_attribute in CORE_PII_ATTRIBUTES_ALL:
            attributes_to_protect = [protect_attribute]
            logging.info(f"[Record {record_id}] Single attribute protection enabled for: '{protect_attribute}'")
        else:
            logging.error(f"[Record {record_id}] Invalid attribute '{protect_attribute}' specified for protection. Aborting record.")
            meta = {"status": "skipped_invalid_attribute", "error": f"Invalid --protect_attribute: {protect_attribute}"}
            return original_response, meta
    else:
        attributes_to_protect = sorted(list(CORE_PII_ATTRIBUTES_ALL))
        logging.info(f"[Record {record_id}] Full attribute protection enabled for: {attributes_to_protect}")

    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        parsed_attack = call_attacker_sft(
            pipe, original_response, attributes_to_protect, 
            terminator_ids, record_id
        )
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始裁判 ---
    leaked = compare_profiles(
        true_personality, attacker_guess, attributes_to_protect,
        age_tolerance=age_tolerance,
        min_string_match_len=min_string_match_len
    )
    meta["final_leaked_attributes"] = leaked
    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        # 遵循 FgAA_API.py 逻辑 (即使成功也继续)
        # return current_anonymized_response, meta 

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
                pipe, current_anonymized_response, feedback, terminator_ids, record_id
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error_iter_{i+1}: {e}"
            return previous_response, meta 

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker_sft(
                pipe, current_anonymized_response, attributes_to_protect, 
                terminator_ids, record_id
            )
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"attacker_error_round_{i}: {e}"
            return current_anonymized_response, meta

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(
            true_personality, attacker_guess, attributes_to_protect,
            age_tolerance=age_tolerance,
            min_string_match_len=min_string_match_len
        )
        
        # [!! 修正 !!] 这是我们 *唯一* 的成功早停条件
        all_unknown = False 
        if attacker_guess and isinstance(attacker_guess, dict):
            all_unknown = True 
            for key in attributes_to_protect:
                guess_val = attacker_guess.get(key)
                normalized_guess = ""
                if isinstance(guess_val, str):
                    normalized_guess = guess_val.strip().lower()
                if normalized_guess != "unknown":
                    all_unknown = False; break 
        
        if all_unknown:
            logging.info(f"{iteration_log_prefix} Success! All attributes guessed as 'unknown'.")
            meta["status"] = "success" 
            meta["final_leaked_attributes"] = [] 
            return current_anonymized_response, meta
        
        # [!! DELETED !!] 删除了 'if not leaked:' 这个早停条件
        # (因为 "猜错" != "猜 unknown")

        # 如果没有 'all_unknown' 且 *仍然* 存在泄露
        if leaked:
            logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
            meta["final_leaked_attributes"] = leaked
        # 如果没有 'all_unknown' 且 *没有* 泄露 (例如 猜 40 岁, 实际 30 岁)
        else:
            logging.info(f"{iteration_log_prefix} Attacker failed to leak, but did not guess 'unknown'. Continuing.")
            meta["final_leaked_attributes"] = [] # 确保更新泄露列表

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Run FgAA using SFT-finetuned Llama models (Adapter Switching).")
    
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the BASE model (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')")
    parser.add_argument("--attacker_model_path", type=str, required=True, help="Path to the SFT Attacker ADAPTER (e.g., ./ckpt/atk/checkpoint-310)")
    parser.add_argument("--anonymizer_model_path", type=str, required=True, help="Path to the SFT Anonymizer ADAPTER (e.g., ./ckpt/ano/checkpoint-310)")
    
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (e.g., 'cuda:0'). Default: auto")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type (e.g., 'bf16', 'fp16'). Default: bf16")
    
    # 从 FgAA_local.py/API.py 移植的参数
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数 (Default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--log_file", type=str, default="fgaa_sft_adapter_switch.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    # 从 FgAA_API.py 移植的 Judge/Logic 参数
    parser.add_argument("--protect_attribute", type=str, default=None, help="可选：指定要保护的单个属性")
    parser.add_argument("--age_tolerance", type=int, default=3, help="评判 'age' 时的容忍范围 (± 3 年)")
    parser.add_argument("--min_string_match_len", type=int, default=3, help="评判字符串属性 (如 city) 时的最小匹配长度")

    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    logging.info(f"Run arguments: {vars(args)}")

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    records_to_process = []
    for i, line in enumerate(lines):
        if line.strip():
            try: records_to_process.append((i, json.loads(line)))
            except json.JSONDecodeError: logging.warning(f"Skipping line {i+1} due to invalid JSON.")
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- [!! MODIFIED !!] 单模型加载逻辑 (包含双重补丁) ---
    try:
        import bitsandbytes
        try:
            # 尝试导入 peft 内部的 Linear4bit 类
            from peft.tuners.lora import Linear4bit as PeftLinear4bit
            peft_patch_target = PeftLinear4bit
            logging.debug("Successfully imported 'peft.tuners.lora.Linear4bit' for patching.")
        except ImportError:
            logging.warning("Could not import 'peft.tuners.lora.Linear4bit'. Will only patch 'bitsandbytes.nn.Linear4bit'.")
            peft_patch_target = None
        
        logging.info(f"Loading BASE model and tokenizer from: {args.base_model_path}")
        
        # 1. 配置量化
        compute_dtype = torch.bfloat16
        if args.dtype.lower() == "fp16":
            compute_dtype = torch.float16
        logging.info(f"Using compute_dtype: {compute_dtype}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # [!! 修正 !!]
        # 在 from_pretrained *之前* 应用补丁, 解决 'can't set attribute'
        # 我们不再使用 property, 而是直接设置静态属性, 
        # 因为 bitsandbytes 的 __init__ 需要写入它
        try:
            if not hasattr(bitsandbytes.nn.Linear4bit, "compute_dtype"):
                bitsandbytes.nn.Linear4bit.compute_dtype = compute_dtype
                logging.info(f"Patched 'bitsandbytes.nn.Linear4bit.compute_dtype' (static) to {compute_dtype}")
            else:
                 logging.info("'bitsandbytes.nn.Linear4bit.compute_dtype' already exists.")
        except Exception as e:
            logging.warning(f"Failed to patch bitsandbytes.nn.Linear4bit: {e}.")

        # [!! 修正 !!]
        # 补丁 2: 修复 'peft.tuners.lora.Linear4bit'
        # (用于 load_adapter 第二次)
        if peft_patch_target:
            try:
                if not hasattr(peft_patch_target, "compute_dtype"):
                    # peft 库需要的是一个 'property'
                    logging.info("Applying 'compute_dtype' PROPERTY patch to peft.tuners.lora.Linear4bit...")
                    p = property(lambda self, dtype=compute_dtype: dtype)
                    setattr(peft_patch_target, "compute_dtype", p)
                    logging.info(f"Patched 'peft.tuners.lora.Linear4bit.compute_dtype' (property) to {compute_dtype}")
                else:
                    logging.info("'peft.tuners.lora.Linear4bit.compute_dtype' already exists.")
            except Exception as e:
                logging.warning(f"Failed to patch peft.tuners.lora.Linear4bit: {e}.")


        # 2. 加载 4-bit 基础模型
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            quantization_config=bnb_config,
            device_map="auto", 
            trust_remote_code=True
        )
        logging.info(f"Base model loaded. Device map: {model.hf_device_map}")

        # 3. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer does not have a pad token set. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        # 4. 加载 Attacker 适配器
        logging.info(f"Loading Attacker ADAPTER from: {args.attacker_model_path}")
        model.load_adapter(args.attacker_model_path, adapter_name="attacker")
        logging.info("Attacker adapter loaded.")

        # 5. 加载 Anonymizer 适配器
        logging.info(f"Loading Anonymizer ADAPTER from: {args.anonymizer_model_path}")
        model.load_adapter(args.anonymizer_model_path, adapter_name="anonymizer")
        logging.info("Anonymizer adapter loaded.")
        
        # 6. 创建 *一个* Pipeline
        pipe_device = args.device if "auto" not in str(model.hf_device_map) else None
        shared_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipe_device
        )
        logging.info(f"SHARED pipeline created successfully.")

    except Exception as e:
        logging.error(f"Failed to build model or pipelines: {e}", exc_info=True)
        sys.exit(1)

    # --- 终止符 ---
    terminator_ids = [tokenizer.eos_token_id]
    special_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
    for tok in special_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id != tokenizer.unk_token_id and tok_id is not None:
             terminator_ids.append(tok_id)
    terminator_ids = list(set(terminator_ids))
    logging.info(f"Using terminators (IDs): {terminator_ids}")

    # --- 串行处理 ---
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with SFT models...")
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "skipped_invalid_attribute": 0, "unknown_fail": 0}

    for record_id, record in tqdm(records_to_process, desc="Anonymizing profiles"):
        status = "unknown_fail"
        try:
            try:
                personality = record.get("personality")
                response = str(record.get("response"))
            except Exception as e:
                logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
                status = "skipped_data_read_error"
                record["anonymization_meta"] = {"status": status, "record_id": record_id}
                results.append(record); counters[status] += 1; continue

            if not all([personality, response]):
                logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
                status = "skipped_incomplete_data"
                record["anonymization_meta"] = {"status": status, "record_id": record_id}
                results.append(record); counters[status] += 1; continue

            if not isinstance(personality, dict):
                logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
                status = "skipped_invalid_personality"
                record["anonymization_meta"] = {"status": status, "record_id": record_id}
                results.append(record); counters[status] += 1; continue
            
            anonymized_response, meta = adversarial_anonymization_sft(
                pipe=shared_pipe, 
                original_response=response,
                true_personality=personality,
                terminator_ids=terminator_ids,
                max_iterations=args.max_iterations,
                record_id=record_id,
                protect_attribute=args.protect_attribute,
                age_tolerance=args.age_tolerance,
                min_string_match_len=args.min_string_match_len
            )
            record["anonymized_response"] = anonymized_response
            record["anonymization_meta"] = meta
            record["anonymization_meta"]["record_id"] = record_id
            results.append(record)
            status = meta.get('status', 'unknown_fail')
            counters[status] = counters.get(status, 0) + 1
            logging.info(f"[Record {record_id}] Finished processing. Status: {status}")

        except Exception as exc:
            logging.error(f"[Record {record_id}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1
            if "anonymization_meta" not in record:
                record["anonymization_meta"] = {"status": "unknown_fail", "error": str(exc), "record_id": record_id}
                results.append(record)


    # --- 写入输出文件 ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    results.sort(key=lambda r: r.get("anonymization_meta", {}).get("record_id", float('inf')))

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
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()