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

# --- 1. 匿名化LLM提示词 ---
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
Then, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
"""

# --- 2. 攻击者LLM提示词 ---
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

def generate_attacker_user_prompt(user_response: str, attributes_to_protect: List[str]) -> str:
    """
    根据要保护的属性动态生成攻击者的用户提示。
    """
    attributes_list_str = ", ".join(attributes_to_protect)
    json_keys_str = ", ".join([f'"{attr}"' for attr in attributes_to_protect])

#     return f"""
# Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

# [Comment]:
# {user_response}

# Your response MUST follow this format exactly:

# Inference: Provide your detailed reasoning for the following attribute guess - **{attributes_list_str}**.

# Guess: A single JSON object containing your best guess for the following key: **{json_keys_str}**.

# **JSON Value Rules:**
# 1. age: MUST be an accurate integer (e.g., 35, not "35" or "30-40").
# 2. sex: MUST be one of the following exact strings: "male" or "female".
# 3. relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced" or "widowed".
# 4. income_level: MUST be one of the following exact strings: "low", "middle", "high" or "very high".
# 5. city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China").
# 6. education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner").
# """

    return f"""
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - **{attributes_list_str}**.

Guess: A single JSON object containing your best guess for the following key: **{json_keys_str}**.
"""

# --- 3. 辅助函数 ---
def build_pipeline(
    model_name: str, 
    device: Optional[str], 
    dtype: Optional[str], 
    gpu_memory_utilization: float,
    use_fast_tokenizer: bool,
    low_cpu_mem_usage: bool,
    attn_implementation: str,
    cuda_alloc_conf: str
):
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

    # 使用传入的参数
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    kwargs = {"device_map": "auto"} if device is None else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=low_cpu_mem_usage, 
        attn_implementation=attn_implementation, 
        **kwargs
    )
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device else None)
    
    # 使用传入的参数
    if cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf)
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))
    return gen_pipe, tokenizer

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

def call_anonymizer(
    pipe, 
    user_response: str, 
    feedback: str, 
    terminator_ids: List[int], 
    record_id: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float
) -> str:
    user_content = PROMPT_ANONYMIZER_USER.format(
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{prompt}\n--- End ---")
    
    # 使用传入的参数
    do_sample = temperature > 0.0
    outputs = pipe(
        prompt, 
        max_new_tokens=max_new_tokens, 
        eos_token_id=terminator_ids, 
        do_sample=do_sample, 
        temperature=temperature if do_sample else None, 
        top_p=top_p if do_sample else None, 
        pad_token_id=pipe.tokenizer.eos_token_id, 
        return_full_text=False
    )
    
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT ---\n{out}\n--- End ---")
    parts = out.split('#', 1)
    explanation = parts[0].strip() if len(parts) == 2 else "No explanation provided (LLM failed to follow format)."
    final_response = parts[1].strip().strip('"').strip() if len(parts) == 2 else out.strip().strip('"').strip()
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION ---\n{explanation}\n--- End ---")
    return final_response

def call_attacker(
    pipe, 
    user_response: str, 
    terminator_ids: List[int], 
    record_id: int, 
    attributes_to_protect: List[str],
    temperature: float,
    max_new_tokens: int,
    top_p: float
) -> Dict[str, Any]:
    
    # 动态生成 user_content
    user_content = generate_attacker_user_prompt(
        user_response=str(user_response),
        attributes_to_protect=attributes_to_protect
    )
    
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{prompt}\n--- End ---")
    
    # 使用传入的参数
    do_sample = temperature > 0.0
    outputs = pipe(
        prompt, 
        max_new_tokens=max_new_tokens, 
        eos_token_id=terminator_ids, 
        do_sample=do_sample, 
        temperature=temperature if do_sample else None, 
        top_p=top_p if do_sample else None, 
        pad_token_id=pipe.tokenizer.eos_token_id, 
        return_full_text=False
    )
    
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ATTACKER FULL OUTPUT ---\n{out}\n--- End ---")
    parsed_data = parse_attacker_output(out)
    logging.debug(f"[Record {record_id}] --- ATTACKER PARSED ---\n{parsed_data}\n--- End ---")
    return parsed_data


def compare_profiles(
    true_profile: Dict[str, Any], 
    guessed_profile: Dict[str, Any], 
    attributes_to_protect: List[str],
    age_tolerance: int,
    min_string_match_len: int
) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles，但仅限于指定的属性"""
    leaked_attributes: List[str] = []
    def _log(key, guess, true): return f"{key} (guessed: '{guess}', true: '{true}')"

    for key in attributes_to_protect:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)

        if key == "age":
            if isinstance(true_val, int) and isinstance(guess_val, int):
                # 使用传入的参数
                if abs(true_val - guess_val) <= age_tolerance: 
                    leaked_attributes.append(_log("age", guess_val, true_val))
        
        elif key in ["sex", "relationship_status"]:
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                if true_val.strip().lower() == guess_val.strip().lower(): 
                    leaked_attributes.append(_log(key, guess_val, true_val))
        
        # 适用于所有其他字符串类型
        else: 
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
                # 使用传入的参数
                if len(guess_s) < min_string_match_len: continue # 忽略太短的猜测
                if (guess_s in true_s) or (true_s in guess_s): 
                    leaked_attributes.append(_log(key, guess_val, true_val))

    logging.debug(f"Rule-based comparison (subset: {attributes_to_protect}) complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- 4. 核心逻辑 ---
def adversarial_anonymization(
    pipe,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int,
    record_id: int,
    protect_attribute: Optional[str],
    anonymizer_temp: float,
    anonymizer_max_tokens: int,
    anonymizer_top_p: float,
    attacker_temp: float,
    attacker_max_tokens: int,
    attacker_top_p: float,
    age_tolerance: int,
    min_string_match_len: int
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""

    # 定义所有可能的有效属性
    CORE_PII_ATTRIBUTES_ALL = {"age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status", "health_issue"}
    
    attributes_to_protect: List[str]

    if protect_attribute:
        # 如果指定了单个属性
        if protect_attribute in CORE_PII_ATTRIBUTES_ALL:
            attributes_to_protect = [protect_attribute]
            logging.info(f"[Record {record_id}] Single attribute protection enabled for: '{protect_attribute}'")
        else:
            # 如果指定的属性无效
            logging.error(f"[Record {record_id}] Invalid attribute '{protect_attribute}' specified for protection. Aborting record.")
            meta = {"status": "skipped_invalid_attribute", "error": f"Invalid --protect_attribute: {protect_attribute}"}
            return original_response, meta
    else:
        # 默认行为：保护所有属性
        attributes_to_protect = sorted(list(CORE_PII_ATTRIBUTES_ALL))
        logging.info(f"[Record {record_id}] Full attribute protection enabled for: {attributes_to_protect}")

    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        # 传递攻击者参数
        parsed_attack = call_attacker(
            pipe, original_response, terminator_ids, record_id, attributes_to_protect,
            temperature=attacker_temp,
            max_new_tokens=attacker_max_tokens,
            top_p=attacker_top_p
        )
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始裁判 ---
    # 传递评判参数
    leaked = compare_profiles(
        true_personality, attacker_guess, attributes_to_protect,
        age_tolerance=age_tolerance,
        min_string_match_len=min_string_match_len
    )
    meta["final_leaked_attributes"] = leaked
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        # return current_anonymized_response, meta
    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            # 传递匿名化器参数
            current_anonymized_response = call_anonymizer(
                pipe, current_anonymized_response, feedback, terminator_ids, record_id,
                temperature=anonymizer_temp,
                max_new_tokens=anonymizer_max_tokens,
                top_p=anonymizer_top_p
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta # 返回上一次的响应

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            # 传递攻击者参数
            parsed_attack = call_attacker(
                pipe, current_anonymized_response, terminator_ids, record_id, attributes_to_protect,
                temperature=attacker_temp,
                max_new_tokens=attacker_max_tokens,
                top_p=attacker_top_p
            )
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            continue # 继续下一次循环，使用旧的 feedback

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        # 传递评判参数
        leaked = compare_profiles(
            true_personality, attacker_guess, attributes_to_protect,
            age_tolerance=age_tolerance,
            min_string_match_len=min_string_match_len
        )
        
        all_unknown = False 
        if attacker_guess and isinstance(attacker_guess, dict):
            all_unknown = True 
            
            for key in attributes_to_protect:
                guess_val = attacker_guess.get(key)
                
                normalized_guess = ""
                if isinstance(guess_val, str):
                    normalized_guess = guess_val.strip().lower()
                
                if normalized_guess != "unknown":
                    all_unknown = False
                    break 
        
        if all_unknown:
            logging.info(f"{iteration_log_prefix} Success! All attributes guessed as 'unknown'.")
            meta["status"] = "success" 
            meta["final_leaked_attributes"] = [] 
            return current_anonymized_response, meta

        logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
        meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

def process_record(
    pipe, 
    data: Dict[str, Any], 
    max_iterations: int, 
    record_id: int, 
    terminator_ids: List[int], 
    protect_attribute: Optional[str],
    anonymizer_temp: float,
    anonymizer_max_tokens: int,
    anonymizer_top_p: float,
    attacker_temp: float,
    attacker_max_tokens: int,
    attacker_top_p: float,
    age_tolerance: int,
    min_string_match_len: int
) -> Dict[str, Any]:
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

    # 传递所有参数
    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        original_response=response,
        true_personality=personality,
        terminator_ids=terminator_ids,
        max_iterations=max_iterations,
        record_id=record_id,
        protect_attribute=protect_attribute,
        anonymizer_temp=anonymizer_temp,
        anonymizer_max_tokens=anonymizer_max_tokens,
        anonymizer_top_p=anonymizer_top_p,
        attacker_temp=attacker_temp,
        attacker_max_tokens=attacker_max_tokens,
        attacker_top_p=attacker_top_p,
        age_tolerance=age_tolerance,
        min_string_match_len=min_string_match_len
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的回答进行匿名化")
    
    # --- 基础设置 ---
    g_basic = parser.add_argument_group("Basic Paths & Limits")
    g_basic.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    g_basic.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    g_basic.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    g_basic.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    g_basic.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    g_basic.add_argument("--log_file", type=str, default="anonymizer_local.log", help="日志文件路径")
    g_basic.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    # --- 模型加载参数 ---
    g_model = parser.add_argument_group("Model Loading Parameters")
    g_model.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", help="Hugging Face 模型名")
    g_model.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    g_model.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    g_model.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    g_model.add_argument("--use_fast_tokenizer", default=True, action=argparse.BooleanOptionalAction, help="是否使用 'fast' tokenizer")
    g_model.add_argument("--low_cpu_mem_usage", default=True, action=argparse.BooleanOptionalAction, help="是否启用 'low_cpu_mem_usage'")
    g_model.add_argument("--attn_implementation", type=str, default="sdpa", help="注意力实现 (例如 'sdpa', 'flash_attention_2')")
    g_model.add_argument("--cuda_alloc_conf", type=str, default="expandable_segments:True", help="PYTORCH_CUDA_ALLOC_CONF 的值 (设置为空字符串以禁用)")
    g_model.add_argument("--terminator_tokens", type=str, nargs='+', default=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"], help="用于提前停止的额外 terminator token 列表")

    # --- 核心逻辑参数 ---
    g_logic = parser.add_argument_group("Adversarial Logic Parameters")
    g_logic.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数")
    g_logic.add_argument("--protect_attribute", type=str, default=None, 
                          help="可选：指定要保护的单个属性（例如 'age', 'sex'）。如果未设置，则保护所有8个属性。")

    # --- 匿名化器 (Anonymizer) LLM 参数 ---
    g_anon = parser.add_argument_group("Anonymizer LLM Parameters")
    g_anon.add_argument("--anonymizer_temperature", type=float, default=0.5, help="匿名化器的 temperature")
    g_anon.add_argument("--anonymizer_max_new_tokens", type=int, default=512, help="匿名化器生成的最大 token 数")
    g_anon.add_argument("--anonymizer_top_p", type=float, default=0.9, help="匿名化器的 top_p")

    # --- 攻击者 (Attacker) LLM 参数 ---
    g_atk = parser.add_argument_group("Attacker LLM Parameters")
    g_atk.add_argument("--attacker_temperature", type=float, default=0.1, help="攻击者的 temperature")
    g_atk.add_argument("--attacker_max_new_tokens", type=int, default=1024, help="攻击者生成的最大 token 数")
    g_atk.add_argument("--attacker_top_p", type=float, default=0.9, help="攻击者的 top_p")

    # --- 评判 (Judge) 逻辑参数 ---
    g_judge = parser.add_argument_group("Judge Logic Parameters")
    g_judge.add_argument("--age_tolerance", type=int, default=3, help="评判 'age' 时的容忍范围 (± 3 年)")
    g_judge.add_argument("--min_string_match_len", type=int, default=3, help="评判字符串属性 (如 city) 时的最小匹配长度")

    
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    logging.info(f"Run arguments: {vars(args)}") # 记录所有参数

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 ---
    try:
        # 传递所有模型加载参数
        gen_pipe, tokenizer = build_pipeline(
            model_name=args.model_name, 
            device=args.device, 
            dtype=args.dtype, 
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_fast_tokenizer=args.use_fast_tokenizer,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=args.attn_implementation,
            cuda_alloc_conf=args.cuda_alloc_conf
        )
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    
    # 使用传入的参数
    terminator_ids = [tokenizer.eos_token_id] + \
                     [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in args.terminator_tokens] 
                      if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    if args.protect_attribute:
        logging.info(f"*** SINGLE ATTRIBUTE MODE: Protecting ONLY '{args.protect_attribute}' ***")
    else:
        logging.info(f"*** FULL PROTECTION MODE: Protecting all 8 PII attributes ***")

    # --- 串行处理 ---
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "skipped_invalid_attribute": 0, "unknown_fail": 0}
    
    # 更新 _task 定义以传递所有参数
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(
            gen_pipe, rec, 
            args.max_iterations, 
            rec_idx, 
            terminator_ids, 
            args.protect_attribute,
            anonymizer_temp=args.anonymizer_temperature,
            anonymizer_max_tokens=args.anonymizer_max_new_tokens,
            anonymizer_top_p=args.anonymizer_top_p,
            attacker_temp=args.attacker_temperature,
            attacker_max_tokens=args.attacker_max_new_tokens,
            attacker_top_p=args.attacker_top_p,
            age_tolerance=args.age_tolerance,
            min_string_match_len=args.min_string_match_len
        )
    
    for i, rec in enumerate(tqdm(records_to_process, desc="Anonymizing profiles")):
        try:
            processed_record = _task(i, rec)
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] = counters.get(status, 0) + 1
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an unhandled exception: {exc}", exc_info=True)
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
    for status, count in sorted(counters.items()):
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()