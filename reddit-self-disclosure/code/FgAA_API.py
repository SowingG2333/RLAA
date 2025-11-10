import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from openai import OpenAI
from tqdm import tqdm

LLM_MODEL = "deepseek-chat"
GLOBAL_CLIENT: Optional[OpenAI] = None
MAX_RETRIES = 3
RETRY_DELAY = 5

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

    return f"""
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - **{attributes_list_str}**.

Guess: A single JSON object containing your best guess for the following key: **{json_keys_str}**.
"""

# --- 3. 辅助函数 ---
def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析推理和猜测(JSON)"""
    inference = ""
    guess_json = {}
    inference_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if inference_match: inference = inference_match.group(1).strip()
    
    # [修改] 使用更健壮的 JSON 提取器
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
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
            logging.debug("Parsed response directly as JSON.")
            inference = "No inference text found (parsed directly as JSON)."
        except json.JSONDecodeError:
            logging.warning(f"No JSON found: {response_text[:100]}..."); 
            guess_json = {"error": "No JSON found"}

    return {"inference": inference, "guess_json": guess_json}

# --- API 调用辅助函数 ---
def call_deepseek_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float,
    record_id: int, 
    purpose: str,
    use_json_mode: bool = False
) -> str:
    """通用的 DeepSeek API 调用函数，包含重试逻辑"""
    log_prefix = f"[Record {record_id}][{purpose}]"
    
    payload_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    if use_json_mode:
        payload_kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            logging.debug(f"{log_prefix} API call attempt {attempt + 1}/{MAX_RETRIES}...")
            completion = client.chat.completions.create(**payload_kwargs)
            content = completion.choices[0].message.content
            
            if content:
                logging.debug(f"{log_prefix} API call successful.")
                return content.strip()
            else:
                logging.warning(f"{log_prefix} API response missing content.")
                
        except Exception as e:
            logging.warning(f"{log_prefix} API call failed (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY * (attempt + 1)
                logging.info(f"{log_prefix} Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"{log_prefix} API call failed after {MAX_RETRIES} attempts.")
                raise e # 抛出异常以使 process_record 捕获
    
    raise Exception(f"{log_prefix} API call failed after all retries and returned no content.")

def call_anonymizer(
    client: OpenAI,
    model: str,    
    user_response: str, 
    feedback: str, 
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
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{user_content}\n--- End ---")
    
    # 调用 API 辅助函数
    out = call_deepseek_api(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        record_id=record_id,
        purpose="Anonymizer"
    )
    
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT ---\n{out}\n--- End ---")
    parts = out.split('#', 1)
    explanation = parts[0].strip() if len(parts) == 2 else "No explanation provided (LLM failed to follow format)."
    final_response = parts[1].strip().strip('"').strip() if len(parts) == 2 else out.strip().strip('"').strip()
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION ---\n{explanation}\n--- End ---")
    return final_response

def call_attacker(
    client: OpenAI,         
    model: str,             
    user_response: str, 
    record_id: int, 
    attributes_to_protect: List[str],
    temperature: float,
    max_new_tokens: int,
    top_p: float
) -> Dict[str, Any]:
    
    user_content = generate_attacker_user_prompt(
        user_response=str(user_response),
        attributes_to_protect=attributes_to_protect
    )
    
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{user_content}\n--- End ---")
    
    # 调用 API 辅助函数
    # 注意：我们不在这里强制使用 json_mode，因为提示要求有 "Inference:" 文本
    out = call_deepseek_api(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        record_id=record_id,
        purpose="Attacker"
    )
    
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

    # 确保我们只检查请求的属性
    for key in attributes_to_protect:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        
        # 为 "unknown" 添加更健壮的检查
        if guess_val is None or (isinstance(guess_val, str) and guess_val.strip().lower() == "unknown"):
            continue

        if key == "age":
            if isinstance(true_val, int) and isinstance(guess_val, int):
                if abs(true_val - guess_val) <= age_tolerance: 
                    leaked_attributes.append(_log("age", guess_val, true_val))
        
        elif key in ["sex", "relationship_status", "income_level"]: # [修改] 添加了 income_level
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                if true_val.strip().lower() == guess_val.strip().lower(): 
                    leaked_attributes.append(_log(key, guess_val, true_val))
        
        # 为 'health_issue' 添加了列表比较逻辑（来自 eval_single.py）
        elif key == "health_issue":
            if (isinstance(true_val, list) and true_val and 
                isinstance(guess_val, str) and guess_val):
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= min_string_match_len:
                    for true_item in true_val:
                        if isinstance(true_item, str):
                            true_s = true_item.strip().lower()
                            if (guess_s in true_s) or (true_s in guess_s):
                                leaked_attributes.append(_log(key, guess_val, true_val))
                                break 
            elif (isinstance(true_val, str) and true_val and 
                  isinstance(guess_val, str) and guess_val):
                true_s = true_val.strip().lower()
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= min_string_match_len:
                    if (guess_s in true_s) or (true_s in guess_s): 
                        leaked_attributes.append(_log(key, guess_val, true_val))
        
        # 适用于所有其他字符串类型
        elif key in ["city_country", "birth_city_country", "education", "occupation"]:
            if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
                true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
                if len(guess_s) < min_string_match_len: continue 
                if (guess_s in true_s) or (true_s in guess_s): 
                    leaked_attributes.append(_log(key, guess_val, true_val))

    logging.debug(f"Rule-based comparison (subset: {attributes_to_protect}) complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- 4. 核心逻辑 ---
def adversarial_anonymization(
    client: OpenAI,        
    model: str,            
    original_response: str,
    true_personality: Dict[str, Any],
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
        parsed_attack = call_attacker(
            client, model, original_response, record_id, attributes_to_protect,
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
    leaked = compare_profiles(
        true_personality, attacker_guess, attributes_to_protect,
        age_tolerance=age_tolerance,
        min_string_match_len=min_string_match_len
    )
    meta["final_leaked_attributes"] = leaked
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        # return current_anonymized_response, meta # [修改] 按照本地脚本逻辑，即使成功也继续
    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer(
                client, model, current_anonymized_response, feedback, record_id,
                temperature=anonymizer_temp,
                max_new_tokens=anonymizer_max_tokens,
                top_p=anonymizer_top_p
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta 

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker(
                client, model, current_anonymized_response, record_id, attributes_to_protect,
                temperature=attacker_temp,
                max_new_tokens=attacker_max_tokens,
                top_p=attacker_top_p
            )
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"attacker_error_round_{i}: {e}"
            return current_anonymized_response, meta # [修改] 攻击者失败时退出

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(
            true_personality, attacker_guess, attributes_to_protect,
            age_tolerance=age_tolerance,
            min_string_match_len=min_string_match_len
        )
        
        # [修改] 使用 eval_single.py 中更健壮的 "all_unknown" 检查
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
    client: OpenAI,         
    model: str,             
    data: Dict[str, Any], 
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
) -> Dict[str, Any]:
    """处理单条记录。"""
    record_log_prefix = f"[Record-{record_id}]"
    logging.debug(f"{record_log_prefix} Starting processing.")
    try:
        personality = data.get("personality")
        response = str(data.get("response"))
    except Exception as e:
        logging.error(f"{record_log_prefix} Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not all([personality, response]):
        logging.warning(f"{record_log_prefix} Skipped due to incomplete data.")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data

    if not isinstance(personality, dict):
        logging.warning(f"{record_log_prefix} Skipped: 'personality' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    try:
        anonymized_response, meta = adversarial_anonymization(
            client=client,
            model=model,
            original_response=response,
            true_personality=personality,
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
        logging.info(f"{record_log_prefix} Finished processing. Status: {meta.get('status')}")
    except Exception as e:
        # [新] 捕获来自 call_deepseek_api 的异常
        logging.error(f"{record_log_prefix} Generated an unhandled exception: {e}", exc_info=True)
        data["anonymization_meta"] = {"status": "unknown_fail", "error": str(e)}
        
    return data

def main():
    global GLOBAL_CLIENT, LLM_MODEL
    
    parser = argparse.ArgumentParser(description="使用 DeepSeek API 并行对 JSONL 中的回答进行匿名化")
    
    # --- 基础设置 ---
    g_basic = parser.add_argument_group("Basic Paths & Limits")
    g_basic.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    g_basic.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    g_basic.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    g_basic.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    g_basic.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    g_basic.add_argument("--log_file", type=str, default="anonymizer_api.log", help="日志文件路径")
    g_basic.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    # --- API 和并行参数 ---
    g_api = parser.add_argument_group("API & Parallelism")
    g_api.add_argument("--max_workers", type=int, default=50, help="最大并行工作线程数")
    g_api.add_argument("--model", type=str, default=LLM_MODEL, help=f"LLM 模型名称 (default: {LLM_MODEL})")
    g_api.add_argument("--api_key", type=str, default=None, help="覆盖 API 密钥 (默认: 使用 API_KEY 环境变量)")
    g_api.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1", help="覆盖 API 基础 URL")

    # --- 核心逻辑参数 ---
    g_logic = parser.add_argument_group("Adversarial Logic Parameters")
    g_logic.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数")
    g_logic.add_argument("--protect_attribute", type=str, default=None, 
                         help="可选：指定要保护的单个属性（例如 'age', 'sex'）。如果未设置，则保护所有属性。")

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
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    logging.info(f"Run arguments: {vars(args)}")

    # --- Client/Model 初始化 ---
    LLM_MODEL = args.model
    try:
        GLOBAL_CLIENT = OpenAI(
            api_key=args.api_key or os.environ.get("API_KEY"),
            base_url=args.base_url
        )
        GLOBAL_CLIENT.models.list() # 验证连接
        logging.info(f"OpenAI client initialized for base_url: {args.base_url}")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        logging.error("Please ensure your API_KEY is set via --api_key or API_KEY environment variable.")
        sys.exit(1)

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    
    records_with_ids = []
    for i, line in enumerate(lines):
        if line.strip():
            try:
                records_with_ids.append((i, json.loads(line)))
            except json.JSONDecodeError:
                logging.warning(f"Skipping line {i+1} due to invalid JSON.")

    if args.limit: records_with_ids = records_with_ids[:args.limit]
    
    if not records_with_ids:
        logging.error("No valid records found in input file.")
        sys.exit(0)

    logging.info(f"Starting parallel processing for {len(records_with_ids)} records with model {LLM_MODEL} using up to {args.max_workers} workers...")

    # --- 并行处理 (来自 FgAA_API.py) ---
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "skipped_invalid_attribute": 0, "unknown_fail": 0}
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_record = {
            executor.submit(
                process_record,
                GLOBAL_CLIENT,
                LLM_MODEL,
                record_data,
                args.max_iterations,
                record_id,
                args.protect_attribute,
                args.anonymizer_temperature,
                args.anonymizer_max_new_tokens,
                args.anonymizer_top_p,
                args.attacker_temperature,
                args.attacker_max_new_tokens,
                args.attacker_top_p,
                args.age_tolerance,
                args.min_string_match_len
            ): record_id 
            for record_id, record_data in records_with_ids
        }
        
        for future in tqdm(as_completed(future_to_record), total=len(records_with_ids), desc="Anonymizing profiles"):
            record_id = future_to_record[future]
            try:
                processed_record = future.result()
                results.append(processed_record)
                status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
                counters[status] = counters.get(status, 0) + 1
            except Exception as exc:
                logging.error(f"[Record {record_id}] Generated an unhandled exception in thread: {exc}", exc_info=True)
                # 记录失败，但继续
                results.append({
                    "record_id": record_id, 
                    "anonymization_meta": {"status": "unknown_fail", "error": str(exc)}
                })
                counters["unknown_fail"] += 1

    # --- 结果写入 ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        
        # 按 record_id 排序以保持原始顺序
        results.sort(key=lambda r: r.get("anonymization_meta", {}).get("record_id", r.get("record_id", float('inf'))))

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