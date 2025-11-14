import os
import json
import argparse
import logging
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# [MODIFIED] Imports to match FgAA_API.py
from openai import OpenAI
from tqdm import tqdm

# [MODIFIED] Globals to match FgAA_API.py
LLM_MODEL = "deepseek-chat"
GLOBAL_CLIENT: Optional[OpenAI] = None
MAX_RETRIES = 3
RETRY_DELAY = 5

# --- 1. 匿名化LLM提示词 (Unchanged, matches FgAA_API.py) ---
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

# --- 2. 攻击者LLM提示词 (Logic imported from FgAA_API.py) ---
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

# [NEW] Copied directly from FgAA_API.py
CORE_PII_ATTRIBUTES_ALL = {"age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status", "health_issue"}

# [NEW] Copied directly from FgAA_API.py
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

# [DELETED] Hardcoded PROMPT_ATTACKER_USER removed

# --- 3. 日志和辅助函数 ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler("generate_sft_data.log", 'w', 'utf-8'),
    logging.StreamHandler(sys.stdout)
])
# [MODIFIED] Supress openai/httpx logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载数据"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping line {i+1} due to JSON decode error: {e}")
        logging.info(f"Loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{filepath}'")
        sys.exit(1)
    return data

def extract_inference(attacker_output: str) -> str:
    """从攻击者输出中提取 Inference 部分"""
    match = re.search(r"Inference:(.*?)(Guess:|$)", attacker_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    logging.warning("Could not find 'Inference:' block in attacker output.")
    return "No Inference block found."

def extract_anonymized_text(anonymizer_output: str) -> str:
    """从匿名化输出中提取实际的匿名化文本"""
    parts = anonymizer_output.split('#', 1)
    if len(parts) == 2:
        anonymized_text = parts[1].strip().strip('"').strip()
        anonymized_text = anonymized_text.split("<|eot_id|>")[0].strip()
        anonymized_text = anonymized_text.split("<|end_of_text|>")[0].strip()
        return anonymized_text
    logging.warning("Anonymizer output did not contain '#'. Returning raw output.")
    return anonymizer_output.strip().strip('"').strip() 

# [NEW] Replaced requests-based function with FgAA_API.py's openai-based function
def call_deepseek_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
    top_p: float, # Added top_p
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
                # Return None instead of raising to allow SFT script to skip
                return None
    
    logging.error(f"{log_prefix} API call failed after all retries and returned no content.")
    return None


# [MODIFIED] Core logic function updated
def generate_teacher_data_for_record(
    client: OpenAI, 
    model: str, 
    record: Dict[str, Any], 
    record_id: int
) -> Optional[Dict[str, Any]]: # [MODIFIED] Returns full record or None
    """为单个记录生成教师数据"""
    
    # [MODIFIED] Check for both response and personality, as FgAA_API.py does
    original_comment = record.get("response")
    personality = record.get("personality")
    
    if not original_comment or not personality or not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipping record due to missing/invalid 'response' or 'personality' field.")
        return None

    # [MODIFIED] Use dynamic attacker prompt generation
    # Mimic FgAA_API.py's default behavior (protect all attributes)
    attributes_to_protect = sorted(list(CORE_PII_ATTRIBUTES_ALL))
    attacker_user_content = generate_attacker_user_prompt(original_comment, attributes_to_protect)

    # 1. Call DeepSeek Attacker
    attacker_messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": attacker_user_content} # [MODIFIED]
    ]
    attacker_output_teacher = call_deepseek_api(
        client=client, model=model, messages=attacker_messages,
        max_tokens=1024, temperature=0.1, top_p=0.9, # [MODIFIED] Added top_p
        record_id=record_id, purpose="Teacher Attacker"
    )
    if not attacker_output_teacher:
        logging.error(f"[Record {record_id}] Failed to get teacher attacker output.")
        return None # Skip record if attacker fails
    
    # [MODIFIED] Add to existing record instead of creating new sft_record
    record["attacker_output_teacher"] = attacker_output_teacher

    # 2. Extract Attacker Inference
    attacker_inference_teacher = extract_inference(attacker_output_teacher)
    record["attacker_inference_teacher"] = attacker_inference_teacher # [MODIFIED]

    # 3. Call DeepSeek Anonymizer
    anonymizer_messages = [
            {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
            {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
                user_response=original_comment,
                feedback=attacker_inference_teacher
            )}
        ]
    anonymizer_full_output = call_deepseek_api(
        client=client, model=model, messages=anonymizer_messages,
        max_tokens=512, temperature=0.5, top_p=0.9, # [MODIFIED] Added top_p
        record_id=record_id, purpose="Teacher Anonymizer"
    )
    if not anonymizer_full_output:
        logging.error(f"[Record {record_id}] Failed to get teacher anonymizer output.")
        return None

    # 4. Extract Anonymized Text
    anonymizer_output_teacher = extract_anonymized_text(anonymizer_full_output)
    record["anonymizer_output_teacher"] = anonymizer_output_teacher # [MODIFIED]

    logging.info(f"[Record {record_id}] Successfully generated teacher data.")
    return record # [MODIFIED] Return the whole, modified record

def main():
    global GLOBAL_CLIENT, LLM_MODEL # [MODIFIED]
    
    parser = argparse.ArgumentParser(description="Generate SFT data using DeepSeek FgAA as teacher.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (e.g., train.jsonl)")
    parser.add_argument("--output_file", type=str, default="sft_teacher_data.jsonl", help="Path to save the generated SFT data")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process")
    parser.add_argument("--workers", type=int, default=100, help="Number of parallel workers for API calls")
    
    # [MODIFIED] Standardized args
    parser.add_argument("--model", type=str, default=LLM_MODEL, help=f"LLM 模型名称 (default: {LLM_MODEL})")
    parser.add_argument("--api_key", type=str, default=None, help="覆盖 API 密钥 (默认: 使用 API_KEY 环境变量)")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1", help="覆盖 API 基础 URL")

    args = parser.parse_args()

    LLM_MODEL = args.model # [MODIFIED]

    # [NEW] Client initialization copied from FgAA_API.py
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
    
    # [DELETED] Old API_KEY check

    all_records = load_jsonl(args.input_file)
    records_to_process = all_records[:args.limit] if args.limit else all_records

    if not records_to_process:
        logging.info("No records to process.")
        return

    logging.info(f"Starting SFT data generation for {len(records_to_process)} records using {args.workers} workers...")
    logging.info(f"Teacher Model: {LLM_MODEL}") # [MODIFIED]

    generated_data = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # [MODIFIED] Updated submit call
        future_to_id = {
            executor.submit(
                generate_teacher_data_for_record, 
                GLOBAL_CLIENT, 
                LLM_MODEL, 
                record, 
                i
            ): i
            for i, record in enumerate(records_to_process)
        }

        for future in tqdm(as_completed(future_to_id), total=len(records_to_process), desc="Generating Teacher Data"):
            record_id = future_to_id[future]
            try:
                result = future.result()
                if result: # Only append successful results
                    generated_data.append(result)
            except Exception as exc:
                logging.error(f'[Record {record_id}] generated an exception: {exc}', exc_info=True)

    # 保存生成的 SFT 数据
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in generated_data: # [MODIFIED] This 'item' is now the full record
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Successfully generated {len(generated_data)} SFT records and saved to '{args.output_file}'")
    except Exception as e:
        logging.error(f"Failed to save SFT data to '{args.output_file}': {e}", exc_info=True)

    logging.info("SFT data generation complete.")

if __name__ == "__main__":
    main()