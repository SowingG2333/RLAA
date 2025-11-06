import os
import json
import argparse
import logging
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import requests
from tqdm import tqdm

API_KEY = os.getenv("API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"

MAX_RETRIES = 3
RETRY_DELAY = 5

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

# 日志配置
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler("generate_sft_data.log", 'w', 'utf-8'),
    logging.StreamHandler(sys.stdout)
])
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 辅助函数
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
    return "No Inference block found." # 或者返回空字符串

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

def call_deepseek_api(messages: List[Dict[str, str]], model_name: str, max_tokens: int, temperature: float, record_id: int, purpose: str) -> Optional[str]:
    """通用 DeepSeek API 调用函数，包含重试逻辑"""
    if not API_KEY: logging.error("API_KEY not set."); return None
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
    payload = json.dumps({"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    url = "https://api.deepseek.com/chat/completions"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content')
                if content: logging.debug(f"[Record {record_id}] DeepSeek API call for {purpose} successful."); return content.strip()
                else: logging.warning(f"[Record {record_id}] DeepSeek API response for {purpose} missing content.")
            else: logging.warning(f"[Record {record_id}] DeepSeek API response for {purpose} missing choices: {result}")
            return None
        except requests.exceptions.Timeout:
             logging.warning(f"[Record {record_id}] DeepSeek API call for {purpose} timed out (attempt {attempt + 1}/{MAX_RETRIES}).")
             if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt + 1))
             else: logging.error(f"[Record {record_id}] DeepSeek API call for {purpose} timed out after {MAX_RETRIES} attempts."); return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"[Record {record_id}] DeepSeek API call for {purpose} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt + 1))
            else: logging.error(f"[Record {record_id}] DeepSeek API call for {purpose} failed after {MAX_RETRIES} attempts."); return None
        except Exception as e:
             logging.error(f"[Record {record_id}] Unexpected error calling DeepSeek API for {purpose}: {e}", exc_info=True); return None

def generate_teacher_data_for_record(record: Dict[str, Any], record_id: int) -> Optional[Dict[str, str]]:
    """为单个记录生成教师数据"""
    original_comment = record.get("response")
    if not original_comment:
        logging.warning(f"[Record {record_id}] Skipping record due to missing 'response' field.")
        return None

    sft_record = {"original_comment": original_comment}

    # 1. Call DeepSeek Attacker
    attacker_messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=original_comment)}
    ]
    attacker_output_teacher = call_deepseek_api(
        messages=attacker_messages, model_name=DEEPSEEK_MODEL,
        max_tokens=1024, temperature=0.1, record_id=record_id, purpose="Teacher Attacker"
    )
    if not attacker_output_teacher:
        logging.error(f"[Record {record_id}] Failed to get teacher attacker output.")
        return None # Skip record if attacker fails
    sft_record["attacker_output_teacher"] = attacker_output_teacher

    # 2. Extract Attacker Inference
    attacker_inference_teacher = extract_inference(attacker_output_teacher)
    sft_record["attacker_inference_teacher"] = attacker_inference_teacher

    # 3. Call DeepSeek Anonymizer
    anonymizer_messages = [
            {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
            {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
                user_response=original_comment,
                feedback=attacker_inference_teacher  # <-- This line is corrected
            )}
        ]
    anonymizer_full_output = call_deepseek_api(
        messages=anonymizer_messages, model_name=DEEPSEEK_MODEL,
        max_tokens=512, temperature=0.5, record_id=record_id, purpose="Teacher Anonymizer"
    )
    if not anonymizer_full_output:
        logging.error(f"[Record {record_id}] Failed to get teacher anonymizer output.")
        return None

    # 4. Extract Anonymized Text
    anonymizer_output_teacher = extract_anonymized_text(anonymizer_full_output)
    sft_record["anonymizer_output_teacher"] = anonymizer_output_teacher

    logging.info(f"[Record {record_id}] Successfully generated teacher data.")
    return sft_record

def main():
    global DEEPSEEK_MODEL
    parser = argparse.ArgumentParser(description="Generate SFT data using DeepSeek FgAA as teacher.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (e.g., train.jsonl)")
    parser.add_argument("--output_file", type=str, default="sft_teacher_data.jsonl", help="Path to save the generated SFT data")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process")
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel workers for API calls")
    parser.add_argument("--deepseek_model", type=str, default=DEEPSEEK_MODEL, help="DeepSeek model name for teacher")

    args = parser.parse_args()

    DEEPSEEK_MODEL = args.deepseek_model

    if not API_KEY:
        logging.error("DeepSeek API key not found. Set API_KEY environment variable.")
        sys.exit(1)

    all_records = load_jsonl(args.input_file)
    records_to_process = all_records[:args.limit] if args.limit else all_records

    if not records_to_process:
        logging.info("No records to process.")
        return

    logging.info(f"Starting SFT data generation for {len(records_to_process)} records using {args.workers} workers...")
    logging.info(f"Teacher Model: {DEEPSEEK_MODEL}")

    generated_data = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(generate_teacher_data_for_record, record, i): i
                        for i, record in enumerate(records_to_process)}

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
            for item in generated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Successfully generated {len(generated_data)} SFT records and saved to '{args.output_file}'")
    except Exception as e:
        logging.error(f"Failed to save SFT data to '{args.output_file}': {e}", exc_info=True)

    logging.info("SFT data generation complete.")

if __name__ == "__main__":
    main()