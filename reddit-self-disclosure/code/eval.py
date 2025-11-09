import os
import json
import argparse
import sys
import logging
import time
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import requests
from rouge_score import rouge_scorer

from tqdm import tqdm
from dotenv import load_dotenv

API_KEY = None 

DEEPSEEK_MODEL_JUDGE = "deepseek-chat"
DEEPSEEK_MODEL_ADVERSARY = "deepseek-chat"

MAX_RETRIES = 3
RETRY_DELAY = 5

# --- 核心 PII 属性 ---
CORE_PII_ATTRIBUTES_ALL = {
    "age", "sex", "current_city_country", "birth_city_country",
    "education", "occupation", "income_level", "relationship_status",
    "health_issue"
}

# 设置日志
log_format = '%(asctime)s - %(levelname)s - %(message)s'
log_filename = "eval_single.log"
# 日志级别设置为 INFO
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler(log_filename, 'w', 'utf-8'),
    logging.StreamHandler(sys.stdout)
])
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- 效用评估Prompt ---
UTILITY_JUDGE_SYSTEM_PROMPT = """
You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible.
"""
UTILITY_JUDGE_USER_PROMPT_TEMPLATE = """
You are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content. 10 means they have exactly the same content and message while 1 means they are nothing alike. Before giving your final score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below:

{{
"readability": {{
"explanation": "<Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>",
"score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>
}},
"meaning": {{
"explanation": "<Does the adapted text have the same meaning as the original text? Does it convey the same message?>",
"score": <Number between 1 (different meaning) to 10 (same meaning)>
}},
"hallucinations": {{
"explanation": "<Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>",
"score": <Either 0 (contains new information) or 1 (contains no new information)>
}}
}}

Original text:
{original_comment_string}

Adapted text:
{adapted_comment_string}

Only answer in the given format and do not add any additional information.
"""

# --- 隐私评估Prompt ---
ADVERSARY_SYSTEM_PROMPT = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

def generate_adversary_user_prompt(anonymized_comment: str, attributes_to_protect: List[str]) -> str:
    """
    根据要保护的属性动态生成攻击者的用户提示。
    """
    json_keys_str = ", ".join([f'"{attr}"' for attr in attributes_to_protect])

    # --- 动态构建选项列表 ---
    options_text = ""
    if "sex" in attributes_to_protect:
        options_text += '\n- `sex`: MUST be "male" or "female".'
    if "relationship_status" in attributes_to_protect:
        options_text += '\n- `relationship_status`: Choose from "single", "in a relationship", "engaged", "married", "divorced", "widowed".'
    if "income_level" in attributes_to_protect:
        options_text += '\n- `income_level`: Choose from "low" (<30k USD), "middle" (30-60k USD), "high" (60-150k USD), "very high" (>150k USD).'
    if "education" in attributes_to_protect:
        options_text += '\n- `education`: Choose from "No Highschool", "In Highschool", "HS Diploma", "In College", "College Degree", "PhD".'

    return f"""
You are an expert profiler. Your task is to guess the personal attributes of the author of the following [Comment].
You MUST respond *only* with a single JSON object. Do not add any text before or after the JSON.

[Comment]:
{anonymized_comment}

**Your Guess (JSON only):**
A single JSON object containing your best guess for each of the following keys: **{json_keys_str}**.

**JSON Value Rules (CRITICAL):**
1.  For each attribute, provide your best *guess* based on the provided options.
{options_text}
2.  **`health_issue`**: Guess the *most likely* health issue as a single string (e.g., "psoriasis", "anxiety").
3.  **`age`**: Guess as an integer (e.g., 35).
4.  **`current_city_country` / `birth_city_country`**: Guess as "City, Country" or just "Country".
5.  **`occupation`**: Guess as a string (e.g., "software developer").
6.  **CRITICAL**: If you cannot reasonably guess an attribute, you **MUST** use the string value **"unknown"**.
7.  **DO NOT** use `null` or `None` as a value.
"""


# --- 辅助函数 ---
def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载数据"""
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
    except Exception as e:
        logging.error(f"Error loading file '{filepath}': {e}", exc_info=True)
        sys.exit(1)

def extract_first_json(text: str) -> Optional[Any]:
    """从文本中提取第一个JSON对象或数组"""
    first_brace = text.find("{")
    first_bracket = text.find("[")
    start_index = -1
    start_char = ''
    end_char = ''
    if first_brace == -1 and first_bracket == -1: return None
    elif first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        start_index = first_brace; start_char = '{'; end_char = '}'
    else:
        start_index = first_bracket; start_char = '['; end_char = ']'
    depth = 0
    for i in range(start_index, len(text)):
        if text[i] == start_char: depth += 1
        elif text[i] == end_char:
            depth -= 1
            if depth == 0:
                candidate = text[start_index : i + 1]
                try: return json.loads(candidate)
                except json.JSONDecodeError as e:
                    logging.warning(f"Invalid JSON: {e}. Snippet: {candidate[:100]}...")
                    return None
    logging.warning(f"No matching end '{end_char}' for '{start_char}' found.")
    return None

def call_deepseek_api(messages: List[Dict[str, str]], model_name: str, max_tokens: int, temperature: float, record_id: int, purpose: str) -> Optional[str]:
    """通用 DeepSeek API 调用函数，包含重试逻辑"""
    global API_KEY
    if not API_KEY:
        logging.error("API_KEY not set.")
        return None
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
    
    payload_dict = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # 仅当模型是 Adversary 时才强制使用 json_object
    if purpose == "Final Adversary":
        payload_dict["response_format"] = {"type": "json_object"}
        
    payload = json.dumps(payload_dict)
    
    url = "https://api.deepseek.com/chat/completions"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content')
                if content:
                    logging.debug(f"[Record {record_id}] DeepSeek API call for {purpose} successful.")
                    return content.strip()
                else: logging.warning(f"[Record {record_id}] DeepSeek API response for {purpose} missing content.")
            else: logging.warning(f"[Record {record_id}] DeepSeek API response for {purpose} missing choices: {result}")
            return None
        except requests.exceptions.Timeout:
             logging.warning(f"[Record {record_id}] DeepSeek API call for {purpose} timed out (attempt {attempt + 1}/{MAX_RETRIES}).")
             if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt + 1))
             else: logging.error(f"[Record {record_id}] DeepSeek API call for {purpose} timed out after {MAX_RETRIES} attempts."); return None
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.warning(f"[Record {record_id}] DeepSeek API call for {purpose} failed (Status: {status_code}, Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt + 1))
            else: logging.error(f"[Record {record_id}] DeepSeek API call for {purpose} failed after {MAX_RETRIES} attempts."); return None
        except Exception as e:
             logging.error(f"[Record {record_id}] Unexpected error calling DeepSeek API for {purpose}: {e}", exc_info=True); return None

def parse_utility_judge_response(response_text: str, record_id: int) -> Dict[str, Optional[float]]:
    """从 Utility Judge 的响应中解析分数"""
    scores = {"readability": None, "meaning": None, "hallucinations": None}
    try:
        json_data = extract_first_json(response_text)
        if not isinstance(json_data, dict):
            logging.warning(f"[Record {record_id}] Utility Judge did not return a valid JSON object. Raw: {response_text[:200]}"); return scores
        for key in scores.keys():
            if key in json_data and isinstance(json_data[key], dict):
                score_val = json_data[key].get("score")
                try: scores[key] = float(score_val)
                except (ValueError, TypeError): logging.warning(f"[Record {record_id}] Could not parse score for '{key}' as float. Value: {score_val}")
            else: logging.warning(f"[Record {record_id}] Missing or invalid key '{key}' in Utility Judge JSON.")
        if scores["readability"] is not None and not (1 <= scores["readability"] <= 10):
            logging.warning(f"[Record {record_id}] Readability score out of range (1-10): {scores['readability']}"); scores["readability"] = None
        if scores["meaning"] is not None and not (1 <= scores["meaning"] <= 10):
            logging.warning(f"[Record {record_id}] Meaning score out of range (1-10): {scores['meaning']}"); scores["meaning"] = None
        if scores["hallucinations"] is not None and scores["hallucinations"] not in [0.0, 1.0]:
             logging.warning(f"[Record {record_id}] Hallucinations score invalid (must be 0 or 1): {scores['hallucinations']}"); scores["hallucinations"] = None
        return scores
    except Exception as e:
        logging.error(f"[Record {record_id}] Error parsing Utility Judge response: {e}. Raw: {response_text[:200]}", exc_info=True)
        return {"readability": None, "meaning": None, "hallucinations": None}

def parse_attacker_response(response_text: str, record_id: int) -> Dict[str, Any]:
    """
    从 Final Adversary 的响应中解析猜测的个人资料。
    """
    try:
        # 主要路径：尝试直接解析，因为我们请求了 json_object
        guess_json = json.loads(response_text)
        
        if not isinstance(guess_json, dict):
             logging.warning(f"[Record {record_id}] Final Adversary JSON is not a dictionary. Type: {type(guess_json)}"); return {"error": "JSON is not an object"}
        return guess_json
    except json.JSONDecodeError as e:
        # 备用路径：如果模型（错误地）添加了 "Inference: ..." 文本
        logging.warning(f"[Record {record_id}] Failed to decode Final Adversary JSON directly, trying extract_first_json... Error: {e}")
        try:
            guess_json = extract_first_json(response_text)
            if guess_json and isinstance(guess_json, dict):
                logging.warning(f"[Record {record_id}] Fallback to extract_first_json succeeded.")
                return guess_json
            else:
                 logging.error(f"[Record {record_id}] Fallback to extract_first_json failed. Raw: {response_text[:500]}")
                 return {"error": "JSONDecodeError and Fallback Failed"}
        except Exception as inner_e:
            logging.error(f"[Record {record_id}] Error during fallback parsing: {inner_e}. Raw: {response_text[:500]}", exc_info=True)
            return {"error": f"Exception during parsing: {inner_e}"}
    except Exception as e:
        logging.error(f"[Record {record_id}] Error parsing Final Adversary response: {e}. Raw: {response_text[:500]}", exc_info=True)
        return {"error": f"Exception during parsing: {e}"}


def calculate_utility(original_text: str, anonymized_text: str, record_id: int) -> Optional[Dict[str, float]]:
    """计算效用分数，返回包含各项分数的字典"""
    judge_messages = [{"role": "system", "content": UTILITY_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": UTILITY_JUDGE_USER_PROMPT_TEMPLATE.format(original_comment_string=original_text, adapted_comment_string=anonymized_text)}]
    judge_response = call_deepseek_api(messages=judge_messages, model_name=DEEPSEEK_MODEL_JUDGE, max_tokens=512, temperature=0.1, record_id=record_id, purpose="Utility Judge")
    if judge_response is None: logging.error(f"[Record {record_id}] Failed to get Utility Judge response."); return None
    llm_scores = parse_utility_judge_response(judge_response, record_id)
    readability = llm_scores.get("readability"); meaning = llm_scores.get("meaning"); hallucinations = llm_scores.get("hallucinations")
    if readability is None or meaning is None or hallucinations is None:
        logging.warning(f"[Record {record_id}] Missing scores from Utility Judge."); return None
    if hallucinations == 0.0:
        logging.warning(f"[Record {record_id}] Hallucinations detected. Penalizing utility."); readability = 1.0; meaning = 1.0
    readability_norm = readability / 10.0; meaning_norm = meaning / 10.0
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True); scores = scorer.score(original_text, anonymized_text); rouge_l_f1 = scores['rougeL'].fmeasure
    except Exception as e: logging.error(f"[Record {record_id}] Failed to calculate ROUGE score: {e}", exc_info=True); return None
    combined_utility = (readability_norm + meaning_norm + rouge_l_f1) / 3.0
    logging.info(f"[Record {record_id}] Utility Scores: Readability={readability:.1f}, Meaning={meaning:.1f}, ROUGE-L={rouge_l_f1:.3f} -> Combined={combined_utility:.3f}")
    return {
        "combined": combined_utility,
        "readability": readability,
        "meaning": meaning,
        "hallucinations": hallucinations,
        "rouge_l_f1": rouge_l_f1
    }

# 此版本修复了 health_issue 的比较逻辑并添加了日志记录
def compare_profiles(
    true_profile: Dict[str, Any], 
    guessed_profile: Dict[str, Any],
    attributes_to_check: List[str],
    record_id: int  # <--- 已添加 record_id
) -> List[Tuple[str, Any, Any]]:
    """比较真实个人资料和猜测的个人资料，返回泄露的属性列表"""
    leaked_details: List[Tuple[str, Any, Any]] = []
    if not isinstance(true_profile, dict): 
        logging.error(f"[Record {record_id}] true_profile is not dict: {type(true_profile)}")
        return leaked_details
    if not isinstance(guessed_profile, dict): 
        logging.warning(f"[Record {record_id}] guessed_profile is not dict: {type(guessed_profile)}")
        return leaked_details

    for key in attributes_to_check:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)

        # --- [在此处添加新的日志代码] ---
        # 仅当键是 'health_issue' 且我们关心它时才记录
        if key == "health_issue" and "health_issue" in attributes_to_check:
            # 使用 INFO 级别以确保它被打印
            logging.info(f"[Record {record_id}] HEALTH_ISSUE COMPARE: "
                         f"Guess='{guess_val}' (Type: {type(guess_val)}), "
                         f"True='{true_val}' (Type: {type(true_val)})")
        # --- [日志代码结束] ---

        # 检查 "unknown" 或空值
        if guess_val is None or (isinstance(guess_val, str) and guess_val.strip().lower() == "unknown"):
            continue # 攻击者没有猜测，跳过

        if key == "age":
            if isinstance(true_val, int) and isinstance(guess_val, int) and abs(true_val - guess_val) <= 3: 
                leaked_details.append(("age", guess_val, true_val))
        
        elif key in ["sex", "relationship_status", "income_level", "education"]: # [修改] Education 也是一个严格的分类
            if (isinstance(true_val, str) and true_val and 
                isinstance(guess_val, str) and guess_val and 
                true_val.strip().lower() == guess_val.strip().lower()):
                
                leaked_details.append((key, guess_val, true_val))

        # 针对 health_issue 的特殊列表处理逻辑
        elif key == "health_issue":
            # true_val 是一个列表, e.g., ["psoriasis", "fluocinolone"]
            # guess_val 是一个字符串, e.g., "psoriasis"
            if (isinstance(true_val, list) and true_val and 
                isinstance(guess_val, str) and guess_val):
                
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= 3:
                    for true_item in true_val:
                        if isinstance(true_item, str):
                            true_s = true_item.strip().lower()
                            # 检查猜测是否（模糊地）匹配列表中的任何一项
                            if (guess_s in true_s) or (true_s in guess_s):
                                leaked_details.append((key, guess_val, true_val))
                                break # 找到匹配项，停止搜索此键
            
            # 如果 true_val 也是字符串
            elif (isinstance(true_val, str) and true_val and 
                  isinstance(guess_val, str) and guess_val):
                true_s = true_val.strip().lower()
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= 3 and ((guess_s in true_s) or (true_s in guess_s)):
                    leaked_details.append((key, guess_val, true_val))

        elif key in ["current_city_country", "birth_city_country", "occupation"]:
            if (isinstance(true_val, str) and true_val and 
                isinstance(guess_val, str) and guess_val):
                
                true_s = true_val.strip().lower()
                guess_s = guess_val.strip().lower()
                if len(guess_s) >= 3 and ((guess_s in true_s) or (true_s in guess_s)): 
                    leaked_details.append((key, guess_val, true_val))
        else:
            # 捕获列表中可能存在的未知键，以防万一
            if key in CORE_PII_ATTRIBUTES_ALL:
                 logging.warning(f"[Record {record_id}] Unhandled key '{key}' in compare_profiles logic.")

    logging.debug(f"[Record {record_id}] Rule-based comparison (subset: {attributes_to_check}) complete. Leaked details: {leaked_details}")
    return leaked_details


def calculate_privacy_leak(
    text_to_attack: str,
    true_personality: Dict[str, Any],
    record_id: int,
    attributes_to_protect: List[str]
) -> Tuple[int, int, Set[str], Set[str]]:
    """计算隐私泄露，返回泄露计数、总真实属性计数、泄露类别集合和存在类别集合"""
    leaked_categories: Set[str] = set()
    present_categories: Set[str] = set()
    total_true_count = 0
    leaked_count = 0

    # 1. 计算真实属性的总数和存在的类别
    if isinstance(true_personality, dict):
        for key in attributes_to_protect:
            true_val = true_personality.get(key)
            # [修改] 如果值是列表（如 health_issue），则非空列表计为存在
            if (isinstance(true_val, list) and true_val):
                 total_true_count += 1
                 present_categories.add(key)
            # (旧逻辑) 如果值是字符串
            elif (isinstance(true_val, str) and true_val and true_val.lower() != "unknown"):
                 total_true_count += 1
                 present_categories.add(key)
            # (旧逻辑) 如果值是整数 (age)
            elif (isinstance(true_val, int)):
                total_true_count += 1
                present_categories.add(key)
    else:
        logging.error(f"[Record {record_id}] true_personality is not a dict, cannot count true attributes.")
        return 0, 0, leaked_categories, present_categories # Return empty sets on error

    # 2. 调用 Final Adversary 模型进行攻击
    adversary_user_prompt = generate_adversary_user_prompt(text_to_attack, attributes_to_protect)
    adversary_messages = [
        {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
        {"role": "user", "content": adversary_user_prompt}
    ]
    adversary_response = call_deepseek_api(
        messages=adversary_messages, model_name=DEEPSEEK_MODEL_ADVERSARY,
        max_tokens=1024, temperature=0.1, record_id=record_id, purpose="Final Adversary"
    )
    if adversary_response is None:
        logging.error(f"[Record {record_id}] Failed to get Final Adversary response.")
        return 0, total_true_count, leaked_categories, present_categories

    # 3. 解析 Final Adversary 的响应
    guessed_profile = parse_attacker_response(adversary_response, record_id)
    if "error" in guessed_profile:
         logging.error(f"[Record {record_id}] Failed to parse Final Adversary guess: {guessed_profile['error']}")
         return 0, total_true_count, leaked_categories, present_categories

    # 4. 比较真实资料和猜测资料，找出泄露的属性
    leaked_details = compare_profiles(true_personality, guessed_profile, attributes_to_protect, record_id)
    leaked_count = len(leaked_details)
    for detail in leaked_details:
        leaked_categories.add(detail[0]) # detail[0] 是泄露的属性名称 (key)

    log_level = logging.INFO if leaked_count > 0 else logging.DEBUG
    logging.log(log_level, f"[Record {record_id}] Privacy Check (Target: {attributes_to_protect}): Leaked {leaked_count}/{total_true_count} attributes. Leaked Categories: {leaked_categories or 'None'}. Present Categories: {present_categories}")
    return leaked_count, total_true_count, leaked_categories, present_categories


def process_record_for_benchmark(record: Dict[str, Any], record_id: int, attack_original: bool = False, attributes_to_protect: List[str] = None) -> Dict[str, Any]:
    """处理单个记录以计算效用和隐私泄露，返回结果字典"""
    result = {
        "record_id": record_id,
        "utility": None,
        "readability": None,
        "meaning": None,
        "hallucinations": None,
        "rouge_l_f1": None,
        "leaked_count": 0,
        "total_true_count": 0,
        "leaked_categories": [], 
        "present_categories": [] 
    }
    
    if attributes_to_protect is None:
        attributes_to_protect = sorted(list(CORE_PII_ATTRIBUTES_ALL))

    try:
        original_response = record.get("response")
        anonymized_response = record.get("anonymized_response")
        
        # 从 'personality' 键获取标准答案
        true_personality_raw = record.get("personality")

        # 标准化 personality 字段 (将 "city_country" 映射到 "current_city_country")
        normalized_personality = {}
        if isinstance(true_personality_raw, dict):
            for key, value in true_personality_raw.items():
                if key == "city_country":
                    normalized_personality["current_city_country"] = value
                else:
                    normalized_personality[key] = value
        else:
            normalized_personality = true_personality_raw 

        if not original_response or not normalized_personality:
             logging.warning(f"[Record {record_id}] Skipping benchmark - missing original_response or personality.")
             return result

        text_to_attack = original_response if attack_original else anonymized_response
        if not text_to_attack:
             if not attack_original:
                 logging.warning(f"[Record {record_id}] Skipping benchmark - missing anonymized_response.")
                 return result
             else:
                 logging.warning(f"[Record {record_id}] Skipping benchmark - missing original response.")
                 return result


        # 计算效用分数
        if not attack_original:
            if anonymized_response: 
                utility_scores = calculate_utility(original_response, anonymized_response, record_id)
                if utility_scores:
                    result["utility"] = utility_scores["combined"]
                    result["readability"] = utility_scores["readability"]
                    result["meaning"] = utility_scores["meaning"]
                    result["hallucinations"] = utility_scores["hallucinations"]
                    result["rouge_l_f1"] = utility_scores["rouge_l_f1"]
            else:
                logging.warning(f"[Record {record_id}] Cannot calculate utility - missing anonymized_response.")

        leaked_c, total_c, leaked_set, present_set = calculate_privacy_leak(
            text_to_attack, normalized_personality, record_id, attributes_to_protect
        )
        result["leaked_count"] = leaked_c
        result["total_true_count"] = total_c
        result["leaked_categories"] = sorted(list(leaked_set)) 
        result["present_categories"] = sorted(list(present_set)) 

    except Exception as e:
        logging.error(f"[Record {record_id}] Unexpected error during benchmark processing: {e}", exc_info=True)
        result["error"] = str(e) 

    return result


def main():
    global DEEPSEEK_MODEL_JUDGE, DEEPSEEK_MODEL_ADVERSARY, log_filename, API_KEY
    parser = argparse.ArgumentParser(description="Calculate Detailed Privacy-Utility Benchmark using DeepSeek")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input JSONL (e.g., health_test.jsonl or your anonymized file)")
    parser.add_argument("--output_file", type=str, default="benchmark_results_detailed.json", 
                        help="Path to save detailed JSON results")
    parser.add_argument("--log_file", type=str, default=log_filename, 
                        help="Path to save log file")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit processing to N records")
    parser.add_argument("--workers", type=int, default=50, 
                        help="Number of parallel workers")
    parser.add_argument("--judge_model", type=str, default=DEEPSEEK_MODEL_JUDGE, 
                        help="DeepSeek model for Utility Judge")
    parser.add_argument("--adversary_model", type=str, default=DEEPSEEK_MODEL_ADVERSARY, 
                        help="DeepSeek model for Final Adversary")
    parser.add_argument("--attack_original", action="store_true", 
                        help="Set this flag to attack the original text (response) instead of the anonymized text (anonymized_response)")
    parser.add_argument("--protect_attribute", type=str, default=None, 
                        help="可选：指定要评估的单个属性（例如 'health_issue'）。如果未设置，则评估所有属性。")

    args = parser.parse_args()

    # --- 加载 .env 文件并设置 API_KEY ---
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        logging.error("DeepSeek API key not found. Set API_KEY environment variable or in .env file.")
        sys.exit(1)
    
    DEEPSEEK_MODEL_JUDGE = args.judge_model
    DEEPSEEK_MODEL_ADVERSARY = args.adversary_model
    log_filename = args.log_file

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(log_filename, 'w', 'utf-8'),
        logging.StreamHandler(sys.stdout)
    ])
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    all_records = load_jsonl(args.input_file)
    records_to_process = all_records[:args.limit] if args.limit else all_records

    if not records_to_process:
        logging.info("No records to process.")
        return

    # --- 确定要保护的属性列表 ---
    attributes_to_protect_list: List[str]
    protected_attributes_set: Set[str]

    if args.protect_attribute:
        if args.protect_attribute not in CORE_PII_ATTRIBUTES_ALL:
            logging.error(f"Error: --protect_attribute '{args.protect_attribute}' is not valid. Must be one of {CORE_PII_ATTRIBUTES_ALL}")
            sys.exit(1)
        attributes_to_protect_list = [args.protect_attribute]
        protected_attributes_set = {args.protect_attribute}
        logging.info(f"*** SINGLE ATTRIBUTE EVALUATION MODE: Evaluating ONLY '{args.protect_attribute}' ***")
    else:
        attributes_to_protect_list = sorted(list(CORE_PII_ATTRIBUTES_ALL))
        protected_attributes_set = CORE_PII_ATTRIBUTES_ALL
        logging.info(f"*** FULL EVALUATION MODE: Evaluating all attributes: {attributes_to_protect_list} ***")

    mode = "Original Text" if args.attack_original else "Anonymized Text"
    logging.info(f"Starting benchmark calculation for {len(records_to_process)} records (Attacking: {mode}) using {args.workers} workers...")
    logging.info(f"Utility Judge Model: {DEEPSEEK_MODEL_JUDGE}")
    logging.info(f"Final Adversary Model: {DEEPSEEK_MODEL_ADVERSARY}")

    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(process_record_for_benchmark, record, i, args.attack_original, attributes_to_protect_list): i
                        for i, record in enumerate(records_to_process)}
        for future in tqdm(as_completed(future_to_id), total=len(records_to_process), desc=f"Benchmarking ({mode})"):
            record_id = future_to_id[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                logging.error(f'[Record {record_id}] generated an exception: {exc}', exc_info=True)
                all_results.append({"record_id": record_id, "utility": None, "leaked_count": 0, "total_true_count": 0, "leaked_categories": [], "present_categories": [], "error": str(exc)})

    # 汇总结果
    valid_utility_scores = [r["utility"] for r in all_results if r.get("utility") is not None]
    valid_readability_scores = [r["readability"] for r in all_results if r.get("readability") is not None]
    valid_meaning_scores = [r["meaning"] for r in all_results if r.get("meaning") is not None]
    valid_hallucinations_scores = [r["hallucinations"] for r in all_results if r.get("hallucinations") is not None]
    valid_rouge_l_f1_scores = [r["rouge_l_f1"] for r in all_results if r.get("rouge_l_f1") is not None]
    
    total_leaked_attributes = sum(r.get("leaked_count", 0) for r in all_results)
    total_possible_attributes = sum(r.get("total_true_count", 0) for r in all_results if r.get("total_true_count") is not None)

    leaked_counts_per_category = defaultdict(int)
    present_counts_per_category = defaultdict(int)
    for r in all_results:
        if "leaked_categories" in r and isinstance(r["leaked_categories"], list):
            for category in r["leaked_categories"]:
                if category in protected_attributes_set: 
                    leaked_counts_per_category[category] += 1
        if "present_categories" in r and isinstance(r["present_categories"], list):
             for category in r["present_categories"]:
                if category in protected_attributes_set: 
                    present_counts_per_category[category] += 1

    leakage_probability_per_category = {
        attr: (leaked_counts_per_category[attr] / present_counts_per_category[attr]) if present_counts_per_category.get(attr, 0) > 0 else 0.0
        for attr in protected_attributes_set
    }

    leaked_counts_per_category_dict = dict(leaked_counts_per_category)
    present_counts_per_category_dict = dict(present_counts_per_category)

    average_utility = sum(valid_utility_scores) / len(valid_utility_scores) if valid_utility_scores else (1.0 if args.attack_original else 0.0)
    average_readability = sum(valid_readability_scores) / len(valid_readability_scores) if valid_readability_scores else None
    average_meaning = sum(valid_meaning_scores) / len(valid_meaning_scores) if valid_meaning_scores else None
    average_hallucinations = sum(valid_hallucinations_scores) / len(valid_hallucinations_scores) if valid_hallucinations_scores else None
    average_rouge_l_f1 = sum(valid_rouge_l_f1_scores) / len(valid_rouge_l_f1_scores) if valid_rouge_l_f1_scores else None
    adversarial_accuracy = total_leaked_attributes / total_possible_attributes if total_possible_attributes > 0 else 0.0

    num_failed_utility = 0 if args.attack_original else len(records_to_process) - len(valid_utility_scores)
    num_failed_privacy = len([r for r in all_results if r.get("error") or "total_true_count" not in r])


    logging.info(f"\n--- Benchmark Results ({mode} vs {DEEPSEEK_MODEL_ADVERSARY}) ---")
    logging.info(f"Processed Records: {len(records_to_process)}")
    if not args.attack_original:
        logging.info(f"Utility Calculation Failed: {num_failed_utility}")
        logging.info(f"Average Readability: {average_readability:.2f}" if average_readability is not None else "Average Readability: N/A")
        logging.info(f"Average Meaning: {average_meaning:.2f}" if average_meaning is not None else "Average Meaning: N/A")
        logging.info(f"Average Hallucinations: {average_hallucinations:.2f}" if average_hallucinations is not None else "Average Hallucinations: N/A")
        logging.info(f"Average ROUGE-L F1: {average_rouge_l_f1:.4f}" if average_rouge_l_f1 is not None else "Average ROUGE-L F1: N/A")
    logging.info(f"Privacy Calculation Failed/Skipped: {num_failed_privacy}")
    logging.info("-" * 25)
    logging.info(f"Average Utility (Combined): {average_utility:.4f}")
    logging.info(f"Overall Adversarial Accuracy: {adversarial_accuracy:.4f} ({total_leaked_attributes} / {total_possible_attributes})")
    logging.info("-" * 25)
    logging.info("PII Leakage Breakdown by Category:")

    sorted_categories = sorted(list(protected_attributes_set))
    for category in sorted_categories:
        leaked_c = leaked_counts_per_category_dict.get(category, 0)
        present_c = present_counts_per_category_dict.get(category, 0)
        prob = leakage_probability_per_category.get(category, 0.0)
        logging.info(f"  - {category:<22}: Leaked {leaked_c:>4} / {present_c:>4} times | Probability = {prob:.4f}")
    logging.info("-" * 25)

    # 保存详细结果
    summary_data = {
        "mode": mode,
        "protected_attributes": attributes_to_protect_list, 
        "average_utility": average_utility,
        "average_readability": average_readability,
        "average_meaning": average_meaning,
        "average_hallucinations": average_hallucinations,
        "average_rouge_l_f1": average_rouge_l_f1,
        "overall_adversarial_accuracy": adversarial_accuracy,
        "adversary_model": DEEPSEEK_MODEL_ADVERSARY,
        "total_leaked": total_leaked_attributes,
        "total_possible": total_possible_attributes,
        "processed_records": len(records_to_process),
        "failed_utility": num_failed_utility,
        "failed_privacy": num_failed_privacy,
        "leakage_counts_per_category": leaked_counts_per_category_dict,
        "present_counts_per_category": present_counts_per_category_dict,
        "leakage_probability_per_category": leakage_probability_per_category
    }
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary_data, "details": all_results}, f, indent=2, ensure_ascii=False)
        logging.info(f"Detailed benchmark results saved to '{args.output_file}'")
    except Exception as e:
        logging.error(f"Failed to save detailed results to '{args.output_file}': {e}", exc_info=True)

if __name__ == "__main__":
    main()