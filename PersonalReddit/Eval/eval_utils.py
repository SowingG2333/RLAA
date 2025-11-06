import os
import json
import logging
import time
import re
import sys
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 全局常量
MAX_RETRIES = 3
RETRY_DELAY = 5

def setup_logging(log_filename: str, log_level: str = "INFO"):
    """配置日志记录器"""
    log_level_enum = getattr(logging, log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    root_logger = logging.getLogger()
    # 移除所有现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    logging.basicConfig(level=log_level_enum, format=log_format, handlers=[
        logging.FileHandler(log_filename, 'w', 'utf-8'),
        logging.StreamHandler(sys.stdout)
    ])
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

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
    if not API_KEY:
        logging.error("API_KEY not set.")
        return None
    
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
    payload = json.dumps({"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    # url = "https://api.deepseek.com/chat/completions"
    url = "https://api.siliconflow.cn/v1/chat/completions"

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
            logging.warning(f"[Record {record_id}] DeepSeek API call for {purpose} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt + 1))
            else: logging.error(f"[Record {record_id}] DeepSeek API call for {purpose} failed after {MAX_RETRIES} attempts."); return None
        except Exception as e:
            logging.error(f"[Record {record_id}] Unexpected error calling DeepSeek API for {purpose}: {e}", exc_info=True); return None

def normalize_personality(true_personality: Any) -> Dict[str, Any]:
    """标准化 personality 字典，处理 'city_country' -> 'current_city_country'"""
    normalized_personality = {}
    if isinstance(true_personality, dict):
        for key, value in true_personality.items():
            if key == "city_country":
                normalized_personality["current_city_country"] = value
            else:
                normalized_personality[key] = value
        return normalized_personality
    return {} # 如果输入不是字典，返回空字典