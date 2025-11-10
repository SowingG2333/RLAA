import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

import time

# --- [全局配置] ---
# 客户端将在 main 函数中初始化
GLOBAL_CLIENT: Optional[OpenAI] = None 
API_LLM_MODEL = "deepseek-chat"
MAX_RETRIES = 3
RETRY_DELAY = 5
# --- [结束] ---

# --- [新] 针对 Health Issue 的 Persona 生成 Prompt ---
PROMPT_PERSONA_GENERATOR_HEALTH_SYSTEM = """
You are a creative privacy strategist. Your goal is to generate an alternative, plausible "health_issue" persona based on a real user's health profile.
"""
PROMPT_PERSONA_GENERATOR_HEALTH_USER = """
Analyze the [Real Profile]'s "health_issue" list. Create a significantly different but plausible [Target Persona] JSON object.

[Real Profile]:
{true_profile_str}

**Instructions & Constraints:**
1.  The [Target Persona] MUST contain ONLY the "health_issue" key.
2.  The "health_issue" value MUST be a list of new health-related terms (1-3 terms).
3.  The new terms MUST be different from the [Real Profile] terms (no overlap).
4.  The new terms MUST be plausible and internally consistent (e.g., a condition and a related medication, or a plausible set of symptoms).
5.  **Example 1:** If [Real Profile] is `["psoriasis", "fluocinolone"]`, a good [Target Persona] is `{{ "health_issue": ["rheumatoid arthritis", "enbrel"] }}`.
6.  **Example 2:** If [Real Profile] is `["adhd", "chronic pain"]`, a good [Target Persona] is `{{ "health_issue": ["insomnia", "ambien"] }}`.
7.  Output ONLY a single, valid JSON object for the [Target Persona]. Do not add explanation or preamble.
"""
# --- [结束] ---

# --- [辅助函数] ---
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
    return data

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

def call_deepseek_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]], 
    temperature: float,
    max_tokens: int,
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
                raise e # 抛出异常
    
    raise Exception(f"{log_prefix} API call failed after all retries and returned no content.")
# --- [辅助函数结束] ---

def generate_target_persona_health_api(
    client: OpenAI, model: str, true_personality: Dict[str, Any],
    record_id: int = -1, max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """使用 API 生成一个虚假的、受约束的 health_issue 目标人设"""
    record_log_prefix = f"[Record-{record_id}]"
    
    # [修改]：只提取 'health_issue'
    real_health_issues = true_personality.get("health_issue")
    if not real_health_issues or not isinstance(real_health_issues, list):
        logging.warning(f"{record_log_prefix} No 'health_issue' list in true profile to generate target from.")
        return None
        
    true_profile_for_prompt = {"health_issue": real_health_issues}
    true_profile_str = json.dumps(true_profile_for_prompt, indent=2, ensure_ascii=False)
    
    user_content = PROMPT_PERSONA_GENERATOR_HEALTH_USER.format(
        true_profile_str=true_profile_str
    )
    
    for attempt in range(max_retries + 1):
        try:
            response_text = call_deepseek_api(
                client=client, model=model,
                messages=[{"role": "system", "content": PROMPT_PERSONA_GENERATOR_HEALTH_SYSTEM.strip()},
                          {"role": "user", "content": user_content}],
                temperature=0.7, max_tokens=512, record_id=record_id,
                purpose="PersonaGenerator", use_json_mode=True # 强制 JSON 输出
            )
            
            target_persona = extract_first_json_object(response_text)
            
            if not target_persona or not isinstance(target_persona, dict) or "health_issue" not in target_persona:
                logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] Failed to parse valid JSON persona with 'health_issue' key.")
                continue

            target_issues = target_persona.get("health_issue")
            if not isinstance(target_issues, list) or not target_issues:
                logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] 'health_issue' in target is not a non-empty list.")
                continue

            # 验证新旧列表是否不同
            real_set = set(str(t).lower() for t in real_health_issues)
            target_set = set(str(t).lower() for t in target_issues)
            
            if not real_set.isdisjoint(target_set):
                logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] Target persona overlaps with real profile: {real_set.intersection(target_set)}. Retrying...")
                continue

            logging.info(f"{record_log_prefix} Successfully generated valid target persona after {attempt+1} attempts.")
            return target_persona # 返回 {"health_issue": ["..."]}

        except Exception as e:
            logging.error(f"{record_log_prefix} [Attempt {attempt+1}] API Error: {e}", exc_info=True)
            if "content management policy" in str(e):
                logging.error(f"{record_log_prefix} Content policy violation. Skipping record.")
                return None # 无法重试
    
    logging.error(f"{record_log_prefix} Failed to generate target persona after {max_retries + 1} attempts.")
    return None

def process_persona_generation(record_tuple: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """并行工作函数：为单个记录生成人设"""
    record_id, data = record_tuple
    try:
        # [修改]：确保我们有 'record_id' 用于排序
        data["record_id"] = record_id 
        true_personality = data.get("personality")
        
        if not isinstance(true_personality, dict):
            data["persona_generation_status"] = "skipped_invalid_personality"
            return data
            
        target_persona = generate_target_persona_health_api(
            GLOBAL_CLIENT, API_LLM_MODEL, true_personality, record_id
        )
        
        if target_persona:
            data["target_persona"] = target_persona
            data["persona_generation_status"] = "success"
        else:
            data["persona_generation_status"] = "failed"
        return data
        
    except Exception as exc:
        logging.error(f"[Record {record_id}] Unhandled error in persona generation: {exc}", exc_info=True)
        data["persona_generation_status"] = "error"
        data["record_id"] = record_id
        return data

def main():
    """主函数：运行阶段 1 - 生成伪造人设"""
    global GLOBAL_CLIENT, API_LLM_MODEL
    
    parser = argparse.ArgumentParser(description="Stage 1: Generate Target Personas for 'health_issue' using API.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path (e.g., health_test.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL path (e.g., health_personas.jsonl)")
    parser.add_argument("--model", type=str, default=API_LLM_MODEL, help=f"API model name (default: {API_LLM_MODEL})")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: use API_KEY env)")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1", help="Override API base URL.")
    parser.add_argument("--max_workers", type=int, default=50, help="Max parallel API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Process first N records")
    parser.add_argument("--log_file", type=str, default="generate_health_personas.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file, 'w', 'utf-8')]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING); logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}.")

    # --- Client/Model Re-initialization ---
    API_LLM_MODEL = args.model
    try:
        GLOBAL_CLIENT = OpenAI(
            api_key=args.api_key or os.environ.get("API_KEY"), 
            base_url=args.base_url
        )
        GLOBAL_CLIENT.models.list(); logging.info("API Client initialized.")
    except Exception as e: 
        logging.error(f"Failed to initialize API client: {e}", exc_info=True)
        sys.exit(1)

    # --- Data Loading ---
    records_with_ids = [(i, r) for i, r in enumerate(load_jsonl(args.input_file))]
    if args.limit: records_with_ids = records_with_ids[:args.limit]

    # --- Persona Generation via API (Parallel) ---
    logging.info(f"Starting Target Persona generation via API for {len(records_with_ids)} records...")
    results = []; skipped_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_record = {executor.submit(process_persona_generation, record): record for record in records_with_ids}
        for future in tqdm(as_completed(future_to_record), total=len(records_with_ids), desc="Generating Personas"):
            try:
                processed_record = future.result()
                results.append(processed_record)
                if processed_record.get("persona_generation_status") != "success": 
                    skipped_count += 1
            except Exception as exc:
                logging.error(f"A task raised an exception: {exc}", exc_info=True)
                skipped_count += 1 

    # --- Write results ---
    logging.info(f"Persona generation complete. Writing {len(results)} records ({skipped_count} skipped/failed) to '{args.output_file}'.")
    
    # 按原始索引排序
    results.sort(key=lambda x: x.get("record_id", float('inf'))) 

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for result in results: 
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    logging.info("Stage 1 (API Persona Generation) finished.")

if __name__ == "__main__":
    # 注意：此脚本只运行 "generate_persona" 阶段
    main()