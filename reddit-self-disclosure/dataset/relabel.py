import json
import argparse
import logging
import asyncio
from typing import Optional, List, Dict, Any, Literal, Tuple, Type
# 修复：从 pydantic 导入 ValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator, FieldValidationInfo
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from tqdm import tqdm as sync_tqdm
import sys # 用于在Python 3.8+上设置asyncio策略

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 修复：在 Windows 上为 Python 3.8+ 设置 asyncio 事件循环策略 ---
# 这个问题在 Python 3.8+ 的 Windows 上很常见，当关闭 AsyncOpenAI 客户端时
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- LLM 提取的系统 Prompt (更新为固定的5个标签) ---
SYSTEM_PROMPT_FOR_EXTRACTION = """
You are an expert biographical analyst and a precise JSON labeler.

**Core Task:**
Analyze the user-provided text. Use disciplined inference and world knowledge to extract information *only* for the schema below.
- "age" must be an integer (e.g., 20).
- "city_country" should be "City, Country" or just "Country" (e.g., "Shanghai, China" or "China").
- "sex" and "relationship_status" must be one of the provided enum values.
- Use `null` (JSON null, not string) *only* when no signal exists.

**Output Rules:**
Respond ONLY with a single, valid JSON object. Do not add extra keys. All enum values must be lowercase. Do not output reasoning.

**Output Schema (exact keys, no others):**
{
  "age": <int or null>,
  "sex": <"male" | "female" | null>,
  "city_country": <string or null>,
  "occupation": <string or null>,
  "relationship_status": <"single" | "in a relationship" | "engaged" | "married" | "divorced" | "widowed" | null>
}

**Example 1:**
Input:
[Biography Text]:
"I (28F) am a social worker living in London, UK. My fiancé and I are planning our wedding."

Expected output:
{
  "age": 28,
  "sex": "female",
  "city_country": "London, UK",
  "occupation": "social worker",
  "relationship_status": "engaged"
}

**Example 2:**
Input:
[Biography Text]:
"My brother, a 45 year old, just lost his job. He's been married for 10 years."

Expected output:
{
  "age": 45,
  "sex": "male",
  "city_country": null,
  "occupation": null,
  "relationship_status": "married"
}

**Example 3:**
Input:
[Biography Text]:
"As a student, I find it hard to date."

Expected output:
{
  "age": null,
  "sex": null,
  "city_country": null,
  "occupation": "student",
  "relationship_status": "single"
}
"""

# --- Pydantic 模型定义 (已修复) ---
class Personality(BaseModel):
    age: Optional[int] = Field(None, description="Inferred age as an integer")
    
    sex: Optional[Literal["male", "female"]] = Field(None, description="Inferred sex")
    
    city_country: Optional[str] = Field(None, description="Inferred city and country, or just country")
    
    occupation: Optional[str] = Field(None, description="Primary occupation mentioned")
    
    relationship_status: Optional[Literal[
        "single", "in a relationship", "engaged", "married", "divorced", "widowed"
    ]] = Field(None, description="Inferred relationship status")

    @field_validator('age')
    @classmethod
    def check_age_range(cls, v):
        """确保年龄在合理范围内"""
        if v is not None and (v < 10 or v > 120):
            logging.warning(f"Extracted age {v} out of plausible range (10-120); coercing to None.")
            return None
        return v

    # --- 修复开始 ---
    @field_validator('sex', 'relationship_status')
    @classmethod
    def normalize_enums(cls, v, info: FieldValidationInfo): # <--- 修复 1: 添加 info: FieldValidationInfo
        """使枚举值小写并去除空格"""
        if v is None:
            return None
        v_lower = v.strip().lower()
        
        # 动态获取字段的允许值
        # 修复 2: 使用 info.field_name 代替 model_fields.current_field.name
        field_type = cls.model_fields[info.field_name].annotation 
        
        # 处理 Optional[Literal[...]]
        if (hasattr(field_type, '__args__') and 
            field_type.__args__ and
            hasattr(field_type.__args__[0], '__args__')):
            
            allowed_values = field_type.__args__[0].__args__
            if v_lower in allowed_values:
                return v_lower
                
        return None # 如果值不在枚举中，则返回 None
    # --- 修复结束 ---
    @field_validator('city_country', 'occupation')
    @classmethod
    def normalize_strings(cls, v):
        """清理字符串字段"""
        if v is None:
            return None
        v_stripped = v.strip()
        # 如果字符串为空，则返回 None
        return v_stripped if v_stripped else None

# --- Async API 调用和处理函数 ---
async def process_entry(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    entry_index: int,
    original_data: Dict[str, Any],
    system_prompt: str,                 # 传入动态提示
    ValidatorModel: Type[BaseModel],    # 传入动态模型
    model_name: str = "deepseek-chat",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[int, Dict[str, Any], Optional[BaseModel]]: # 返回类型更新为 BaseModel

    # *** 关键更改：从 "text" 切换到 "response" ***
    text_to_process = original_data.get("response")
    if not text_to_process or not text_to_process.strip():
        logging.warning(f"Entry {entry_index + 1}: Received empty 'response' text, skipping processing.")
        return entry_index, original_data, None

    user_prompt = f"[Biography Text]:\n{text_to_process}"
    messages = [
        {"role": "system", "content": system_prompt}, # 使用传入的动态提示
        {"role": "user", "content": user_prompt}
    ]

    async with semaphore:
        logging.debug(f"Entry {entry_index + 1}: Acquired semaphore, starting API call.")
        for attempt in range(max_retries):
            try:
                logging.debug(f"Entry {entry_index + 1}: Attempt {attempt + 1}/{max_retries} calling LLM API.")
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_response = completion.choices[0].message.content
                logging.debug(f"Entry {entry_index + 1}: Raw API response: {raw_response}")

                if not raw_response:
                    logging.warning(f"Entry {entry_index + 1}: API returned an empty response on attempt {attempt + 1}.")
                    extracted_data = {}
                else:
                    # 清理可能的 markdown 代码块
                    if raw_response.strip().startswith("```json"):
                        raw_response = raw_response.strip()[7:-3].strip()
                    elif raw_response.strip().startswith("```"):
                         raw_response = raw_response.strip()[3:-3].strip()
                    extracted_data = json.loads(raw_response)

                # *** 关键更改：使用动态验证器 ***
                validated_personality = ValidatorModel.model_validate(extracted_data)
                logging.debug(f"Entry {entry_index + 1}: Validation successful.")
                return entry_index, original_data, validated_personality

            except json.JSONDecodeError as e:
                logging.error(f"Entry {entry_index + 1}: Failed JSON decode attempt {attempt + 1}. Error: {e}. Response: '{raw_response}'")
                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for JSON decoding.")
                    return entry_index, original_data, None

            except ValidationError as e: # 修复：现在已导入 ValidationError
                logging.error(f"Entry {entry_index + 1}: Pydantic validation failed attempt {attempt + 1}. Error: {e}. Data: '{extracted_data}'")
                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for Pydantic validation.")
                    return entry_index, original_data, None

            except (APIError, RateLimitError, APITimeoutError, APIConnectionError) as e:
                logging.warning(f"Entry {entry_index + 1}: API call failed attempt {attempt + 1}/{max_retries}: {e}")
                sleep_time = retry_delay * (attempt + 1)
                if isinstance(e, RateLimitError):
                    logging.warning(f"Rate limit hit, increasing retry delay to {sleep_time}s.")

                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for API call.")
                    return entry_index, original_data, None

            except Exception as e:
                logging.exception(f"Entry {entry_index + 1}: Unexpected error attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached after unexpected error.")
                    return entry_index, original_data, None

        logging.debug(f"Entry {entry_index + 1}: Releasing semaphore.")
    return entry_index, original_data, None

# --- Async 主函数 ---
async def main():
    parser = argparse.ArgumentParser(description="Extract 5 specific personality attributes from a JSONL file using an LLM API.")
    
    # --- *** 更新参数 (移除了 --label_to_extract) *** ---
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file (e.g., extracted_Health.jsonl).")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file (e.g., relabeled_Health.jsonl).")
    # --- *** ---
    
    parser.add_argument("--api_key", required=True, help="API key for the LLM service.")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1", help="Base URL for the API endpoint (default: DeepSeek).")
    parser.add_argument("--model", default="deepseek-chat", help="Name of the LLM model to use (default: deepseek-chat).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N entries (optional).")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of API call retries on failure.")
    parser.add_argument("--retry_delay", type=int, default=5, help="Base delay (in seconds) between API call retries.")
    parser.add_argument("--concurrency", type=int, default=50, help="Maximum number of concurrent API requests.")

    args = parser.parse_args()

    # --- *** 移除了动态创建逻辑 *** ---
    logging.info(f"Task: Extracting fixed 5 personality labels.")
    logging.info(f"Input: {args.input_file}, Output: {args.output_file}")

    # --- 初始化 Async API 客户端 ---
    try:
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
        logging.info(f"Initialized AsyncOpenAI client for endpoint: {args.base_url} with concurrency {args.concurrency}")
    except Exception as e:
        logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
        return

    # --- 读取输入文件 ---
    lines_to_process = []
    total_lines_read = 0
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if args.limit is not None and i >= args.limit:
                    logging.info(f"Reached input limit of {args.limit} lines.")
                    break
                try:
                    original_data = json.loads(line.strip())
                    lines_to_process.append((i, original_data)) # 存储索引和数据
                except json.JSONDecodeError:
                    logging.error(f"Skipping line {i + 1} due to invalid JSON.")
                
                total_lines_read += 1
            
            if args.limit is None:
                total_lines_read = len(lines_to_process)

        logging.info(f"Read {len(lines_to_process)} valid entries to process from {args.input_file}")
        if not lines_to_process:
            logging.warning("Input file was empty or contained no valid JSON.")
            return

    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    except IOError as e:
        logging.error(f"Error reading input file: {e}")
        return

    # --- 创建并发任务 ---
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for index, data in lines_to_process:
        tasks.append(
            asyncio.create_task(
                process_entry(
                    client,
                    semaphore,
                    index,
                    data,
                    SYSTEM_PROMPT_FOR_EXTRACTION,    # *** 更改：传入静态 Prompt ***
                    Personality,                     # *** 更改：传入静态模型类 ***
                    model_name=args.model,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay
                )
            )
        )

    logging.info(f"Created {len(tasks)} tasks for concurrent processing.")

    # --- 执行任务并收集结果 ---
    results_list: List[Tuple[int, Dict[str, Any], Optional[BaseModel]]] = []
    try:
        logging.info("Processing entries (use Ctrl+C to interrupt)...")
        for f in sync_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            results_list.append(await f)
            
        logging.info("All tasks completed.")
        
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Shutting down tasks... (this may take a moment)")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("Tasks cancelled.")
        
    except Exception as e:
        logging.exception(f"An error occurred during asyncio.as_completed: {e}")
        logging.error("Aborting due to error during task execution.")
    
    finally:
         await client.close() # 确保客户端关闭

    # --- 处理结果并写入输出文件 ---
    processed_count = 0
    error_count = 0
    
    results_list.sort(key=lambda x: x[0]) # 按原始索引排序

    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for index, original_data, personality_result in sync_tqdm(results_list, desc="Writing results"):
                
                if personality_result:
                    original_data["personality"] = personality_result.model_dump(mode='json')
                    processed_count += 1
                else:
                    original_data["personality"] = None
                    if original_data.get("response"): 
                        error_count += 1
                    else:
                        pass

                outfile.write(json.dumps(original_data, ensure_ascii=False) + '\n')
                
    except IOError as e:
        logging.error(f"Error writing output file: {e}")
    finally:
        logging.info("--- Processing Summary ---")
        logging.info(f"Total entries in file (approx): {total_lines_read}")
        logging.info(f"Entries read (up to limit): {len(lines_to_process)}")
        logging.info(f"Entries processed (results received): {len(results_list)}")
        logging.info(f"Successfully extracted: {processed_count}")
        logging.info(f"Errors or empty results: {error_count}")
        logging.info(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())