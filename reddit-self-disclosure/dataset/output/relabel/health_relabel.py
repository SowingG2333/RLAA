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
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- LLM 提取的系统 Prompt (更新为 Health_Issue 核心术语) ---
SYSTEM_PROMPT_FOR_HEALTH_EXTRACTION = """
You are an expert medical and biographical analyst. Your task is to extract *only* the core terms for health-related issues from the provided text.

**Core Task:**
Analyze the user-provided text. Identify personal health issues and extract *only the core medical term or concept*.
- Focus on: diagnoses (e.g., "psoriasis", "sleep apnea"), medications (e.g., "Plaquenil", "Lexapro"), or significant symptoms (e.g., "chronic pain", "insomnia", "migraines").
- Do NOT extract the full sentence (e.g., DO NOT extract "I have psoriasis,").
- Extract ONLY the key term (e.g., "psoriasis").
- Do NOT extract general feelings of sadness unless they are a clinical diagnosis (e.g., extract "depression" or "anxiety" if used as a condition, not just a feeling).
- Output a list of these key terms.
- If no health issues are mentioned, output `null` (JSON null).

**Output Rules:**
Respond ONLY with a single, valid JSON object. Do not add extra keys. All terms should be lowercase.

**Output Schema (exact keys, no others):**
{
  "health_issue": <List[str] or null>
}

**Example 1:**
Input:
[Text]:
"I struggle with this.... I have adhd among other things... also chronic pain from a spinal injury..."

Expected output:
{
  "health_issue": ["adhd", "chronic pain", "spinal injury"]
}

**Example 2:**
Input:
[Text]:
"I've been on Plaquenil for over a decade now. I have very dry skin."

Expected output:
{
  "health_issue": ["plaquenil", "dry skin"]
}

**Example 3:**
Input:
[Text]:
"My cool boss said that she thought it was a professional courtesy to not."

Expected output:
{
  "health_issue": null
}

**Example 4:**
Input:
[Text]:
"I've spent almost all of 2022 in treatment. I've made great strides and am even taking Antabuse."

Expected output:
{
  "health_issue": ["antabuse"]
}
"""

# --- Pydantic 模型定义 (更新为 HealthData 并添加小写和去重) ---
class HealthData(BaseModel):
    health_issue: Optional[List[str]] = Field(
        None, 
        description="A list of core terms (lowercase) from the text that describe a health issue, symptom, medication, or diagnosis."
    )

    @field_validator('health_issue')
    @classmethod
    def clean_health_issues(cls, v: Optional[List[str]]):
        """清理 health_issue 列表，转为小写并去重"""
        if v is None:
            return None
        # 清理列表：去除每个项目前后的空格，转为小写，并移除空字符串
        cleaned_list = [item.strip().lower() for item in v if item and item.strip()]
        # 移除重复项并排序，以保持一致性
        unique_list = sorted(list(set(cleaned_list)))
        # 如果清理后列表为空，则返回 None
        return unique_list if unique_list else None

# --- Async API 调用和处理函数 ---
async def process_entry(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    entry_index: int,
    original_data: Dict[str, Any],
    system_prompt: str,              # 传入动态提示
    ValidatorModel: Type[BaseModel],   # 传入动态模型
    model_name: str = "deepseek-chat",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[int, Dict[str, Any], Optional[BaseModel]]: 

    text_to_process = original_data.get("response")
    if not text_to_process or not text_to_process.strip():
        logging.warning(f"Entry {entry_index + 1}: Received empty 'response' text, skipping processing.")
        return entry_index, original_data, None

    user_prompt = f"[Text]:\n{text_to_process}"
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

                validated_data = ValidatorModel.model_validate(extracted_data)
                logging.debug(f"Entry {entry_index + 1}: Validation successful.")
                return entry_index, original_data, validated_data

            except json.JSONDecodeError as e:
                logging.error(f"Entry {entry_index + 1}: Failed JSON decode attempt {attempt + 1}. Error: {e}. Response: '{raw_response}'")
                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for JSON decoding.")
                    return entry_index, original_data, None

            except ValidationError as e: 
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
    parser = argparse.ArgumentParser(description="Extract core health issue terms from a JSONL file using an LLM API.")
    
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file (e.g., health.jsonl).")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file (e.g., relabeled_health_terms.jsonl).")
    
    parser.add_argument("--api_key", required=True, help="API key for the LLM service.")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1", help="Base URL for the API endpoint (default: DeepSeek).")
    parser.add_argument("--model", default="deepseek-chat", help="Name of the LLM model to use (default: deepseek-chat).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N entries (optional).")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of API call retries on failure.")
    parser.add_argument("--retry_delay", type=int, default=5, help="Base delay (in seconds) between API call retries.")
    parser.add_argument("--concurrency", type=int, default=100, help="Maximum number of concurrent API requests.")

    args = parser.parse_args()

    logging.info(f"Task: Extracting 'health_issue' core terms.")
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
                    SYSTEM_PROMPT_FOR_HEALTH_EXTRACTION,  # 传入 Health Prompt
                    HealthData,                           # 传入 HealthData 模型类
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
            for index, original_data, health_result in sync_tqdm(results_list, desc="Writing results"):
                
                if health_result:
                    original_data["personality"] = health_result.model_dump(mode='json')
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