import json
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rouge_score import rouge_scorer
from tqdm import tqdm

# 从共享工具文件中导入
from eval_utils import (
    load_jsonl, 
    extract_first_json, 
    call_deepseek_api, 
    setup_logging, 
    API_KEY
)

# --- 常量和 Prompt ---
DEEPSEEK_MODEL_JUDGE_DEFAULT = "deepseek-chat"

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

# --- 效用评估核心函数 ---

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

def calculate_utility(original_text: str, anonymized_text: str, record_id: int, judge_model: str) -> Optional[Dict[str, float]]:
    """计算效用分数，返回包含各项分数的字典"""
    judge_messages = [{"role": "system", "content": UTILITY_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": UTILITY_JUDGE_USER_PROMPT_TEMPLATE.format(original_comment_string=original_text, adapted_comment_string=anonymized_text)}]
    judge_response = call_deepseek_api(messages=judge_messages, model_name=judge_model, max_tokens=512, temperature=0.1, record_id=record_id, purpose="Utility Judge")
    
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
        "utility": combined_utility,
        "readability": readability,
        "meaning": meaning,
        "hallucinations": hallucinations,
        "rouge_l_f1": rouge_l_f1
    }

def process_record_for_utility(record: Dict[str, Any], record_id: int, judge_model: str) -> Dict[str, Any]:
    """处理单个记录以计算效用，返回结果字典"""
    result = {
        "record_id": record_id,
        "utility": None,
        "readability": None,
        "meaning": None,
        "hallucinations": None,
        "rouge_l_f1": None,
        "error": None
    }
    try:
        original_response = record.get("response")
        anonymized_response = record.get("anonymized_response")

        if not original_response:
            logging.warning(f"[Record {record_id}] Skipping utility - missing original_response.")
            result["error"] = "Missing original_response"
            return result

        if not anonymized_response:
            logging.warning(f"[Record {record_id}] Skipping utility - missing anonymized_response.")
            result["error"] = "Missing anonymized_response"
            return result

        utility_scores = calculate_utility(original_response, anonymized_response, record_id, judge_model)
        
        if utility_scores:
            result.update(utility_scores)
        else:
            result["error"] = "Utility calculation failed"

    except Exception as e:
        logging.error(f"[Record {record_id}] Unexpected error during utility processing: {e}", exc_info=True)
        result["error"] = str(e) 

    return result

# --- 主执行函数 ---

def main():
    parser = argparse.ArgumentParser(description="Calculate Utility Benchmark using DeepSeek")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL (must contain 'response' and 'anonymized_response')")
    parser.add_argument("--output_file", type=str, default="benchmark_results_utility.json", help="Path to save detailed JSON results for utility")
    parser.add_argument("--log_file", type=str, default="eval_utility.log", help="Path to save log file")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to N records")
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel workers")
    parser.add_argument("--judge_model", type=str, default=DEEPSEEK_MODEL_JUDGE_DEFAULT, help="DeepSeek model for Utility Judge")
    
    args = parser.parse_args()

    setup_logging(args.log_file)

    if not API_KEY:
        logging.error("DeepSeek API key not found. Set API_KEY environment variable.")
        sys.exit(1)

    all_records = load_jsonl(args.input_file)
    records_to_process = all_records[:args.limit] if args.limit else all_records

    if not records_to_process:
        logging.info("No records to process.")
        return

    logging.info(f"Starting utility calculation for {len(records_to_process)} records using {args.workers} workers...")
    logging.info(f"Utility Judge Model: {args.judge_model}")

    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(process_record_for_utility, record, i, args.judge_model): i
                        for i, record in enumerate(records_to_process)}
        
        for future in tqdm(as_completed(future_to_id), total=len(records_to_process), desc="Calculating Utility"):
            record_id = future_to_id[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                logging.error(f'[Record {record_id}] generated an exception: {exc}', exc_info=True)
                all_results.append({"record_id": record_id, "utility": None, "error": str(exc)})

    # --- 汇总结果 ---
    valid_utility_scores = [r["utility"] for r in all_results if r.get("utility") is not None]
    valid_readability_scores = [r["readability"] for r in all_results if r.get("readability") is not None]
    valid_meaning_scores = [r["meaning"] for r in all_results if r.get("meaning") is not None]
    valid_hallucinations_scores = [r["hallucinations"] for r in all_results if r.get("hallucinations") is not None]
    valid_rouge_l_f1_scores = [r["rouge_l_f1"] for r in all_results if r.get("rouge_l_f1") is not None]
    num_failed_utility = len(records_to_process) - len(valid_utility_scores)

    average_utility = sum(valid_utility_scores) / len(valid_utility_scores) if valid_utility_scores else 0.0
    average_readability = sum(valid_readability_scores) / len(valid_readability_scores) if valid_readability_scores else None
    average_meaning = sum(valid_meaning_scores) / len(valid_meaning_scores) if valid_meaning_scores else None
    average_hallucinations = sum(valid_hallucinations_scores) / len(valid_hallucinations_scores) if valid_hallucinations_scores else None
    average_rouge_l_f1 = sum(valid_rouge_l_f1_scores) / len(valid_rouge_l_f1_scores) if valid_rouge_l_f1_scores else None

    logging.info(f"\n--- Utility Benchmark Results (vs {args.judge_model}) ---")
    logging.info(f"Processed Records: {len(records_to_process)}")
    logging.info(f"Utility Calculation Failed: {num_failed_utility}")
    logging.info(f"Average Readability: {average_readability:.2f}" if average_readability is not None else "Average Readability: N/A")
    logging.info(f"Average Meaning: {average_meaning:.2f}" if average_meaning is not None else "Average Meaning: N/A")
    logging.info(f"Average Hallucinations: {average_hallucinations:.2f}" if average_hallucinations is not None else "Average Hallucinations: N/A")
    logging.info(f"Average ROUGE-L F1: {average_rouge_l_f1:.4f}" if average_rouge_l_f1 is not None else "Average ROUGE-L F1: N/A")
    logging.info("-" * 25)
    logging.info(f"Average Utility (Combined): {average_utility:.4f}")
    logging.info("-" * 25)

    # --- 保存详细结果 ---
    summary_data = {
        "judge_model": args.judge_model,
        "average_utility": average_utility,
        "average_readability": average_readability,
        "average_meaning": average_meaning,
        "average_hallucinations": average_hallucinations,
        "average_rouge_l_f1": average_rouge_l_f1,
        "processed_records": len(records_to_process),
        "failed_utility": num_failed_utility,
    }
    
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary_data, "details": all_results}, f, indent=2, ensure_ascii=False)
        logging.info(f"Detailed utility results saved to '{args.output_file}'")
    except Exception as e:
        logging.error(f"Failed to save detailed results to '{args.output_file}': {e}", exc_info=True)

if __name__ == "__main__":
    main()