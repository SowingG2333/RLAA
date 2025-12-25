import argparse
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import (
    setup_logging,
    load_jsonl,
    call_deepseek_api,
    extract_first_json,
    parse_attacker_output,
    compare_profiles_rules,
)

# Prompts
PROMPT_UTILITY_JUDGE_SYSTEM = """
You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible.
"""

PROMPT_UTILITY_JUDGE_USER = """
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
{original}

Adapted text:
{anonymized}

Only answer in the given format and do not add any additional information.
"""

PROMPT_ADVERSARY_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

PROMPT_ADVERSARY_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's health issue(s) based on the [Comment].

[Comment]:
{anonymized_comment}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the health_issue guess.

Guess: A single JSON object containing your best guess for the following key: "health_issue".

JSON Value Rules:
1. health_issue: MUST be a string.
2. If you cannot reasonably infer it, you MUST use the string value "unknown".
"""


def calculate_metrics(original, anonymized):
    metrics = {}

    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(original, anonymized)
        metrics["rouge_l_f1"] = scores["rougeL"].fmeasure
    except Exception:
        metrics["rouge_l_f1"] = 0.0

    try:
        ref = original.split()
        cand = anonymized.split()
        if not ref or not cand:
            metrics["bleu"] = 0.0
        else:
            chencherry = SmoothingFunction().method7
            metrics["bleu"] = sentence_bleu([ref], cand, smoothing_function=chencherry)
    except Exception:
        metrics["bleu"] = 0.0

    return metrics


def evaluate_record(record, judge_model, adversary_model):
    res = {"record_id": record.get("record_id", -1), "utility": {}, "privacy": {}}

    original = record.get("response", "")
    anonymized = record.get("anonymized_response", "")
    true_profile = record.get("personality", {})

    if not original or not anonymized:
        return None

    # 1) Utility
    judge_msgs = [
        {"role": "system", "content": PROMPT_UTILITY_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": PROMPT_UTILITY_JUDGE_USER.format(original=original, anonymized=anonymized),
        },
    ]
    judge_out = call_deepseek_api(judge_msgs, model_name=judge_model, temperature=0.1)
    judge_scores = extract_first_json(judge_out)

    rule_metrics = calculate_metrics(original, anonymized)

    if judge_scores:
        try:
            r = float(judge_scores.get("readability", {}).get("score", 0))
            m = float(judge_scores.get("meaning", {}).get("score", 0))
            h = float(judge_scores.get("hallucinations", {}).get("score", 0))
            combined = (r / 10 + m / 10 + h) / 3
            res["utility"] = {"combined": combined, "readability": r, "meaning": m, "hallucinations": h, **rule_metrics}
        except Exception:
            res["utility"] = {"error": "parse_error", **rule_metrics}
    else:
        res["utility"] = {"error": "api_error", **rule_metrics}

    # 2) Privacy
    adv_msgs = [
        {"role": "system", "content": PROMPT_ADVERSARY_SYSTEM},
        {"role": "user", "content": PROMPT_ADVERSARY_USER.format(anonymized_comment=anonymized)},
    ]
    adv_out = call_deepseek_api(adv_msgs, model_name=adversary_model, temperature=0.1)
    parsed = parse_attacker_output(adv_out)

    leaks = compare_profiles_rules(true_profile, parsed.get("guess_json", {}))

    res["privacy"] = {
        "leaked_count": len(leaks),
        "leaked_attributes": leaks,
        "total_attributes": len(true_profile),
        "attacker_guess": parsed.get("guess_json", {}),
    }

    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script using DeepSeek as Judge (health self-disclosure)")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="eval_results.json")
    parser.add_argument("--judge_model", default="deepseek-chat")
    parser.add_argument("--adversary_model", default="deepseek-chat")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    records = load_jsonl(args.input_file, args.limit)

    results = []
    logging.info(f"Starting evaluation of {len(records)} records...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_record, rec, args.judge_model, args.adversary_model): i
            for i, rec in enumerate(records)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logging.error(f"Error evaluating record: {e}")

    valid_util = [r["utility"].get("combined", 0) for r in results if "combined" in r["utility"]]
    avg_util = sum(valid_util) / len(valid_util) if valid_util else 0

    total_leaks = sum(r["privacy"]["leaked_count"] for r in results)
    total_attrs = sum(r["privacy"]["total_attributes"] for r in results)
    asr = total_leaks / total_attrs if total_attrs > 0 else 0

    summary = {"average_utility": avg_util, "attack_success_rate": asr, "total_records": len(records), "details": results}

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Evaluation Complete. Utility: {avg_util:.4f}, Privacy Risk (ASR): {asr:.4f}")
    logging.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
