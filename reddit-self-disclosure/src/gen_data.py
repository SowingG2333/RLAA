"""
Data generation script for creating SFT training data (Health Self-Disclosure).

Generates anonymized versions of text using DeepSeek API for supervised fine-tuning.

Usage:
    python reddit-self-disclosure/src/gen_data.py \
        --input_file reddit-self-disclosure/data/train.jsonl \
        --output_file reddit-self-disclosure/data/sft_train.jsonl
"""

import argparse
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import (
    setup_logging,
    load_jsonl,
    call_deepseek_api,
    make_context_tag,
)

# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_ANONYMIZER_SYSTEM = """
You are a privacy-preserving text rewriter. Your task is to rewrite comments to remove or obscure health-related personal information while maintaining the original meaning and readability.
"""

PROMPT_ANONYMIZER_USER = """
Please rewrite the following Reddit comment to protect the author's health privacy. Remove or generalize any information that could reveal specific health conditions, medical diagnoses, or treatments.

{context}

Original comment:
{comment}

Requirements:
1. Maintain the core message and intent
2. Keep the text natural and readable
3. Remove or generalize specific health-related details
4. Do not add new information not present in the original

Rewritten comment:
"""


def generate_anonymized_data(record, model_name):
    """
    Generate anonymized version of a record.
    
    Args:
        record: Input record with original text.
        model_name: DeepSeek model name.
        
    Returns:
        Record with anonymized_response field.
    """
    original = record.get("response", "")
    if not original:
        return None

    context = make_context_tag(record.get("context", ""))

    messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
        {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
            context=context, comment=original
        )},
    ]

    anonymized = call_deepseek_api(messages, model_name=model_name, temperature=0.7)

    if anonymized:
        result = record.copy()
        result["anonymized_response"] = anonymized.strip()
        return result

    return None


def main():
    """Main entry point for data generation script."""
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    args = parser.parse_args()

    setup_logging()
    records = load_jsonl(args.input_file, args.limit)

    results = []
    logging.info(f"Generating anonymized data for {len(records)} records...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_anonymized_data, rec, args.model): i
            for i, rec in enumerate(records)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logging.error(f"Error generating data: {e}")

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Generated {len(results)} anonymized records. Saved to {args.output_file}")


if __name__ == "__main__":
    main()
