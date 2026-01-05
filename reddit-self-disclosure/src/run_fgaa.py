"""
FgAA (Fine-grained Adversarial Anonymization) baseline script for health self-disclosure.

Supports two modes:
- Naive mode: Single shared model for both attacker and anonymizer
- SFT mode: Separate fine-tuned models for attacker and anonymizer

Usage:
    # Naive mode
    python reddit-self-disclosure/src/run_fgaa.py \
        --model_path /path/to/model \
        --input_file reddit-self-disclosure/data/test.jsonl \
        --output_file reddit-self-disclosure/results/fgaa_output.jsonl
    
    # SFT mode
    python reddit-self-disclosure/src/run_fgaa.py \
        --attacker_model_path /path/to/attacker \
        --anonymizer_model_path /path/to/anonymizer \
        --input_file reddit-self-disclosure/data/test.jsonl \
        --output_file reddit-self-disclosure/results/fgaa_sft_output.jsonl
"""

import sys
import argparse
import logging
from tqdm import tqdm

from local_llm import LocalModelHandler
from utils import (
    setup_logging,
    load_jsonl,
    save_jsonl,
    parse_attacker_output,
    make_context_tag,
    log_full_messages,
    log_io_block,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.

Example: 'I have psoriasis' -> 'I have a skin condition' is valid.
Example: removing specific medication names is valid.
"""

PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user as well as [Personal Inferences] made about the user that can be made from the [Comment]. Your task is to anonymize the [Comment] to prevent these [Personal Inferences], while keeping the rest of the text intact.

[Comment]:
{user_response}

[Personal Inferences]:
{feedback}

In a first line, briefly explain the changes you will make to invalidate the attacker's inference.
In the next line, write a single #
In the last line, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to infer the author's likely health issue(s) based on the [Comment].

[Comment]:
{user_response}

Your response must follow this format exactly:

Inference: Provide your detailed reasoning for the health_issue guess.

Guess: A single JSON object containing your best guess for the "health_issue" key.

JSON Value Rules:
1. health_issue: must be a string (e.g., "lung cancer").
2. If you cannot reasonably infer it, you must use the string value "unknown".
"""


def process_record(attacker_model, anonymizer_model, record, max_iterations, record_index=None):
    """
    Process a single record through the FgAA pipeline.
    
    Args:
        attacker_model: Model for generating attacks.
        anonymizer_model: Model for anonymization.
        record: Input record dictionary.
        max_iterations: Maximum anonymization iterations.
        record_index: Current record index for logging.
        
    Returns:
        Processed record with anonymized response and metadata.
    """
    original_text = record.get("response", "")
    if not original_text:
        return {**record, "anonymized_response": "", "meta": {"status": "skipped"}}

    current_text = original_text

    def is_safe(guess_json):
        """Check if health_issue is 'unknown'."""
        if not guess_json or "error" in guess_json:
            return False
        v = str(guess_json.get("health_issue", "unknown")).lower()
        return v == "unknown"

    # Round 0: Initial Attack
    msgs_att = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=current_text)},
    ]
    tag_att0 = f"{make_context_tag(record_index, 0)}[attacker]"
    log_io_block(tag_att0, "INPUT_TEXT", current_text)
    log_full_messages(tag_att0, msgs_att)
    attack_out = attacker_model.generate(msgs_att, temperature=0.1)
    log_io_block(tag_att0, "OUTPUT_RAW", attack_out)
    parsed = parse_attacker_output(attack_out)

    if is_safe(parsed["guess_json"]):
        return {**record, "anonymized_response": current_text, "meta": {"status": "success_original"}}

    feedback = parsed["inference"]

    # Iteration Loop
    for i in range(max_iterations):
        # 1. Anonymize
        msgs_anon = [
            {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
            {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
                user_response=current_text, feedback=feedback
            )},
        ]
        tag_anon = f"{make_context_tag(record_index, i + 1)}[anonymizer]"
        log_io_block(tag_anon, "INPUT_TEXT", current_text)
        log_io_block(tag_anon, "INPUT_FEEDBACK", feedback)
        log_full_messages(tag_anon, msgs_anon)
        raw_anon = anonymizer_model.generate(msgs_anon, temperature=0.5)
        log_io_block(tag_anon, "OUTPUT_RAW", raw_anon)

        # Extract text after '#' separator
        parts = raw_anon.split("#", 1)
        current_text = parts[1].strip() if len(parts) > 1 else raw_anon.strip()
        log_io_block(tag_anon, "OUTPUT_ANONYMIZED_TEXT", current_text)

        # 2. Attack
        msgs_att = [
            {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
            {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=current_text)},
        ]
        tag_att = f"{make_context_tag(record_index, i + 1)}[attacker]"
        log_io_block(tag_att, "INPUT_TEXT", current_text)
        log_full_messages(tag_att, msgs_att)
        attack_out = attacker_model.generate(msgs_att, temperature=0.1)
        log_io_block(tag_att, "OUTPUT_RAW", attack_out)
        parsed = parse_attacker_output(attack_out)

        if is_safe(parsed["guess_json"]):
            return {
                **record,
                "anonymized_response": current_text,
                "meta": {"status": "success", "iterations": i + 1},
            }

        feedback = parsed["inference"]

    return {**record, "anonymized_response": current_text, "meta": {"status": "max_iterations"}}


def main():
    """Main entry point for FgAA script."""
    parser = argparse.ArgumentParser(description="Run FgAA Baseline (health self-disclosure)")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max anonymization iterations")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log_file", type=str, default="run_fgaa.log", help="Log file path")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_path", type=str, help="Single model path (Naive mode)")
    model_group.add_argument("--attacker_model_path", type=str, help="Attacker model path (SFT mode)")
    model_group.add_argument("--anonymizer_model_path", type=str, help="Anonymizer model path (SFT mode)")

    args = parser.parse_args()
    setup_logging(log_level_str=args.log_level, log_file=args.log_file)

    # Model loading
    if args.model_path:
        logging.info(f"Running in NAIVE mode (Shared Model: {args.model_path})")
        shared_model = LocalModelHandler(args.model_path, load_in_4bit=True)
        attacker_model = shared_model
        anonymizer_model = shared_model
    elif args.attacker_model_path and args.anonymizer_model_path:
        logging.info("Running in SFT mode (Separate Models)")
        logging.info(f"Loading Attacker: {args.attacker_model_path}")
        attacker_model = LocalModelHandler(args.attacker_model_path, load_in_4bit=True)
        logging.info(f"Loading Anonymizer: {args.anonymizer_model_path}")
        anonymizer_model = LocalModelHandler(args.anonymizer_model_path, load_in_4bit=True)
    else:
        logging.error("Invalid arguments. Provide either --model_path OR both --attacker_model_path and --anonymizer_model_path")
        sys.exit(1)

    # Process data
    records = load_jsonl(args.input_file)
    results = []

    logging.info(f"Starting processing of {len(records)} records...")
    for idx, rec in enumerate(tqdm(records)):
        results.append(process_record(
            attacker_model, anonymizer_model, rec, args.max_iterations, record_index=idx
        ))
        # Periodic checkpoint save
        if len(results) % 10 == 0:
            save_jsonl(results, args.output_file)

    save_jsonl(results, args.output_file)
    logging.info("Done.")


if __name__ == "__main__":
    main()
