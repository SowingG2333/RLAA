import sys
import argparse
import logging
from tqdm import tqdm

from local_llm import LocalModelHandler
from utils import setup_logging, load_jsonl, save_jsonl, parse_attacker_output

# Prompts
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

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
In the next line, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
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
1. health_issue: must be a string (e.g., "psoriasis", "diabetes").
2. If you cannot reasonably infer it, you must use the string value "unknown".
"""


def process_record(attacker_model, anonymizer_model, record, max_iterations):
    original_text = record.get("response", "")
    if not original_text:
        return {**record, "anonymized_response": "", "meta": {"status": "skipped"}}

    current_text = original_text

    def is_safe(guess_json):
        if not guess_json or "error" in guess_json:
            return False
        v = str(guess_json.get("health_issue", "unknown")).lower()
        return v == "unknown"

    # Round 0: Initial Attack
    msgs_att = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=current_text)},
    ]
    attack_out = attacker_model.generate(msgs_att, temperature=0.1)
    parsed = parse_attacker_output(attack_out)

    if is_safe(parsed["guess_json"]):
        return {**record, "anonymized_response": current_text, "meta": {"status": "success_original"}}

    feedback = parsed["inference"]

    for i in range(max_iterations):
        # 1) Anonymize
        msgs_anon = [
            {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
            {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(user_response=current_text, feedback=feedback)},
        ]
        raw_anon = anonymizer_model.generate(msgs_anon, temperature=0.5)
        parts = raw_anon.split("#", 1)
        current_text = parts[1].strip() if len(parts) > 1 else raw_anon.strip()

        # 2) Attack
        msgs_att = [
            {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
            {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=current_text)},
        ]
        attack_out = attacker_model.generate(msgs_att, temperature=0.1)
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
    parser = argparse.ArgumentParser(description="Run FgAA Baseline (Naive or SFT modes) for health self-disclosure")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=5)

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_path", type=str, help="Single model path (Naive mode)")
    model_group.add_argument("--attacker_model_path", type=str, help="Attacker model path (SFT mode)")
    model_group.add_argument("--anonymizer_model_path", type=str, help="Anonymizer model path (SFT mode)")

    args = parser.parse_args()
    setup_logging()

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

    records = load_jsonl(args.input_file)
    results = []

    logging.info(f"Starting processing of {len(records)} records...")
    for rec in tqdm(records):
        results.append(process_record(attacker_model, anonymizer_model, rec, args.max_iterations))
        if len(results) % 10 == 0:
            save_jsonl(results, args.output_file)

    save_jsonl(results, args.output_file)
    logging.info("Done.")


if __name__ == "__main__":
    main()
