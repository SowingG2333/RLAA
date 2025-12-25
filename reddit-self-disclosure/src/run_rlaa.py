import argparse
import json
from tqdm import tqdm

from local_llm import LocalModelHandler
from utils import (
    setup_logging, 
    load_jsonl, 
    save_jsonl, 
    parse_attacker_output, 
    extract_first_json
)

# Prompts
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to infer the author's likely health issue(s) based on the [Comment].

[Comment]:
{user_response}

Your response must follow this format exactly:

Inference: Provide your detailed reasoning for the health_issue attribute guess.

Guess: A single JSON object containing your best guess for the "health_issue" key.
"""

PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.

Example: 'I have psoriasis' -> 'I have a skin condition' is valid.
Example: removing specific medication names is valid.
Example: If feedback indicates 'ya know' implies a certain age group, changing 'ya know' is valid neutral rephrasing. Rewriting the entire sentence with a completely different style is NOT desired.
"""
PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user as well as a list of [Leaked Concepts]. Your task is to edit the [Comment] to neutralize these concepts, while keeping the rest of the text intact.

How to Neutralize:
1. If a concept has validity_level: "high", it's likely a direct leak (e.g., a specific disease or medication name). Focus on generalizing or removing the specific word(s) identified in "reasoning_evidence".
2. If a concept has validity_level: "medium", it might be a leak based on strong topic cues. Focus on rephrasing the phrase(s) in "reasoning_evidence" to be more neutral or generic, while preserving meaning.
3. Try to address them with a minimal change if possible. Prioritize addressing concepts with validity_level: "high".

[Comment]:
{user_response}

[Leaked Concepts]:
{feedback}

In a first step, briefly explain the changes you will make to invalidate the attacker's inference.
In a new line, write a single #
In the next line, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
"""

PROMPT_ARBITRATOR_SYSTEM = """
You are an expert Arbitrator. Your job is to validate an Attacker's reasoning and extract the core concepts behind valid leaks. You must output a single, valid JSON list.
"""
PROMPT_ARBITRATOR_USER = """
Below is an [Attacker Reasoning] block, a [Attributes to protect] and the [Comment] text.

Your task is to analyze each attribute and determine the validity level of the Attacker's reasoning for it.
A reasoning's validity level is defined as:
1. "high": Reasoning is based on a direct statement or an undeniable logical deduction from the [Comment] (e.g., [Comment] says "I live in Paris", Attacker guesses "Paris, France").
2. "medium": Reasoning is based on a strong stylistic cue, specific topic, or strong pattern (e.g., "use of specific slang like 'mate'", "detailed discussion of fashion design", "use of specific engineering jargon"). Use this for gender/occupation cues that are based on specific text rather than vague feelings.
3. "low": Reasoning is based on a vague stereotype or weak association (e.g., "polite tone implies female", "using 'ahaha' implies young"). These are generally invalid.
4. "invalid": Reasoning is based on information not found in the [Comment] or is a clear hallucination.

For EACH leaked attribute, provide a JSON object with the following fields:
- "attribute": The name of the leaked attribute (e.g., "age").
- "validity_level": The validity level as a string: "high", "medium", "low", or "invalid".
- "reasoning_evidence": The specific phrase(s) from the [Comment] that the Attacker used.
- "leaked_concept": If validity is "high" or "medium", describe the concept that needs to be neutralized (e.g., "Author works in fashion industry" or "Author lives in Paris" or "Author uses male-coded slang").
- "validation_notes": Explain why you chose that validity level (e.g., "Medium: Based on specific stylistic cues 'mate' and 'reckon' which are statistically correlated." or "Low: Vague guess based on tone.").

Return only a single JSON list containing these objects.

[Comment]:
{current_comment}

[Attributes to protect]:
{attributes_to_protect}

[Attacker Reasoning]:
{attacker_inference_block}

[Arbitrator's JSON Output]:
"""

def is_sanity_check_failed(new_comment: str, old_comment: str, min_len: int = 20):
    if not new_comment or len(new_comment.strip()) < min_len:
        return f"catastrophic_anonymizer_failure: output too short (len={len((new_comment or '').strip())})"
    if new_comment.strip() == old_comment.strip():
        return "anonymizer_stuck: output identical to previous iteration"
    return None


def filter_actionable_leaks(validated_leaks):
    if not validated_leaks:
        return []
    if isinstance(validated_leaks, dict):
        validated_leaks = [validated_leaks]
    if not isinstance(validated_leaks, list):
        return []

    actionable = []
    for item in validated_leaks:
        if not isinstance(item, dict):
            continue
        validity = str(item.get("validity_level", "")).lower()
        if validity in ("high"):
            actionable.append(item)
    return actionable


def format_feedback_for_anonymizer(actionable_leaks):
    normalized = []
    for item in actionable_leaks or []:
        evidence = item.get("reasoning_evidence", [])
        if evidence is None:
            evidence = []
        if isinstance(evidence, str):
            evidence = [evidence]

        normalized.append(
            {
                "attribute": item.get("attribute"),
                "validity_level": item.get("validity_level"),
                "leaked_concept": item.get("leaked_concept"),
                "evidence": evidence,
                "reasoning_evidence": evidence,
                "validation_notes": item.get("validation_notes"),
            }
        )
    return json.dumps(normalized, ensure_ascii=False, indent=2)

def get_attacker_guess(model, text: str):
    messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=text)},
    ]
    response = model.generate(messages, temperature=0.1)
    return parse_attacker_output(response)

def get_arbitrator_validation(model, text: str, inference: str, attributes):
    if not attributes:
        return []
    messages = [
        {"role": "system", "content": PROMPT_ARBITRATOR_SYSTEM},
        {
            "role": "user",
            "content": PROMPT_ARBITRATOR_USER.format(
                current_comment=text,
                attacker_inference_block=inference,
                attributes_to_protect=json.dumps(attributes),
            ),
        },
    ]
    response = model.generate(messages, temperature=0.0)
    return extract_first_json(response) or []

def run_anonymizer(model, text: str, feedback: str):
    messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
        {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(user_response=text, feedback=feedback)},
    ]
    response = model.generate(messages, temperature=0.5)
    parts = response.split("#", 1)
    return parts[1].strip() if len(parts) > 1 else response.strip()

def process_single_record(model, record, max_iterations: int):
    original_text = record.get("response", "")
    if not original_text:
        return {**record, "status": "skipped"}

    attributes_to_protect = ["health_issue"]

    current_text = original_text
    meta = {"iteration_history": [], "status": "max_iterations"}

    attack_res = get_attacker_guess(model, current_text)
    validated_leaks = get_arbitrator_validation(model, current_text, attack_res.get("inference", ""), attributes_to_protect)

    actionable_leaks = filter_actionable_leaks(validated_leaks)

    has_leaks = len(actionable_leaks) > 0 and str(attack_res.get("guess_json", {}).get("health_issue", "unknown")).lower() != "unknown"

    meta["iteration_history"].append(
        {
            "iteration": 0,
            "text": current_text,
            "attacker": attack_res,
            "leaks": validated_leaks,
            "actionable_leaks": actionable_leaks,
        }
    )

    if not has_leaks:
        return {**record, "anonymized_response": current_text, "meta": {**meta, "status": "success_original"}}

    for i in range(max_iterations):
        feedback = format_feedback_for_anonymizer(actionable_leaks)

        previous_text = current_text
        current_text = run_anonymizer(model, current_text, feedback)

        sanity_fail_reason = is_sanity_check_failed(current_text, previous_text)
        if sanity_fail_reason:
            return {
                **record,
                "anonymized_response": previous_text,
                "meta": {
                    **meta,
                    "status": "model_error",
                    "error": sanity_fail_reason,
                    "final_attacker_guess": attack_res.get("guess_json"),
                },
            }

        attack_res = get_attacker_guess(model, current_text)
        validated_leaks = get_arbitrator_validation(model, current_text, attack_res.get("inference", ""), attributes_to_protect)
        actionable_leaks = filter_actionable_leaks(validated_leaks)
        has_leaks = len(actionable_leaks) > 0 and str(attack_res.get("guess_json", {}).get("health_issue", "unknown")).lower() != "unknown"

        meta["iteration_history"].append(
            {
                "iteration": i + 1,
                "text": current_text,
                "attacker": attack_res,
                "leaks": validated_leaks,
                "actionable_leaks": actionable_leaks,
            }
        )

        if not has_leaks:
            return {
                **record,
                "anonymized_response": current_text,
                "meta": {
                    **meta,
                    "status": "success",
                    "final_attacker_guess": attack_res.get("guess_json"),
                    "final_leaked_attributes": [
                        f"{x.get('attribute')} ({x.get('validity_level')})" for x in actionable_leaks
                    ],
                },
            }

    return {
        **record,
        "anonymized_response": current_text,
        "meta": {
            **meta,
            "final_attacker_guess": attack_res.get("guess_json"),
            "final_leaked_attributes": [f"{x.get('attribute')} ({x.get('validity_level')})" for x in actionable_leaks],
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Run RLAA (health self-disclosure)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_iterations", type=int, default=10)
    args = parser.parse_args()

    setup_logging()
    model = LocalModelHandler(args.model_path, load_in_4bit=True)
    records = load_jsonl(args.input_file)

    results = []
    for record in tqdm(records):
        res = process_single_record(model, record, args.max_iterations)
        results.append(res)
        if len(results) % 10 == 0:
            save_jsonl(results, args.output_file)

    save_jsonl(results, args.output_file)

if __name__ == "__main__":
    main()