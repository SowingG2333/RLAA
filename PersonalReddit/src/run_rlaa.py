"""
RLAA (Reinforcement Learning Adversarial Anonymization) main script.

This script implements the iterative anonymization loop:
1. Attacker: Attempts to infer personal attributes from text
2. Arbitrator: Validates the attacker's reasoning
3. Anonymizer: Modifies text to prevent valid leaks

Usage:
    python PersonalReddit/src/run_rlaa.py \
        --model_path /path/to/model \
        --input_file PersonalReddit/data/test.jsonl \
        --output_file PersonalReddit/results/rlaa_output.jsonl
"""

import argparse
import json
import logging
from tqdm import tqdm

from local_llm import LocalModelHandler
from utils import (
    setup_logging,
    load_jsonl,
    save_jsonl,
    parse_attacker_output,
    extract_first_json,
    make_context_tag,
    log_full_messages,
    log_io_block,
    is_sanity_check_failed,
    filter_actionable_leaks,
    format_feedback_for_anonymizer,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response must follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, current_city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "current_city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".

JSON Value Rules:
1. age: must be an accurate integer (e.g., 35, not "35" or "30-40").
2. sex: must be one of the following exact strings: "male" or "female".
3. relationship_status: must be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced" or "widowed".
4. income_level: must be one of the following exact strings: "low", "middle", "high" or "very high".
5. current_city_country & birth_city_country: Provide your best guess as a string in the format "City, Country" (e.g., "Shanghai, China") or just "Country" (e.g., "China").
6. education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner").
"""

PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.

Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""

PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user as well as a list of [Leaked Concepts]. Your task is to edit the [Comment] to neutralize these concepts, while keeping the rest of the text intact.

How to Neutralize:
1. If a concept has "validity_level: high", it's likely a direct leak (e.g., specific name, location). Focus on generalizing the specific word(s) identified in "reasoning_evidence" (e.g., "Paris" -> "a major city").
2. If a concept has "validity_level: medium", it might be a leak based on style, tone, or topic choice (e.g., implying gender, age). Focus on rephrasing the phrase(s) in "reasoning_evidence" to be more neutral or generic. Crucially, ensure the core meaning and intent of the original phrase are preserved after rephrasing.
3. Try to address them with a minimal change if possible. Prioritize addressing concepts with "high" validity.

[Comment]:
{user_response}

[Leaked Concepts]:
{feedback}

In a first line, briefly explain the changes you will make to invalidate the attacker's inference.
In the next line, write a single #
In the last line, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
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

# =============================================================================
# Core Processing Functions
# =============================================================================

def get_attacker_guess(model, text, record_index=None, iteration=None):
    """
    Generate attacker's guess for personal attributes.
    
    Args:
        model: The LLM handler.
        text: Input comment text.
        record_index: Current record index for logging.
        iteration: Current iteration number for logging.
        
    Returns:
        Dictionary with 'inference' and 'guess_json' keys.
    """
    tag = f"{make_context_tag(record_index, iteration)}[attacker]"
    log_io_block(tag, "INPUT_TEXT", text or "")

    messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
        {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=text)},
    ]
    log_full_messages(tag, messages)
    response = model.generate(messages, temperature=0.1)
    log_io_block(tag, "OUTPUT_RAW", response or "")
    
    return parse_attacker_output(response)


def get_arbitrator_validation(model, text, inference, attributes, record_index=None, iteration=None):
    """
    Validate attacker's reasoning using the arbitrator.
    
    Args:
        model: The LLM handler.
        text: Current comment text.
        inference: Attacker's inference block.
        attributes: List of attributes to protect.
        record_index: Current record index for logging.
        iteration: Current iteration number for logging.
        
    Returns:
        List of validated leak dictionaries.
    """
    if not attributes:
        return []

    tag = f"{make_context_tag(record_index, iteration)}[arbitrator]"
    log_io_block(tag, "INPUT_COMMENT", text or "")
    log_io_block(tag, "INPUT_INFERENCE", inference or "")
    log_io_block(tag, "INPUT_ATTRIBUTES_JSON", json.dumps(attributes, ensure_ascii=False, indent=2))

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
    log_full_messages(tag, messages)
    response = model.generate(messages, temperature=0.0)
    log_io_block(tag, "OUTPUT_RAW", response or "")
    
    return extract_first_json(response) or []


def run_anonymizer(model, text, feedback, record_index=None, iteration=None):
    """
    Run the anonymizer to neutralize leaked concepts.
    
    Args:
        model: The LLM handler.
        text: Current comment text.
        feedback: JSON string of leaked concepts.
        record_index: Current record index for logging.
        iteration: Current iteration number for logging.
        
    Returns:
        Anonymized text.
    """
    tag = f"{make_context_tag(record_index, iteration)}[anonymizer]"
    log_io_block(tag, "INPUT_COMMENT", text or "")
    log_io_block(tag, "INPUT_FEEDBACK", feedback or "")

    messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
        {
            "role": "user",
            "content": PROMPT_ANONYMIZER_USER.format(user_response=text, feedback=feedback),
        },
    ]
    log_full_messages(tag, messages)
    response = model.generate(messages, temperature=0.5)
    log_io_block(tag, "OUTPUT_RAW", response or "")
    
    # Extract anonymized text after the '#' separator
    parts = response.split("#", 1)
    anonymized = parts[1].strip() if len(parts) > 1 else response.strip()
    log_io_block(tag, "OUTPUT_ANONYMIZED_TEXT", anonymized or "")
    
    return anonymized


def process_single_record(model, record, max_iterations, record_index=None):
    """
    Process a single record through the RLAA pipeline.
    
    Args:
        model: The LLM handler.
        record: Input record dictionary.
        max_iterations: Maximum anonymization iterations.
        record_index: Current record index for logging.
        
    Returns:
        Processed record with anonymized response and metadata.
    """
    original_text = record.get("response", "")
    if not original_text:
        return {**record, "status": "skipped"}

    current_text = original_text
    meta = {"iteration_history": [], "status": "max_iterations"}

    # Attributes to protect
    attributes_to_protect = [
        "age", "education", "income_level", "current_city_country",
        "birth_city_country", "occupation", "relationship_status", "sex",
    ]

    # Initial attack
    attack_res = get_attacker_guess(model, current_text, record_index=record_index, iteration=0)

    # Validate attack
    validated_leaks = get_arbitrator_validation(
        model, current_text, attack_res.get("inference", ""),
        attributes_to_protect, record_index=record_index, iteration=0,
    )

    actionable_leaks = filter_actionable_leaks(validated_leaks)
    has_leaks = len(actionable_leaks) > 0

    meta["iteration_history"].append({
        "iteration": 0,
        "text": current_text,
        "attacker": attack_res,
        "leaks": validated_leaks,
        "actionable_leaks": actionable_leaks,
    })

    # If no leaks, return original
    if not has_leaks:
        return {
            **record,
            "anonymized_response": current_text,
            "meta": {**meta, "status": "success_original"},
        }

    # Iterative anonymization loop
    for i in range(max_iterations):
        feedback = format_feedback_for_anonymizer(actionable_leaks)

        # Anonymize
        previous_text = current_text
        current_text = run_anonymizer(
            model, current_text, feedback, record_index=record_index, iteration=i + 1
        )

        # Sanity check
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

        # Attack again
        attack_res = get_attacker_guess(
            model, current_text, record_index=record_index, iteration=i + 1
        )

        # Validate again
        validated_leaks = get_arbitrator_validation(
            model, current_text, attack_res.get("inference", ""),
            attributes_to_protect, record_index=record_index, iteration=i + 1,
        )
        actionable_leaks = filter_actionable_leaks(validated_leaks)
        has_leaks = len(actionable_leaks) > 0

        meta["iteration_history"].append({
            "iteration": i + 1,
            "text": current_text,
            "attacker": attack_res,
            "leaks": validated_leaks,
            "actionable_leaks": actionable_leaks,
        })

        # Check if anonymization succeeded
        if not has_leaks:
            return {
                **record,
                "anonymized_response": current_text,
                "meta": {
                    **meta,
                    "status": "success",
                    "final_attacker_guess": attack_res.get("guess_json"),
                    "final_leaked_attributes": [
                        f"{x.get('attribute')} ({x.get('validity_level')})"
                        for x in actionable_leaks
                    ],
                },
            }

    # Max iterations reached
    return {
        **record,
        "anonymized_response": current_text,
        "meta": {
            **meta,
            "final_attacker_guess": attack_res.get("guess_json"),
            "final_leaked_attributes": [
                f"{x.get('attribute')} ({x.get('validity_level')})" for x in actionable_leaks
            ],
        },
    }


def main():
    """Main entry point for RLAA script."""
    parser = argparse.ArgumentParser(description="Run RLAA (PersonalReddit)")
    parser.add_argument("--model_path", required=True, help="Path to the local model")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max anonymization iterations")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log_file", type=str, default="run_rlaa.log", help="Log file path")
    args = parser.parse_args()

    setup_logging(log_level_str=args.log_level, log_file=args.log_file)
    model = LocalModelHandler(args.model_path, load_in_4bit=True)
    records = load_jsonl(args.input_file)

    results = []
    for idx, record in enumerate(tqdm(records)):
        res = process_single_record(model, record, args.max_iterations, record_index=idx)
        results.append(res)
        # Periodic checkpoint save
        if len(results) % 10 == 0:
            save_jsonl(results, args.output_file)

    save_jsonl(results, args.output_file)


if __name__ == "__main__":
    main()
