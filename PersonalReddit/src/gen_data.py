import argparse
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import setup_logging, load_jsonl, call_deepseek_api

# Prompts
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not. 
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not. 
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
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
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response must follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".

JSON Value Rules:
1. age: must be an accurate integer (e.g., 35, not "35" or "30-40").
2. sex: must be one of the following exact strings: "male" or "female".
3. relationship_status: must be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced" or "widowed".
4. income_level: must be one of the following exact strings: "low", "middle", "high" or "very high".
5. city_country & birth_city_country: provide your best guess as a string in the format "City, Country" (e.g., "Shanghai, China") or just "Country" (e.g., "China").
6. education & occupation: provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner").
7. If you cannot reasonably infer an attribute, you must use the string value "unknown" instead of randomly guessing an answer.
"""

def process_single_record(record, model_name):
    """Generate teacher data for one record (attacker reasoning + anonymizer output)."""
    original_text = record.get("response")
    if not original_text: return None

    sft_record = {"original_comment": original_text}

    # 1. Teacher attacker: generate reasoning
    msgs_att = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=original_text)}]
    
    att_output = call_deepseek_api(msgs_att, model_name=model_name, temperature=0.1)
    if not att_output: return None
    
    sft_record["attacker_output_teacher"] = att_output
    
    # Extract the inference block for the next step
    import re
    match = re.search(r"Inference:(.*?)(Guess:|$)", att_output, re.DOTALL | re.IGNORECASE)
    inference = match.group(1).strip() if match else "No inference found."
    sft_record["attacker_inference_teacher"] = inference

    # 2. Teacher anonymizer: generate anonymized text
    msgs_anon = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
                 {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(user_response=original_text, feedback=inference)}]
    
    anon_output = call_deepseek_api(msgs_anon, model_name=model_name, temperature=0.5)
    if not anon_output: return None
    
    sft_record["anonymizer_output_teacher"] = anon_output

    # Extract plain text for downstream SFT
    parts = anon_output.split('#', 1)
    clean_anon = parts[1].strip() if len(parts) > 1 else anon_output.strip()
    sft_record["anonymizer_text_clean"] = clean_anon

    return sft_record

def main():
    parser = argparse.ArgumentParser(description="Generate SFT data using DeepSeek as teacher.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--teacher_model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    setup_logging()
    records = load_jsonl(args.input_file, args.limit)
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_record, rec, args.teacher_model): i for i, rec in enumerate(records)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating SFT Data"):
            res = future.result()
            if res: results.append(res)
            
    # Save results
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Saved {len(results)} SFT records to {args.output_file}")
    except Exception as e:
        logging.error(f"Save failed: {e}")

if __name__ == "__main__":
    main()