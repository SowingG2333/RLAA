import argparse
import logging
import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompts (保持不变)
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

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".

JSON Value Rules:
1. age: MUST be an accurate integer (e.g., 35, not "35" or "30-40").
2. sex: MUST be one of the following exact strings: "male" or "female".
3. relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced" or "widowed".
4. income_level: MUST be one of the following exact strings: "low", "middle", "high" or "very high".
5. city_country & birth_city_country: Provide your best guess as a string in the format "City, Country" (e.g., "Shanghai, China") or just "Country" (e.g., "China").
6. education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner").
7. If you cannot reasonably infer an attribute, you MUST use the string value "unknown" instead of randomly guessing an answer.
"""

def format_dataset(data_path, task, tokenizer):
    """读取生成的 JSONL 并格式化为 Chat Template"""
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_path}")
        exit(1)
            
    formatted = []
    for rec in data:
        messages = []
        if task == "attacker":
            if "original_comment" not in rec or "attacker_output_teacher" not in rec: continue
            messages = [
                {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                {"role": "user", "content": PROMPT_ATTACKER_USER.format(user_response=rec["original_comment"])},
                {"role": "assistant", "content": rec["attacker_output_teacher"]}
            ]
        elif task == "anonymizer":
            if "original_comment" not in rec or "attacker_inference_teacher" not in rec: continue
            messages = [
                {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
                {"role": "user", "content": PROMPT_ANONYMIZER_USER.format(
                    user_response=rec["original_comment"], 
                    feedback=rec["attacker_inference_teacher"]
                )},
                {"role": "assistant", "content": rec["anonymizer_output_teacher"]}
            ]
            
        # 应用 Chat Template
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted.append({"text": text})
        except Exception as e:
            logging.warning(f"Skipping record due to formatting error: {e}")
        
    return Dataset.from_list(formatted)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 for Attacker/Anonymizer tasks")
    parser.add_argument("--task", required=True, choices=["attacker", "anonymizer"], help="Task type")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model path")
    parser.add_argument("--data_path", required=True, help="Path to input jsonl data")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints")
    parser.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit total amount of checkpoints")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value")

    args = parser.parse_args()

    # 1. Tokenizer
    logging.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if not tokenizer.pad_token: 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # 2. Model & Quantization
    bnb_config = None
    if args.use_qlora:
        logging.info("Using QLoRA configuration...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # QLoRA Prep
    peft_config = None
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    # 3. Data
    logging.info("Formatting dataset...")
    dataset = format_dataset(args.data_path, args.task, tokenizer)
    split = dataset.train_test_split(test_size=0.1)

    # 4. Trainer
    logging.info("Initializing SFTTrainer...")
    
    # 检测是否支持 bf16
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,                 
            logging_steps=args.logging_steps,                 
            save_total_limit=args.save_total_limit,           
            warmup_ratio=args.warmup_ratio,                   
            fp16=not is_bf16_supported,
            bf16=is_bf16_supported,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            group_by_length=True,
        )
    )

    logging.info(f"Starting training for {args.task}...")
    trainer.train()
    
    logging.info(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Training complete.")

if __name__ == "__main__":
    main()