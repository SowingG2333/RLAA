"""
SFT Training script using LoRA for privacy-preserving text rewriting.

Fine-tunes a language model on anonymized data using parameter-efficient training.

Usage:
    python reddit-self-disclosure/src/train.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_file reddit-self-disclosure/data/sft_train.jsonl \
        --output_dir reddit-self-disclosure/models/sft_lora
"""

import argparse
import logging
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from utils import setup_logging

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

PROMPT_TEMPLATE = """You are a privacy-preserving text rewriter. Rewrite the following text to protect personal information while maintaining the original meaning.

{context}

Original text:
{original}

Rewritten text:
{anonymized}"""


def load_training_data(file_path, limit=None):
    """
    Load and prepare training data.
    
    Args:
        file_path: Path to JSONL training file.
        limit: Maximum number of records to load.
        
    Returns:
        List of training examples.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            if "response" in record and "anonymized_response" in record:
                context = record.get("context", "")
                context_str = f"Context: {context}" if context else ""
                
                text = PROMPT_TEMPLATE.format(
                    context=context_str,
                    original=record["response"],
                    anonymized=record["anonymized_response"],
                )
                data.append({"text": text})
    return data


def create_dataset(data, tokenizer, max_length=512):
    """
    Create tokenized dataset.
    
    Args:
        data: List of training examples.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        
    Returns:
        Tokenized Dataset object.
    """
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="LoRA SFT Training")
    parser.add_argument("--model_name", required=True, help="Base model name/path")
    parser.add_argument("--train_file", required=True, help="Training JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--limit", type=int, default=None, help="Limit training samples")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Loading model: {args.model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        target_modules=DEFAULT_LORA_CONFIG["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    logging.info(f"Loading training data from: {args.train_file}")
    data = load_training_data(args.train_file, args.limit)
    logging.info(f"Loaded {len(data)} training examples")

    dataset = create_dataset(data, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    logging.info("Starting training...")
    trainer.train()

    # Save LoRA weights
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
