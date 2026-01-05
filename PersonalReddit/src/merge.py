"""
Merge LoRA weights with base model.

Combines the fine-tuned LoRA adapters with the base model for inference.

Usage:
    python PersonalReddit/src/merge.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --lora_path PersonalReddit/models/sft_lora \
        --output_path PersonalReddit/models/merged
"""

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import setup_logging


def main():
    """Main entry point for model merging."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", required=True, help="Base model name/path")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA weights")
    parser.add_argument("--output_path", required=True, help="Output path for merged model")
    args = parser.parse_args()

    setup_logging()

    logging.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logging.info(f"Loading LoRA weights from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    logging.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    logging.info(f"Saving merged model to: {args.output_path}")
    merged_model.save_pretrained(args.output_path)

    # Save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    logging.info("Merge complete!")


if __name__ == "__main__":
    main()
