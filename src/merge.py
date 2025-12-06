import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", required=True, help="Base model path or ID")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_path", required=True, help="Where to save the merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving to: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()