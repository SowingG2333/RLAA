# 一个单独的 merge_lora.py 脚本
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2" 
adapter_path = "/root/autodl-tmp/RLAA/Base/AF/sft/ckpt/attacker/checkpoint-648"
merged_model_path = "/root/autodl-tmp/RLAA/Base/AF/sft/ckpt/attacker/merged"

print(f"Loading base model: {base_model_name}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"Loading adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging model...")
model = model.merge_and_unload() # <-- 关键步骤：合并

print(f"Saving merged model to: {merged_model_path}")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("Merge complete.")