import json
import argparse
import logging
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
import torch
import sys
# 导入 FgAA_API.py 中的类型提示
from typing import List, Dict, Any, Tuple, Optional

# 设置日志记录
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# --- 1. 匿名化LLM提示词 (保持不变) ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""
PROMPT_ANONYMIZER_USER_TEMPLATE = """
Below I give you a [Comment] from a user as well as [Personal Inferences] made about the user that can be made from the [Comment]. Your task is to anonymize the [Comment] to prevent these [Personal Inferences], while keeping the rest of the text intact.

[Comment]:
{user_response}

[Personal Inferences]:
{feedback}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
Then, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".
"""

# --- 2. 攻击者LLM提示词 (从 FgAA_API.py 导入逻辑) ---
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""

# 从 FgAA_API.py 复制的核心属性定义
CORE_PII_ATTRIBUTES_ALL = {"age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status", "health_issue"}

# 从 FgAA_API.py 复制的动态 Prompt 生成函数
def generate_attacker_user_prompt(user_response: str, attributes_to_protect: List[str]) -> str:
    """
    根据要保护的属性动态生成攻击者的用户提示。
    """    
    attributes_list_str = ", ".join(attributes_to_protect)
    json_keys_str = ", ".join([f'"{attr}"' for attr in attributes_to_protect])

    return f"""
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - **{attributes_list_str}**.

Guess: A single JSON object containing your best guess for the following key: **{json_keys_str}**.
"""


# --- 3. SFT 数据加载与格式化 ---

def load_and_format_sft_data(data_path: str, task: str, tokenizer: AutoTokenizer) -> Dataset:
    """加载并格式化 SFT 数据集以适应聊天模板"""
    records = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping line {i+1} in {data_path} due to JSON decode error: {e}")
    except FileNotFoundError:
        logging.error(f"SFT data file not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading SFT data from {data_path}: {e}", exc_info=True)
        sys.exit(1)


    formatted_data = []
    skipped_count = 0
    
    # 在循环外定义一次
    attacker_attributes_to_protect = sorted(list(CORE_PII_ATTRIBUTES_ALL))

    for i, record in enumerate(records):
        messages = []
        
        # Attacker 任务逻辑
        if task == "attacker":
            if "response" in record and "attacker_output_teacher" in record:
                messages.append({"role": "system", "content": PROMPT_ATTACKER_SYSTEM})
                user_content = generate_attacker_user_prompt(
                    user_response=record["response"],
                    attributes_to_protect=attacker_attributes_to_protect
                )
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": record["attacker_output_teacher"]})
            else:
                 logging.warning(f"Record {i}: Skipping for attacker SFT due to missing 'response' or 'attacker_output_teacher'.")
                 skipped_count += 1
                 continue

        # Anonymizer 任务逻辑
        elif task == "anonymizer":
            if ("response" in record and
                "attacker_inference_teacher" in record and
                "anonymizer_output_teacher" in record):

                messages.append({"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM})
                user_content = PROMPT_ANONYMIZER_USER_TEMPLATE.format(
                    user_response=record["response"],
                    feedback=record["attacker_inference_teacher"]
                )
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": record["anonymizer_output_teacher"]})
            else:
                 logging.warning(f"Record {i}: Skipping for anonymizer SFT due to missing fields ('response', 'attacker_inference_teacher', 'anonymizer_output_teacher').")
                 skipped_count += 1
                 continue
        else:
             raise ValueError(f"Unknown task: {task}")

        try:
            # [!! 修正 !!] 确保 tokenizer 在这里被正确使用
            # (注意: 这里的 'tokenizer' 是从 run_sft 传递过来的)
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            if not full_text.strip().endswith(tokenizer.eos_token):
                full_text += tokenizer.eos_token

            formatted_data.append({"text": full_text})
        except Exception as e:
            logging.error(f"Record {i}: Error applying chat template: {e}. Messages: {messages}")
            skipped_count += 1

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} records during formatting.")

    if not formatted_data:
        raise ValueError(f"No valid data could be formatted for task '{task}' in {data_path}")

    logging.info(f"Successfully formatted {len(formatted_data)} records for SFT task '{task}'.")
    return Dataset.from_list(formatted_data)

# --- 4. SFT 训练逻辑 ---

def run_sft(args):
    """运行 SFT 过程"""
    # 1. 加载 Tokenizer 和模型
    model_name = args.base_model
    logging.info(f"Loading base model and tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        logging.warning(f"Tokenizer for {model_name} does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 

    # 为旧版 TRL 设置 max_seq_length
    if args.max_seq_length > tokenizer.model_max_length and tokenizer.model_max_length > 1:
        logging.warning(f"Setting max_seq_length to {args.max_seq_length}, which is > tokenizer.model_max_length {tokenizer.model_max_length}. This is OK.")
    tokenizer.model_max_length = args.max_seq_length


    # 可选: 使用 QLoRA
    bnb_config = None
    if args.use_qlora:
        logging.info("Using QLoRA (4-bit quantization).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Using compute dtype: {compute_dtype}")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    
    # [!! 修正 !!] 将 tokenizer 附加到 model 对象上
    # 旧版 SFTTrainer 会从 model.tokenizer 获取
    model.tokenizer = tokenizer 

    lora_config = None 
    
    if args.use_qlora:
         if hasattr(model, 'enable_input_require_grads'):
             model.enable_input_require_grads()
         else:
             def make_inputs_require_grad(module, input, output):
                 output.requires_grad_(True)
             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

         from peft import LoraConfig, prepare_model_for_kbit_training
         
         model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing) 

         lora_config = LoraConfig(
             r=args.lora_r,
             lora_alpha=args.lora_alpha,
             target_modules=[ 
                 "q_proj",
                 "k_proj",
                 "v_proj",
                 "o_proj",
                 "gate_proj",
                 "up_proj",
                 "down_proj",
             ],
             lora_dropout=args.lora_dropout,
             bias="none",
             task_type="CAUSAL_LM",
         )


    # 2. 加载和格式化数据
    logging.info(f"Loading and formatting data for task: {args.task}")
    # [!! 修正 !!] 确保将 tokenizer 传入数据格式化函数
    full_dataset = load_and_format_sft_data(args.data_path, args.task, tokenizer)
    logging.info(f"Full dataset loaded with {len(full_dataset)} samples.")

    if len(full_dataset) < 2:
        logging.warning("Dataset is too small to split. Using full dataset for training and evaluation.")
        train_dataset = full_dataset
        eval_dataset = full_dataset
    else:
        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logging.info(f"Dataset split: {len(train_dataset)} train samples, {len(eval_dataset)} evaluation samples.")


    training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            
            # 确保使用 'eval_strategy'
            eval_strategy="epoch" if len(eval_dataset) > 0 else "no",
            load_best_model_at_end=True if len(eval_dataset) > 0 else False,
            save_total_limit=1,
            
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            fp16=False,
            bf16=True if compute_dtype == torch.bfloat16 else False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # 4. 初始化 SFTTrainer
    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
            args=training_args,
            peft_config=lora_config,
            
            # [!! 修正 !!] 移除 'tokenizer', 'dataset_text_field', 'max_seq_length', 'packing'
            # tokenizer=tokenizer, 
            # dataset_text_field="text",
            # max_seq_length=args.max_seq_length,
            # packing=True, 
        )
    
    # 5. 开始训练
    logging.info("Starting SFT training...")
    train_result = trainer.train()
    logging.info("Training complete.")
    logging.info(f"TrainOutput: {train_result}")


    # 6. 保存最终模型
    logging.info(f"Saving final model adapter (if LoRA) or full model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # [!! 修正 !!] 确保 tokenizer 被保存
    # (注意: 这里的 'tokenizer' 仍然是我们在顶部定义的变量)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Model/Adapter and tokenizer saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT for Attacker or Anonymizer using Llama3 Chat Template")
    parser.add_argument("--task", type=str, required=True, choices=["attacker", "anonymizer"], help="Task to fine-tune for.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model identifier.")
    parser.add_argument("--data_path", type=str, default="sft_teacher_data.jsonl", help="Path to the generated teacher data JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model/adapter.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps.")

    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (common for QLoRA).")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")

    # QLoRA arguments
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

    args = parser.parse_args()

    if not args.use_qlora:
        logging.warning("Not using QLoRA. Setting learning rate to a more conservative 1e-5.")
        args.learning_rate = 1e-6
    
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = True

    run_sft(args)