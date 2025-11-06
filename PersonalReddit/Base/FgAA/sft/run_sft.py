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

# 设置日志记录
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# 匿名化LLM提示词
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
**Then, ONLY return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:".**
"""

# 攻击者LLM提示词
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER_TEMPLATE = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
**JSON Value Rules (CRITICAL):**
1.  age: MUST be an accurate integer (e.g., 35, not "35" or "30-40") or string "unknown".
2.  sex: MUST be one of the following exact strings: "male", "female", or "unknown".
3.  relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced", "widowed", or "unknown".
4.  income_level: MUST be one of the following exact strings: "low", "middle", "high", "very high", or "unknown".
5.  city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China") if the city is unknown, or "unknown".
6.  education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner") or "unknown".
7.  **If you cannot reasonably infer an attribute with high confidence, you MUST use the string value "unknown" instead of randomly guessing an answer.**
"""

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
    for i, record in enumerate(records):
        messages = []
        if task == "attacker":
            # 检查 Attacker SFT 需要的字段
            if "original_comment" in record and "attacker_output_teacher" in record:
                messages.append({"role": "system", "content": PROMPT_ATTACKER_SYSTEM})
                # 构建 User Prompt 内容
                user_content = PROMPT_ATTACKER_USER_TEMPLATE.format(user_response=record["original_comment"])
                messages.append({"role": "user", "content": user_content})
                # 添加教师模型的输出作为 Assistant 回答
                messages.append({"role": "assistant", "content": record["attacker_output_teacher"]})
            else:
                 logging.warning(f"Record {i}: Skipping for attacker SFT due to missing 'original_comment' or 'attacker_output_teacher'.")
                 skipped_count += 1
                 continue # 跳过此记录

        elif task == "anonymizer":
            # 检查 Anonymizer SFT 需要的字段
            # **重要假设**: 'anonymizer_output_teacher' 必须包含教师模型生成的解释和 # 分隔符
            if ("original_comment" in record and
                "attacker_inference_teacher" in record and
                "anonymizer_output_teacher" in record):

                messages.append({"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM})
                # 构建 User Prompt 内容，将 feedback 替换为 attacker_inference_teacher
                user_content = PROMPT_ANONYMIZER_USER_TEMPLATE.format(
                    user_response=record["original_comment"],
                    feedback=record["attacker_inference_teacher"] # 使用推理作为 feedback
                )
                messages.append({"role": "user", "content": user_content})
                # 添加教师模型的完整输出 (含解释和#) 作为 Assistant 回答
                messages.append({"role": "assistant", "content": record["anonymizer_output_teacher"]})
            else:
                 logging.warning(f"Record {i}: Skipping for anonymizer SFT due to missing fields ('original_comment', 'attacker_inference_teacher', 'anonymizer_output_teacher').")
                 skipped_count += 1
                 continue # 跳过此记录
        else:
             raise ValueError(f"Unknown task: {task}")

        try:
            # add_generation_prompt=False 因为我们已经包含了 assistant 的回答
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            # 确保文本以 EOS token 结束，SFTTrainer 可能需要这个
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

def run_sft(args):
    """运行 SFT 过程"""
    # 1. 加载 Tokenizer 和模型
    model_name = args.base_model # 使用参数指定基础模型
    logging.info(f"Loading base model and tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 设置 pad token - Llama3 通常将 pad token 设为 eos token
    if tokenizer.pad_token is None:
        logging.warning(f"Tokenizer for {model_name} does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # 可能还需要设置 padding_side="right" or "left" depending on training stability

    # 可选: 使用 QLoRA 进行更高效的微调
    bnb_config = None
    if args.use_qlora:
        logging.info("Using QLoRA (4-bit quantization).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, # Often recommended
        )

    # 确定 torch_dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Using compute dtype: {compute_dtype}")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, # 如果启用 QLoRA
        device_map="auto", # 自动分配到 GPU
        trust_remote_code=True,
        torch_dtype=compute_dtype # 使用计算类型加载
    )

    if args.use_qlora:
         # For QLoRA stability
         if hasattr(model, 'enable_input_require_grads'):
             model.enable_input_require_grads()
         else:
             def make_inputs_require_grad(module, input, output):
                 output.requires_grad_(True)
             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

         # 配置 PEFT (LoRA)
         from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
         model = prepare_model_for_kbit_training(model) # 准备量化模型进行训练

         # LoRA 配置 - 需要根据 Llama3 架构调整 target_modules
         # Common targets for Llama: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
         lora_config = LoraConfig(
             r=args.lora_r,
             lora_alpha=args.lora_alpha,
             target_modules=[ # These might need adjustment for Llama3
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
         # model = get_peft_model(model, lora_config) # <--- [!! 关键修改 !!] 注释掉
         # model.print_trainable_parameters()          # <--- [!! 关键修改 !!] 注释掉
    else:
         lora_config = None # No PEFT if not using QLoRA/LoRA


# 2. 加载和格式化数据
    logging.info(f"Loading and formatting data for task: {args.task}")
    # 在这里传入 tokenizer 以便应用聊天模板
    full_dataset = load_and_format_sft_data(args.data_path, args.task, tokenizer)
    logging.info(f"Full dataset loaded with {len(full_dataset)} samples.")

    # 将数据集拆分为训练集和验证集 (80/20)
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42) # seed 保证可复现
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
            save_strategy="epoch", # 每轮 Epoch 保存一次
            
            eval_strategy="epoch",   # 告诉训练器在每个 epoch 结束时评估
            load_best_model_at_end=True,     # 训练结束后，自动加载 eval_loss 最低的那个模型
            save_total_limit=1,              # 只保留最好的1个 checkpoint，节省空间

            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch", # 优化器
            fp16=False, # 使用 bf16
            bf16=True, # 使用 bfloat16
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine", # 余弦学习率调度
            report_to="tensorboard", # 报告到 TensorBoard
            gradient_checkpointing=True, # 节省显存，但训练稍慢
        )

    # 4. 初始化 SFTTrainer
    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=lora_config if args.use_qlora else None,  # 传入 LoRA 配置
        )
    
    # 5. 开始训练
    logging.info("Starting SFT training...")
    train_result = trainer.train()
    logging.info("Training complete.")
    logging.info(f"TrainOutput: {train_result}")


    # 6. 保存最终模型
    logging.info(f"Saving final model adapter (if LoRA) or full model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # 保存 tokenizer
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
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps (effective batch size = batch_size * num_gpus * grad_accum).")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")

    # QLoRA arguments
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

    args = parser.parse_args()

    run_sft(args)