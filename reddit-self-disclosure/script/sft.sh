#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATA_PATH="${DATA_PATH:-./data/sft_teacher_data.jsonl}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
MODEL_DIR="${MODEL_DIR:-./models}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-./src/train.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-./src/merge.py}"

LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-10}"
SEQ_LEN="${SEQ_LEN:-1024}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

mkdir -p "$CKPT_DIR" "$MODEL_DIR"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: teacher data not found at $DATA_PATH" >&2
    echo "Hint: run script/gen_data.sh first." >&2
    exit 1
fi

echo "---------------------------------------"
echo ">>> [1/4] Training Attacker..."
echo "---------------------------------------"
python $TRAIN_SCRIPT \
    --task attacker \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$CKPT_DIR/attacker_lora" \
    --use_qlora \
    --learning_rate $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --max_seq_length $SEQ_LEN \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --save_total_limit 1

echo "---------------------------------------"
echo ">>> [2/4] Merging Attacker..."
echo "---------------------------------------"
python $MERGE_SCRIPT \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CKPT_DIR/attacker_lora" \
    --output_path "$MODEL_DIR/sft_attacker_merged"

echo "---------------------------------------"
echo ">>> [3/4] Training Anonymizer..."
echo "---------------------------------------"
python $TRAIN_SCRIPT \
    --task anonymizer \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$CKPT_DIR/anonymizer_lora" \
    --use_qlora \
    --learning_rate $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --max_seq_length $SEQ_LEN \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --save_total_limit 1

echo "---------------------------------------"
echo ">>> [4/4] Merging Anonymizer..."
echo "---------------------------------------"
python $MERGE_SCRIPT \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CKPT_DIR/anonymizer_lora" \
    --output_path "$MODEL_DIR/sft_anonymizer_merged"

echo "======================================="
echo "SFT Pipeline Complete!"
echo "Merged models are ready at:"
echo "  Attacker:   $MODEL_DIR/sft_attacker_merged"
echo "  Anonymizer: $MODEL_DIR/sft_anonymizer_merged"
echo "======================================="
