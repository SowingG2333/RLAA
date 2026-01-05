#!/usr/bin/env bash
# Generate SFT training data for PersonalReddit
# Usage: API_KEY="your_key" bash PersonalReddit/script/gen_data.sh

export API_KEY="${API_KEY:-your_key_here}"

INPUT="PersonalReddit/data/train.jsonl"
OUTPUT="PersonalReddit/data/sft_teacher_data.jsonl"
MODEL="${MODEL:-deepseek-chat}"
WORKERS="${WORKERS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Generating SFT data..."
python PersonalReddit/src/gen_data.py \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --model "$MODEL" \
    --workers "$WORKERS"

echo "Done. SFT data saved to $OUTPUT"
