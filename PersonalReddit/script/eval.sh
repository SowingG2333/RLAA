#!/usr/bin/env bash
# Evaluate RLAA/FgAA results for PersonalReddit
# Usage: API_KEY="your_key" bash PersonalReddit/script/eval.sh

export API_KEY="${API_KEY:-your_key_here}"

INPUT="PersonalReddit/results/rlaa_output.jsonl"
OUTPUT="PersonalReddit/results/eval_score.json"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
ADVERSARY_MODEL="${ADVERSARY_MODEL:-deepseek-chat}"
WORKERS="${WORKERS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python PersonalReddit/src/eval.py \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --judge_model "$JUDGE_MODEL" \
    --adversary_model "$ADVERSARY_MODEL" \
    --workers "$WORKERS"
