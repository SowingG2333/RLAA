#!/usr/bin/env bash
# Evaluate RLAA/FgAA results for reddit-self-disclosure
# Usage: API_KEY="your_key" bash reddit-self-disclosure/script/eval.sh

export API_KEY="${API_KEY:-your_key_here}"

INPUT="reddit-self-disclosure/results/rlaa_output.jsonl"
OUTPUT="reddit-self-disclosure/results/eval_score.json"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
ADVERSARY_MODEL="${ADVERSARY_MODEL:-deepseek-chat}"
WORKERS="${WORKERS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python reddit-self-disclosure/src/eval.py \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --judge_model "$JUDGE_MODEL" \
    --adversary_model "$ADVERSARY_MODEL" \
    --workers "$WORKERS"
