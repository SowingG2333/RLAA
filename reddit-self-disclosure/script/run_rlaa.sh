#!/usr/bin/env bash
# Run RLAA pipeline for reddit-self-disclosure
# Usage: bash reddit-self-disclosure/script/run_rlaa.sh

MODEL_PATH="${MODEL_PATH:-path/to/your/model}"
INPUT="reddit-self-disclosure/data/test.jsonl"
OUTPUT="reddit-self-disclosure/results/rlaa_output.jsonl"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python reddit-self-disclosure/src/run_rlaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations "$MAX_ITERATIONS"
