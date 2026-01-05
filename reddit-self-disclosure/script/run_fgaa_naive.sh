#!/usr/bin/env bash
# Run FgAA (Naive mode) for reddit-self-disclosure
# Usage: bash reddit-self-disclosure/script/run_fgaa_naive.sh

MODEL_PATH="${MODEL_PATH:-path/to/your/fgaa_model}"
INPUT="reddit-self-disclosure/data/test.jsonl"
OUTPUT="reddit-self-disclosure/results/fgaa_naive_output.jsonl"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python reddit-self-disclosure/src/run_fgaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations "$MAX_ITERATIONS"
