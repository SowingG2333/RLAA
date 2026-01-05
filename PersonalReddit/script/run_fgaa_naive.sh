#!/usr/bin/env bash
# Run FgAA (Naive mode) for PersonalReddit
# Usage: bash PersonalReddit/script/run_fgaa_naive.sh

MODEL_PATH="${MODEL_PATH:-path/to/your/model}"
INPUT="PersonalReddit/data/test.jsonl"
OUTPUT="PersonalReddit/results/fgaa_naive_output.jsonl"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python PersonalReddit/src/run_fgaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations "$MAX_ITERATIONS"
