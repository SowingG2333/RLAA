#!/usr/bin/env bash
# Run RLAA pipeline for PersonalReddit
# Usage: bash PersonalReddit/script/run_rlaa.sh

MODEL_PATH="${MODEL_PATH:-paths/to/your/model}"
INPUT="PersonalReddit/data/test.jsonl"
OUTPUT="PersonalReddit/results/rlaa_output.jsonl"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

python PersonalReddit/src/run_rlaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations "$MAX_ITERATIONS"
