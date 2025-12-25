#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
INPUT_FILE="${INPUT_FILE:-data/test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-results/rlaa_output.jsonl}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

python src/run_rlaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_iterations "$MAX_ITERATIONS"
