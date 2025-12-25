#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-./models}"
ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-$MODEL_DIR/sft_attacker_merged}"
ANONYMIZER_MODEL_PATH="${ANONYMIZER_MODEL_PATH:-$MODEL_DIR/sft_anonymizer_merged}"
INPUT_FILE="${INPUT_FILE:-data/test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-results/fgaa_sft_output.jsonl}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

if [[ ! -d "$ATTACKER_MODEL_PATH" ]]; then
    echo "ERROR: attacker model not found at $ATTACKER_MODEL_PATH" >&2
    echo "Hint: run script/sft.sh first, or set ATTACKER_MODEL_PATH." >&2
    exit 1
fi
if [[ ! -d "$ANONYMIZER_MODEL_PATH" ]]; then
    echo "ERROR: anonymizer model not found at $ANONYMIZER_MODEL_PATH" >&2
    echo "Hint: run script/sft.sh first, or set ANONYMIZER_MODEL_PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Running FgAA (SFT mode)..."
python src/run_fgaa.py \
    --attacker_model_path "$ATTACKER_MODEL_PATH" \
    --anonymizer_model_path "$ANONYMIZER_MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_iterations "$MAX_ITERATIONS"

echo "Done. Results saved to $OUTPUT_FILE"
