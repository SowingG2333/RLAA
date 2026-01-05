#!/usr/bin/env bash
# Run FgAA (SFT mode) for PersonalReddit
# Usage: bash PersonalReddit/script/run_fgaa_sft.sh

MODEL_DIR="${MODEL_DIR:-PersonalReddit/models}"
ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-$MODEL_DIR/sft_attacker_merged}"
ANONYMIZER_MODEL_PATH="${ANONYMIZER_MODEL_PATH:-$MODEL_DIR/sft_anonymizer_merged}"
INPUT="PersonalReddit/data/test.jsonl"
OUTPUT="PersonalReddit/results/fgaa_sft_output.jsonl"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Running FgAA (SFT mode)..."
python PersonalReddit/src/run_fgaa.py \
    --attacker_model_path "$ATTACKER_MODEL_PATH" \
    --anonymizer_model_path "$ANONYMIZER_MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations "$MAX_ITERATIONS"

echo "Done. Results saved to $OUTPUT"
