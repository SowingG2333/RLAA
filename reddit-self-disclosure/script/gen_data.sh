#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${API_KEY:-}" ]]; then
    echo "ERROR: API_KEY is not set. Export API_KEY first." >&2
    exit 1
fi

INPUT_FILE="${INPUT_FILE:-data/train.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-data/sft_teacher_data.jsonl}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-chat}"
WORKERS="${WORKERS:-10}"
LIMIT="${LIMIT:-}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Generating SFT teacher data..."

ARGS=(
    --input_file "$INPUT_FILE"
    --output_file "$OUTPUT_FILE"
    --teacher_model "$TEACHER_MODEL"
    --workers "$WORKERS"
)

if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

python src/gen_data.py "${ARGS[@]}"
