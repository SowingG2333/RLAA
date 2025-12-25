#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${API_KEY:-}" ]]; then
    echo "ERROR: API_KEY is not set. Export API_KEY first." >&2
    exit 1
fi

INPUT_FILE="${INPUT_FILE:-results/rlaa_output.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-results/eval_score.json}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
ADVERSARY_MODEL="${ADVERSARY_MODEL:-deepseek-chat}"
WORKERS="${WORKERS:-10}"
LIMIT="${LIMIT:-}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

ARGS=(
    --input_file "$INPUT_FILE"
    --output_file "$OUTPUT_FILE"
    --judge_model "$JUDGE_MODEL"
    --adversary_model "$ADVERSARY_MODEL"
    --workers "$WORKERS"
)

if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

python src/eval.py "${ARGS[@]}"
