export API_KEY="your_key_here"

TARGET_FILE="results/rlaa_output.jsonl"
EVAL_OUT="results/eval_score.json"

python src/eval.py \
    --input_file "$TARGET_FILE" \
    --output_file "$EVAL_OUT" \
    --judge_model "deepseek-chat" \
    --adversary_model "deepseek-chat"