MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
INPUT="data/test.jsonl"
OUTPUT="results/rlaa_output.jsonl"

mkdir -p $(dirname "$OUTPUT")

python src/run_rlaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations 10