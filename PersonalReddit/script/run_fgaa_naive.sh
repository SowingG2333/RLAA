MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
INPUT="RLAA/data/test.jsonl"
OUTPUT="results/fgaa_naive_output.jsonl"

mkdir -p $(dirname "$OUTPUT")

python src/run_fgaa.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations 10