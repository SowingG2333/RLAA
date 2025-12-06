MODEL_DIR="path/to/model_dir"
INPUT="data/test.jsonl"
OUTPUT="results/fgaa_sft_output.jsonl"

mkdir -p $(dirname "$OUTPUT")

echo "Running Inference using SFT models..."
python src/run_fgaa.py \
    --attacker_model_path "$MODEL_DIR/sft_attacker_merged" \
    --anonymizer_model_path "$MODEL_DIR/sft_anonymizer_merged" \
    --input_file "$INPUT" \
    --output_file "$OUTPUT" \
    --max_iterations 10

echo "Done. Results saved to $OUTPUT"