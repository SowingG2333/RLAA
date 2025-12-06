export API_KEY="your_key_her"

INPUT_SRC="data/train.jsonl"
OUTPUT_SFT="data/sft_teacher_data.jsonl"

echo "Generating SFT data..."
python src/gen_data.py \
    --input_file "$INPUT_SRC" \
    --output_file "$OUTPUT_SFT" \
    --teacher_model "deepseek-chat" \
    --workers 10