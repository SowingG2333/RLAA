export API_KEY='sk-2a19754e8dcf4ac1972be6753b4efc15'

python /root/autodl-tmp/RLAA/reddit-self-disclosure/code/eval.py \
--input_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/output/rlaa/age_anony.jsonl \
--output_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/results/rlaa/age_eval.json \
--protect_attribute age