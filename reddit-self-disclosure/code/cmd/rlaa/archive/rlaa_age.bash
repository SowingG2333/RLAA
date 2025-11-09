python /root/autodl-tmp/RLAA/reddit-self-disclosure/code/RLAA_llama.py \
--model_name /root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
--input_file /root/autodl-tmp/RLAA/reddit-self-disclosure/dataset/processed/age.jsonl \
--output_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/output/rlaa/age_anony.jsonl \
--log_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/log/rlaa/age.log \
--log_level DEBUG \
--protect_attribute age