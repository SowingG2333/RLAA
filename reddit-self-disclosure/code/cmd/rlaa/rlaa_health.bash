python /root/autodl-tmp/RLAA/reddit-self-disclosure/code/RLAA_llama.py \
--input_file /root/autodl-tmp/RLAA/reddit-self-disclosure/dataset/processed/health_test.jsonl \
--output_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/output/rlaa/health_anony_qwen2.5.jsonl \
--log_file /root/autodl-tmp/RLAA/reddit-self-disclosure/code/log/rlaa/health_qwen2.5.log \
--log_level DEBUG \
--protect_attribute health_issue \
--model_name /root/autodl-tmp/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28