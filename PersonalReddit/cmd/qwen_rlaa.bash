python /root/autodl-tmp/RLAA/PersonalReddit/RLAA_llama.py \
--model_name /root/autodl-tmp/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
--input_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/data/PR_resplit/test.jsonl \
--output_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/output/3-iter/rlaa_qwen2.5.json \
--log_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/output/3-iter/log/rlaa_qwen2.5.log \
--log_level DEBUG