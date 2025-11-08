python /root/autodl-tmp/RLAA/PersonalReddit/Base/FgAA/FgAA_llama.py \
--model_name /root/autodl-tmp/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
--input_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/data/PR_resplit/test.jsonl \
--output_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/output/5-iter/fgaa_qwen2.5-2.jsonl \
--log_file /root/autodl-tmp/RLAA/PersonalReddit/Eval/output/5-iter/log/fgaa_qwen2.5-2.log \
--log_level DEBUG