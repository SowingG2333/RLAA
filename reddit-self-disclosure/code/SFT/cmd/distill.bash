python /root/autodl-tmp/RLAA/reddit-self-disclosure/code/SFT/sft_distill.py \
--task anonymizer \
--base_model /root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
--data_path /root/autodl-tmp/RLAA/reddit-self-disclosure/code/SFT/sft_data.jsonl \
--output_dir /root/autodl-tmp/RLAA/reddit-self-disclosure/code/SFT/ckpt/ano \
--epochs 10 \
--batch_size 4 \
--use_qlora