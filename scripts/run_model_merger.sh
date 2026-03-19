exp_name='qwen3_vl_4b_prn103kv4_dynamic3k_prefillmix_grpo_v2'

# step='84'
# python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

for step in {20..110..10}; do
    python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
        --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
done