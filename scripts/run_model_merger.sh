exp_name='qwen3_vl_4b_snmix_selfnorm15k_grpo_analysis'

step='168'
python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

for step in {20..160..20}; do
    python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
        --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
done