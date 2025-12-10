exp_name='qwen2_5_vl_3b_448_navsim_norm_inputproj_cot_dynamic_2k_dapo'

# step='12'
# python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

for step in {5..10..5}; do
    python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
        --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
done