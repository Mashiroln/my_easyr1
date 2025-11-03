exp_name='qwen2_5_vl_3b_navsim_action_k12_dynamic_6k'
step='96'
python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

# for step in {10..90..10}; do
#     python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
#         --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
# done