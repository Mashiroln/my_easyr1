exp_name='qwen3_vl_4b_openvit_refine_prn103k_v3tov4_3k_sdr'

step='56'
python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

# for step in {20..50..10}; do
#     python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
#         --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
# done