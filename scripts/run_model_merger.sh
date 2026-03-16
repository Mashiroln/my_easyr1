exp_name='qwen3_vl_4b_ctr_segment_openvit_sapo_3x'

# step='84'
# python /mnt/data/ccy/EasyR1/scripts/model_merger.py --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor

for step in {140..200..20}; do
    python /mnt/data/ccy/EasyR1/scripts/model_merger.py \
        --local_dir /mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor
done