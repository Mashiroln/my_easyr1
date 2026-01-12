#!/bin/bash

set -x

# python /mnt/data/ccy/verl/scripts_train/check_gpu.py

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen3_VL/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1
data_path=/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_dynamic3k
exp_name=qwen3_vl_4b_openvit_refine_prn103k_v3tov4_3k_sdr

reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

# For traj only:
# data.format_prompt=/mnt/data/ccy/EasyR1/examples/format_prompt/trajonly.jinja

# load path
# trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k/global_step_88

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=8 \
    worker.actor.model.freeze_vision_tower=false \




