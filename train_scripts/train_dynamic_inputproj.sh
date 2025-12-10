#!/bin/bash

set -x

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_distill_box448_inputproj_thinking_traj_pretrain_stg2_openvit

data_path=/mnt/data/ccy/EasyR1/data/navsim_inputproj_norm_cot_dynamic_6k

exp_name=qwen2_5_vl_3b_448_navsim_norm_inputproj_cot_dynamic_6k

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
    data.max_response_length=2048 \
    data.max_pixels=352000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=8 \
    # trainer.val_only=true \