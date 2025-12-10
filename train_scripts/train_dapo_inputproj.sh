#!/bin/bash

set -x

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_distill_box448_inputproj_thinking_traj_pretrain_stg2_openvit

data_path=/mnt/data/ccy/EasyR1/data/navsim_inputproj_norm_cot_dynamic_2k

exp_name=qwen2_5_vl_3b_448_navsim_norm_inputproj_cot_dynamic_2k_dapo

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
    data.max_response_length=1024 \
    data.max_pixels=352000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=32 \
    worker.actor.optim.lr_warmup_ratio=0.03 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.rollout.n=16 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=overall \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=4 \
    trainer.save_freq=5 \
    # trainer.val_only=true \