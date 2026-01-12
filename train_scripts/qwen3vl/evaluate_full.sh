#!/bin/bash

set -x

# python /mnt/data/ccy/verl/scripts_train/check_gpu.py

MODEL_PATH=/mnt/data/ccy/VLA_train/output/train/Qwen3-VL/8B_navsim_poutine_cot_1view_thinking_bs32_stage2_openvit_norm_qwen3v2/checkpoint-9588/
data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_full_easyr1
exp_name=qwen3_vl_8b_navsim_normtrajtext_cot_full
reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py


# For traj only:
# data.format_prompt=/mnt/data/ccy/EasyR1/examples/format_prompt/trajonly.jinja

# load path
# trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k/global_step_88

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@train \
    data.format_prompt=None \
    data.val_batch_size=4096 \
    worker.actor.global_batch_size=512 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optim.lr=0.0 \
    worker.actor.optim.weight_decay=0.0 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=9 \
    trainer.val_freq=1 \
    trainer.save_freq=-1 \
    # trainer.val_only=true \