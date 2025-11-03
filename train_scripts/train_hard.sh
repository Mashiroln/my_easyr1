#!/bin/bash

set -x
python /mnt/data/ccy/verl/scripts_train/check_gpu.py

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_1view_thinking_bs32_stage2_openvit # replace it with your local file path
data_path=/mnt/data/ccy/EasyR1/data/navsim_cot1view_3k_easyr1/hard_curriculum__50-30-20__cap40__n2186
exp_name=qwen2_5_vl_3b_navsim_grpo_hard
reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \

python /mnt/data/ccy/verl/scripts_train/check_gpu.py
data_path=/mnt/data/ccy/EasyR1/data/navsim_cot1view_3k_easyr1/edge_cluster__k25_low40__cap200__n3000
exp_name=qwen2_5_vl_3b_navsim_grpo_edge
python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \