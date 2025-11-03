#!/bin/bash

set -x

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_1view_thinking_bs32_stage1_freezezit # replace it with your local file path
data_path=/mnt/data/ccy/EasyR1/data/navsim_cot1view_3k_easyr1/pdms_balanced__60-30-10__cap25__n3000
# exp_name=qwen2_5_vl_3b_navsim_grpo_balanced_analysis
# exp_name=entropy_max_analysis
exp_name=qwen2_5_vl_3b_navsim_grpo_balanced_earlystop_analysis
reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15 \
    # trainer.val_only=true \