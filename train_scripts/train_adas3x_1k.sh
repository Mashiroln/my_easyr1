#!/bin/bash

set -x

# python /mnt/data/ccy/verl/scripts_train/check_gpu_quiet.py --kill-quiet

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_1view_thinking_bs32_stage2_openvit_normalized2
data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_dynamic_1k_005_adas2x3ksdr_step110
reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

# exp_name=qwen2_5_vl_3b_navsim_adas_1x_6k_sdr

exp_name=qwen2_5_vl_3b_navsim_adas_3x_1k
export EXP_NAME=${exp_name}

python3 -m verl.trainer.main \
      config=examples/config_vla.yaml \
      data.train_files=${data_path}@train \
      data.val_files=${data_path}@test \
      data.format_prompt=None \
      data.max_response_length=2048 \
      worker.actor.model.model_path=${MODEL_PATH} \
      worker.rollout.tensor_parallel_size=1 \
      worker.rollout.n=16 \
      worker.reward.reward_function=${reward_function_path}:compute_score_fast \
      trainer.experiment_name=${exp_name} \
      trainer.n_gpus_per_node=8 \
      trainer.total_epochs=50 \
      trainer.save_freq=10 \
      trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_adas_2x_3k/global_step_110