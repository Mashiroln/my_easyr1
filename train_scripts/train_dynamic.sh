#!/bin/bash

set -x

# python /mnt/data/ccy/verl/scripts_train/check_gpu.py

# MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_1view_thinking_bs32_stage2_openvit
MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_1view_thinking_bs32_stage2_openvit_normalized
# MODEL_PATH=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_mix_88step_3k_span_reward/global_step_130/actor/huggingface
# MODEL_PATH=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k/global_step_88/actor/huggingface
# MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen2_5_VL/3B_navsim_poutine_cot_simp_thinking_bs32_stage2_openvit_normalized

# data_path=/mnt/data/ccy/EasyR1/data/navsim_cot1view_filter_mix_dynamic_hard_3k
data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_dynamic_6k
# data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_mix_88step_1_5k
# data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_mix_88step_3k
# data_path=/mnt/data/ccy/EasyR1/data/navsim_simp_norm_cot_dynamic_8k
# data_path=/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_mix_130step_10k

reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py


# exp_name=qwen2_5_vl_3b_navsim_grpo_balanced_analysis
# exp_name=entropy_max_analysis
# exp_name=qwen2_5_vl_3b_navsim_grpo_mix_dynamic_hard_3k
# exp_name=qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_mix_88step_3k
# exp_name=qwen2_5_vl_3b_navsim_simp_norm_cot_dynamic_8k
# exp_name=qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_mix_130step_10k_span_reward
exp_name=qwen2_5_vl_3b_navsim_adas_1x_6k

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
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10 \
    # trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_mix_130step_10k_span_reward/global_step_30 \
    # # trainer.val_only=true \
