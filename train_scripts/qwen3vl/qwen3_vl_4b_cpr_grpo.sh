#!/bin/bash

set -x

# python /mnt/data/ccy/verl/scripts_train/check_gpu.py

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen3_VL/4B_navsim_cot_CPR_bs32_stage2_openvit_norm
data_path=/mnt/data/ccy/EasyR1/data/CPR103k_v4_filter_dynamic_5k
exp_name=qwen3_vl_4b_cpr_grpo_analysis

# Override trajectory stats paths (main + optional syn).
# export NAVSIM_STAT_PATH=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/trajectory_stats_train.json
export NAVSIM_STAT_PATH="/mnt/data/ccy/datasets/golden_navtrain/golden_de/output/stats/trajectory_stats_refine_v4.json"
# export NAVSIM_STAT_PATH_SYN=/path/to/trajectory_stats_syn.json

export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_cpr_text_waypoint

# reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py
reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text_rpc.py

# For traj only:
# data.format_prompt=/mnt/data/ccy/EasyR1/examples/format_prompt/cpr.jinja

# load path
# trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k/global_step_88

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=/mnt/data/ccy/EasyR1/examples/format_prompt/cpr.jinja \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=8 \
    trainer.save_freq=20




