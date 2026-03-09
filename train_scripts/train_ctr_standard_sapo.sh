#!/bin/bash
set -x

# === Model & Data (same as segment_sapo) ===

export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7

python /mnt/data/ccy/verl/scripts_train/check_gpu.py

MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen3_VL/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1
data_path=/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_dynamic3k
exp_name=qwen3_vl_4b_ctr_standard_sapo

# === Environment ===
export EXP_NAME=${exp_name}
export NAVSIM_STAT_PATH="/mnt/data/ccy/datasets/golden_navtrain/golden_de/output/stats/trajectory_stats_refine_v4.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text_rpc.py

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    data.rollout_batch_size=448 \
    data.val_batch_size=896 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=224 \
    worker.actor.loss_type=sapo \
    worker.actor.tau_positive=1.0 \
    worker.actor.tau_negative=1.05 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=7 \
    trainer.total_epochs=12 \
    trainer.save_freq=20
