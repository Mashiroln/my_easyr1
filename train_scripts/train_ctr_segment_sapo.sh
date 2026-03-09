#!/bin/bash

set -x
# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7

# === Model & Data ===
MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen3_VL/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1
data_path=/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_dynamic3k
exp_name=qwen3_vl_4b_ctr_segment_openvit_sapo

# === Environment: reward parsing & stats ===
export EXP_NAME=${exp_name}
export NAVSIM_STAT_PATH="/mnt/data/ccy/datasets/golden_navtrain/golden_de/output/stats/trajectory_stats_refine_v4.json"
# export NAVSIM_STAT_PATH_SYN=/path/to/trajectory_stats_syn.json
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text_rpc.py

# === Segment-Aware SAPO hyperparameters ===
# tau controls sech²(tau/2 * (r-1)) width: larger tau → narrower trust region
#   tau_coarse: protect coarse segment (SFT already good, ~89.7 PDMS)
#   tau_exp:    moderate learning for explanation/reasoning
#   tau_future: normal learning for refined trajectory (main RL target)
#   neg_ratio:  extra strictness for negative advantage (logit gradient diffusion fix)
TAU_COARSE=5.0    # ablation: [3.0, 5.0, 10.0]
TAU_EXP=2.0       # ablation: [1.5, 2.0, 3.0]
TAU_FUTURE=1.0    # ablation: [0.8, 1.0, 1.5]
NEG_RATIO=1.05    # ablation: [1.0, 1.05, 1.1]

# === Checkpoint resume (uncomment to continue training) ===
# LOAD_CKPT=trainer.load_checkpoint_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/${exp_name}/global_step_XX

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.loss_type=segment_sapo \
    worker.actor.tau_coarse=${TAU_COARSE} \
    worker.actor.tau_exp=${TAU_EXP} \
    worker.actor.tau_future=${TAU_FUTURE} \
    worker.actor.neg_ratio=${NEG_RATIO} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=12 \
    trainer.save_freq=20 \
    worker.actor.model.freeze_vision_tower=false
