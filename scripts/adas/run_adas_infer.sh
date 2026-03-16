#!/bin/bash

# ===== To modify =====
PROJECT_ROOT="/path/to/"

MODEL_PATH="$PROJECT_ROOT/EasyR1/checkpoints/sft/your_model_name" # to load tokenizer correctly, don't change when start a new outer loop
data_path="$PROJECT_ROOT/EasyR1/data/QA_navtrain_poutine_style_full"
exp_name=your_adas_exp_name
load_path="YOUR_CHECKPOINT_TO_ADAS_INFER" # if first outer loop, set Null; e.g. /path/to/your_checkpoint/global_step_x

reward_function_path="$PROJECT_ROOT/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py"

export EXP_NAME=$exp_name
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag


python -m verl.trainer.main_adas \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=null \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 \
    worker.rollout.n=32 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    # trainer.load_checkpoint_path=${load_path} \
