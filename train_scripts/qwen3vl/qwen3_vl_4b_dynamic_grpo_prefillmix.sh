#!/bin/bash

set -x

# GRPO with Prefill Mix training.
#
# Usage: nohup bash this_script.sh &
# The script waits for GPUs to become idle (vLLM inference done or hung),
# kills any lingering vLLM processes, then starts training.

# ── Step 0: Wait for GPU idle / kill hung vLLM ──
# exit 0 = GPUs idle normally, exit 2 = quiet-kill triggered (vLLM hung)
# python3 /mnt/data/ccy/verl/scripts_train/check_gpu_quiet.py --kill-quiet
# rc=$?
# if [ $rc -ne 0 ] && [ $rc -ne 2 ]; then
#     echo "GPU monitor exited with unexpected code $rc, aborting."
#     exit 1
# fi
# echo "GPUs clear (exit=$rc), starting training..."

# ── Step 1: Paths ──
MODEL_PATH=/mnt/data/ccy/VLA_train/output/Qwen3_VL/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1
data_path=/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_full
exp_name=qwen3_vl_4b_prn103kv4_dynamic3k_prefillmix_grpo_v2
export EXP_NAME=$exp_name

# Standard ADAS (no-prefill) txt + Prefill ADAS jsonl → list for dataloader
# Both from the same checkpoint output directory
INFER_DIR=/mnt/data/ccy/VLA_train/parallel_infer/output/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1
STANDARD_TXT=${INFER_DIR}/generations_full_filtered_0.3.txt
PREFILL_JSONL=${INFER_DIR}/generations_full_prefill_filtered_0.5.jsonl
TOKEN_FILTER_FILE="[${STANDARD_TXT},${PREFILL_JSONL}]"

export NAVSIM_STAT_PATH="/mnt/data/ccy/datasets/golden_navtrain/expand_tools/trajectory_stats_refine_v4.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

reward_function_path=/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text_rpc.py

# ── Step 2: Train ──
python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    "data.token_filter_file=${TOKEN_FILTER_FILE}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10 \
    trainer.save_freq=20 \

python3 /mnt/data/ccy/vllm_guard.py 