#!/bin/bash
set -x

# 8-GPU minimal-batch debug launcher.
# Goal: avoid 1-GPU OOM while reproducing NaN/crash quickly and collecting HW/driver evidence.

# Baseline GRPO with the SAME model/data as SAPO experiments
# Purpose: verify GRPO still works before debugging SAPO

pkill -9 VLLM
pkill -9 llamafact

# Record run start time so kernel logs can be sliced precisely for this run.
RUN_START_ISO="$(date -Is)"
export RUN_START_ISO
export NCCL_P2P_DISABLE=1

# -------- Debug / crash evidence collection --------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

# NCCL settings for better crash reporting.
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

timestamp() { date +"%Y%m%d_%H%M%S"; }

collect_logs() {
    local exit_code="$1"
    local log_dir="$2"
    mkdir -p "$log_dir" || true

    {
        echo "timestamp=$(date -Is)"
        echo "pwd=$PWD"
        echo "user=$(id -u -n 2>/dev/null || true) uid=$(id -u 2>/dev/null || true)"
        echo "hostname=$(hostname 2>/dev/null || true)"
        echo "exit_code=$exit_code"
        echo "run_start_iso=$RUN_START_ISO"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "NCCL_DEBUG=$NCCL_DEBUG"
        echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
    } >"$log_dir/summary.txt" 2>&1 || true

    (uname -a >"$log_dir/uname.txt") 2>&1 || true
    (ulimit -a >"$log_dir/ulimit.txt") 2>&1 || true

    # GPU snapshots
    (nvidia-smi -L >"$log_dir/nvidia-smi-L.txt") 2>&1 || true
    (nvidia-smi >"$log_dir/nvidia-smi.txt") 2>&1 || true
    # Note: avoid -d filters here because supported sections vary across driver versions.
    (nvidia-smi -q >"$log_dir/nvidia-smi-q.txt") 2>&1 || true
    (nvidia-smi topo -m >"$log_dir/nvidia-smi-topo.txt") 2>&1 || true
    (nvidia-smi nvlink -s >"$log_dir/nvlink.txt") 2>&1 || true

    # Kernel / driver logs (may require privileges; ignore failures)
    (dmesg -T | tail -n 400 >"$log_dir/dmesg_tail.txt") 2>&1 || true
    (journalctl -k -n 400 --no-pager >"$log_dir/journalctl-k_tail.txt") 2>&1 || true
    (journalctl -k --since "$RUN_START_ISO" --no-pager >"$log_dir/journalctl-k_since_run.txt") 2>&1 || true

    # Optional: DCGM quick diag (may be slow; ignore failures)
    if command -v dcgmi >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            (timeout 600 dcgmi diag -r 1 >"$log_dir/dcgmi_diag_r1.txt") 2>&1 || true
        else
            (dcgmi diag -r 1 >"$log_dir/dcgmi_diag_r1.txt") 2>&1 || true
        fi
    fi

    # Python package versions (keep short)
    (python3 - <<'PY'
import sys
print('python', sys.version)
try:
    import torch
    print('torch', torch.__version__)
    print('cuda', torch.version.cuda)
    print('cudnn', torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
except Exception as e:
    print('torch import failed:', repr(e))
try:
    import vllm
    print('vllm', getattr(vllm, '__version__', 'unknown'))
except Exception as e:
    print('vllm import failed:', repr(e))
PY
    >"$log_dir/python_versions.txt") 2>&1 || true
}

LOG_ROOT=${LOG_ROOT:-/mnt/data/ccy/EasyR1/debug/hw_debug_logs}
RUN_TAG="$(timestamp)"

on_exit() {
    local exit_code=$?
    # EXP_NAME may be set later; fall back to script name.
    local name="${EXP_NAME:-ctr_grpo_baseline_8gpu_debug}"
    local log_dir="$LOG_ROOT/${name}_${RUN_TAG}"
    collect_logs "$exit_code" "$log_dir"
}
trap on_exit EXIT

MODEL_PATH=/mnt/data/ccy/VLA_train/output/train/Qwen3-VL-CTR/4B_navsim_ctr_correct_multi_style_prn103kv4_stage2_from_pn103kv3_use_refinev4_norm_stage1
data_path=/mnt/data/ccy/EasyR1/data/CTR_multistyle_subset/3k_dynamic
exp_name=qwen3_vl_4b_ctr_grpo_baseline_check_multistyle

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
    data.rollout_batch_size=128 \
    data.val_batch_size=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.loss_type=default \
    worker.actor.fsdp.enable_full_shard=false \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=2 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.max_steps=3 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1
