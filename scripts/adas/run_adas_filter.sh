#!/bin/bash
# ADAS single-round pipeline example.
#
# Processes scorer CSV output, identifies dynamic training samples,
# and outputs a token list for the verl dataloader.
#
# Usage:
#   1. Run parallel inference + NAVSIM scoring externally -> scorer CSVs
#   2. Run this script to filter dynamic samples
#   3. Train with: data.token_filter_file=<path_to_txt>
#
# For multi-round ADAS, repeat steps 1-3 with each new checkpoint.

set -euo pipefail
cd "$(dirname "$0")"

INFER_FOLDER="/mnt/data/ccy/VLA_train/parallel_infer/output/4B_navsim_ctr_correct_multi_style_prn103kv4_stage2_from_pn103kv3_use_refinev4_norm_stage1"

python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    -p 0.2 \
    --conf 0.1
