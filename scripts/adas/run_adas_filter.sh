#!/bin/bash
# ADAS standard (no-prefill) filter pipeline.
#
# Auto-tune mode: sweeps diversity thresholds and picks the one closest
# to --target_min / --target_max sample count range.
#
# Usage:
#   1. Run parallel inference + NAVSIM scoring externally -> scorer CSVs
#   2. Run this script to filter dynamic samples
#   3. Train with: data.token_filter_file=<path_to_txt>
#
# For multi-round ADAS, repeat steps 1-3 with each new checkpoint.

set -euo pipefail
cd "$(dirname "$0")"

INFER_FOLDER="${1:-/mnt/data/ccy/VLA_train/parallel_infer/output/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1}"

python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    -p 0.1 \
    --conf 0.1
