#!/bin/bash
# ADAS single-round pipeline example.
#
# This script processes scorer CSV output from parallel inference,
# identifies dynamic (high-variance) training samples, and outputs
# a token list file for the verl dataloader.
#
# Usage:
#   1. Run parallel inference + NAVSIM scoring externally -> scorer CSVs
#   2. Run this script to filter dynamic samples
#   3. Train with: data.token_filter_file=<path_to_txt>
#
# For multi-round ADAS, repeat steps 1-3 with each new checkpoint.

set -euo pipefail
cd "$(dirname "$0")"

# ====== Configuration ======
INFER_FOLDER="/path/to/parallel_infer/output/your_experiment"
TARGET_MIN=3000
TARGET_MAX=6000
# ===========================

python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    --target_min "$TARGET_MIN" \
    --target_max "$TARGET_MAX"

# The output token list path is printed at the end.
# Use it in your training script:
#   data.token_filter_file=/path/to/group_stats_filtered_0.1.txt
