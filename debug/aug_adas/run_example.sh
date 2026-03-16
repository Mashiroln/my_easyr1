#!/bin/bash
# ADAS数据增强流程示例脚本

# 配置参数
# 1) 推荐：直接用 infer_folder 指向 parallel_infer 的输出目录（prefill/no-prefill 都支持）
INFER_FOLDER="/mnt/data/ccy/VLA_train/parallel_infer/output/qwen3_vl_4b_ctr_segment_openvit_sapo_2x_120step"

# 2) 你要从哪个源数据集里“按 token 过滤出子集”
SOURCE_DATASET="/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_full/"
OUTPUT_NAME="qwen3vl_4b_prn103kv4_norm_cot_ssapo_adas2x_120step"

# （可选）如果你要跑 Step1 navsim_data 准备，才需要这个 INPUT_JSON
INPUT_JSON="/mnt/data/ccy/agentic_writing/26eccv/data/QA_navsim_ctr_multistyle_103k.json"

# 筛选超参 (可调整以控制样本量在3-6k)
DIVERSITY_THRESHOLD=0.2
N_ROLLOUT=16
GROUP_SIZE=32

cd /mnt/data/ccy/EasyR1/debug/aug_adas

# 运行完整流程 (跳过navsim_data准备步骤，假设数据已准备好)
# Prefill 模式：
# - 只合并文件名包含 'prefill' 的 scorer CSV
# - 默认忽略 *_bak.csv，避免重复合并
# - group_size_mode=from_csv：按每个 token 的实际 rollout 数做置信度检验（prefill 推荐）
python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    --input_json_path "$INPUT_JSON" \
    --source_dataset_path "$SOURCE_DATASET" \
    --output_dataset_name "$OUTPUT_NAME" \
    --diversity_threshold $DIVERSITY_THRESHOLD \
    --n_rollout $N_ROLLOUT \
    --group_size $GROUP_SIZE \
    --csv_mode no_prefill \
    --group_size_mode from_csv \
    --target_min 3000 \
    --target_max 6000 \
    --skip_navsim_data

# 如果需要运行完整流程(包括navsim_data准备)，去掉 --skip_navsim_data 参数
