# archive_selected_trajectories.py

import pickle
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# 1. 指定您手动挑选的、最终确认的token列表文件
#    假设您将最终挑选的token保存在这个文件中，每行一个。
SELECTED_TOKENS_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/selected_tokens.txt"

# 2. 指定预处理数据的目录
PREPROCESSED_DATA_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/preprocessed_data/"

# 3. 指定最终归档输出的目录
ARCHIVE_OUTPUT_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/archived_data/"

# ==============================================================================
# =============================== 核心函数 (CORE FUNCTIONS) ===============================
# ==============================================================================

def select_trajectory_subset(trajs_with_pdms: list, num_each: int = 4) -> list:
    """
    【保持一致】: 与可视化脚本中完全相同的筛选逻辑，
    以确保我们提取的是同一批中低分数的轨迹。
    """
    if not trajs_with_pdms:
        return []

    sorted_trajs = sorted(trajs_with_pdms, key=lambda x: x[1], reverse=True)
    total_count = len(sorted_trajs)
    if total_count < num_each * 2: return sorted_trajs
    
    low_pdms = sorted_trajs[-num_each:]
    mid_index = total_count // 2
    mid_start_index = max(0, mid_index - (num_each // 2))
    mid_pdms = sorted_trajs[mid_start_index : mid_start_index + num_each]
    
    subset = mid_pdms + low_pdms
    return subset

def archive_trajectories(selected_tokens_path: str, preprocessed_dir: str, archive_dir: str):
    """
    读取最终挑选的token列表，提取对应的轨迹和PDMS，并归档为JSONL文件。
    """
    preprocessed_path = Path(preprocessed_dir)
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)

    # --- 1. 加载所有预处理数据 ---
    print("Loading preprocessed data...")
    try:
        with open(preprocessed_path / "curious_vla_trajs.pkl", 'rb') as f:
            curious_trajs_all = pickle.load(f)
        with open(preprocessed_path / "qwen_vla_trajs.pkl", 'rb') as f:
            qwen_trajs_all = pickle.load(f)
        with open(preprocessed_path / "gt_trajs.pkl", 'rb') as f:
            gt_trajs_all = pickle.load(f)
        
        with open(selected_tokens_path, 'r') as f:
            selected_tokens = {line.strip() for line in f if line.strip()}
            
    except FileNotFoundError as e:
        print(f"[ERROR] Required file not found: {e}. Aborting.")
        return
        
    print(f"Loaded {len(selected_tokens)} tokens to archive from {selected_tokens_path}")

    # --- 2. 准备输出文件 ---
    # 使用 'w' 模式来确保每次运行时都创建新文件
    output_files = {
        'curious_vla': open(archive_path / 'curious_vla_selected.jsonl', 'w'),
        'qwen_vla': open(archive_path / 'qwen_vla_selected.jsonl', 'w'),
        'ground_truth': open(archive_path / 'ground_truth_selected.jsonl', 'w')
    }

    # --- 3. 遍历挑选的token并提取数据 ---
    archived_count = 0
    for token in tqdm(selected_tokens, desc="Archiving selected tokens"):
        if not all(token in d for d in [curious_trajs_all, qwen_trajs_all, gt_trajs_all]):
            print(f"\nWarning: Skipping token '{token}' as it's missing from one of the preprocessed files.")
            continue
        
        # --- a. 提取并写入 Ground Truth ---
        gt_poses, gt_pdms = gt_trajs_all[token][0]
        gt_record = {"token": token, "poses": gt_poses.tolist(), "pdms": gt_pdms}
        output_files['ground_truth'].write(json.dumps(gt_record) + '\n')
        
        # --- b. 提取并写入 Curious-VLA 子集 ---
        curious_subset = select_trajectory_subset(curious_trajs_all[token])
        for poses, pdms in curious_subset:
            record = {"token": token, "poses": poses.tolist(), "pdms": pdms}
            output_files['curious_vla'].write(json.dumps(record) + '\n')
            
        # --- c. 提取并写入 QwenVL2.5 子集 ---
        qwen_subset = select_trajectory_subset(qwen_trajs_all[token])
        for poses, pdms in qwen_subset:
            record = {"token": token, "poses": poses.tolist(), "pdms": pdms}
            output_files['qwen_vla'].write(json.dumps(record) + '\n')

        archived_count += 1
            
    # --- 4. 关闭所有文件 ---
    for f in output_files.values():
        f.close()
        
    print(f"\nArchiving complete. Processed {archived_count} tokens.")
    print("Generated files:")
    for key, f in output_files.items():
        print(f"  - {f.name}")

if __name__ == "__main__":
    # 确保您已经创建了 'selected_tokens.txt' 文件，并填入了您挑选的token
    if not Path(SELECTED_TOKENS_PATH).exists():
        print(f"[ERROR] The file '{SELECTED_TOKENS_PATH}' was not found.")
        print("Please create this file and add the tokens you want to archive, one token per line.")
    else:
        archive_trajectories(
            selected_tokens_path=SELECTED_TOKENS_PATH,
            preprocessed_dir=PREPROCESSED_DATA_DIR,
            archive_dir=ARCHIVE_OUTPUT_DIR
        )