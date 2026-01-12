# filter_tokens_advanced_v2.py

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# 1. 定义预处理数据和输出文件的路径
PREPROCESSED_DATA_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/preprocessed_data/"
OUTPUT_FILE_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/top_100_tokens_advanced.txt"
TOP_K = 100

# ==============================================================================
# =============================== 核心函数 (CORE FUNCTIONS) ===============================
# ==============================================================================

def calculate_apd(poses_list: list) -> float:
    """
    计算一个token的k条轨迹之间的平均成对L2距离 (Average Pairwise Distance)。
    """
    if len(poses_list) < 2:
        return 0.0
    total_distance, pair_count = 0.0, 0
    for i in range(len(poses_list)):
        for j in range(i + 1, len(poses_list)):
            traj1, traj2 = poses_list[i], poses_list[j]
            min_len = min(len(traj1), len(traj2))
            diff = traj1[:min_len, :2] - traj2[:min_len, :2]
            distances_over_time = np.linalg.norm(diff, axis=1)
            total_distance += np.mean(distances_over_time)
            pair_count += 1
    return total_distance / pair_count if pair_count > 0 else 0.0

def filter_and_rank_advanced_v2(preprocessed_dir: str, output_path: str, top_k: int):
    """
    执行V2版高级筛选：结合PDMS差异和对比多样性指数(CDI)进行双重排序。
    """
    preprocessed_path = Path(preprocessed_dir)

    # --- 1. 加载预处理好的数据 ---
    print("Loading preprocessed data...")
    try:
        with open(preprocessed_path / "curious_vla_trajs.pkl", 'rb') as f:
            curious_trajs = pickle.load(f)
        with open(preprocessed_path / "qwen_vla_trajs.pkl", 'rb') as f:
            qwen_trajs = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Preprocessed file not found: {e}. Please run the preprocessing script first.")
        return

    # --- 2. 计算每个token的指标 ---
    common_tokens = set(curious_trajs.keys()) & set(qwen_trajs.keys())
    print(f"Found {len(common_tokens)} common tokens. Calculating metrics...")
    
    analysis_data = []
    for token in tqdm(common_tokens, desc="Analyzing tokens"):
        curious_poses = [rec[0] for rec in curious_trajs[token]]
        curious_pdms_scores = [rec[1] for rec in curious_trajs[token] if rec[1] != -1.0]
        qwen_poses = [rec[0] for rec in qwen_trajs[token]]
        qwen_pdms_scores = [rec[1] for rec in qwen_trajs[token] if rec[1] != -1.0]
        
        if not curious_pdms_scores or not qwen_pdms_scores: continue

        pdms_mean_curious = np.mean(curious_pdms_scores)
        pdms_mean_qwen = np.mean(qwen_pdms_scores)
        
        apd_curious = calculate_apd(curious_poses)
        apd_qwen = calculate_apd(qwen_poses)

        # --- 计算V2版指标 ---
        pdms_diff = pdms_mean_curious - pdms_mean_qwen
        diversity_diff = apd_curious - apd_qwen
        comparative_diversity_index = diversity_diff * (1 - pdms_mean_qwen)

        analysis_data.append({
            "token": token,
            "pdms_diff": pdms_diff,
            "comparative_diversity_index": comparative_diversity_index,
            "pdms_mean_curious": pdms_mean_curious,
            "pdms_mean_qwen": pdms_mean_qwen,
            "apd_curious": apd_curious,
            "apd_qwen": apd_qwen,
        })
        
    # --- 3. 转换为DataFrame并进行筛选和排序 ---
    df = pd.DataFrame(analysis_data)
    filtered_df = df[df['pdms_diff'] > 0].copy()
    print(f"\nFound {len(filtered_df)} tokens where Curious-VLA performed better.")

    # V2版双重排序
    ranked_df = filtered_df.sort_values(
        by=['pdms_diff', 'comparative_diversity_index'],
        ascending=[False, False]
    )

    # --- 4. 选取Top-K并保存 ---
    top_tokens = ranked_df.head(top_k)['token'].tolist()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for token in top_tokens:
            f.write(f"{token}\n")
            
    print(f"\nSuccessfully saved the Top {len(top_tokens)} tokens (advanced V2 ranking) to: {output_file}")
    
    # --- 5. 打印最有价值的信息以供验证 ---
    print("\n--- Top 5 Tokens based on advanced V2 ranking ---")
    print("Primary metric: 'pdms_diff' (performance gap, bigger is better)")
    print("Secondary metric: 'comparative_diversity_index' (CDI, bigger is better for showcasing contrast)")
    # 设置pandas显示格式
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(ranked_df[[
        'token', 
        'pdms_diff', 
        'comparative_diversity_index',
        'apd_curious',
        'apd_qwen',
        'pdms_mean_qwen'
    ]].head(5).to_string(index=False))

if __name__ == "__main__":
    filter_and_rank_advanced_v2(
        preprocessed_dir=PREPROCESSED_DATA_DIR,
        output_path=OUTPUT_FILE_PATH,
        top_k=TOP_K
    )