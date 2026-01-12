# filter_tokens_threshold.py

import pandas as pd
from pathlib import Path

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# 1. 输入文件路径 (CSV格式，包含pdms_mean统计信息)
CURIOUS_VLA_CSV_PATH = "/mnt/data/ccy/EasyR1/debug/policy_stats/outputs/k=8/norm_cot_text_88step_npu_scene_stats.csv"
QWEN_VLA_CSV_PATH = "/mnt/data/ccy/EasyR1/debug/policy_stats/outputs/k=8/navie_cot_text_scene_stats.csv"

# 2. 输出文件路径
OUTPUT_FILE_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/top_200_tokens_threshold.txt"

# 3. 筛选参数
TOP_K = 200
# 【核心参数】Curious-VLA 的 PDMS 必须高于此分数才会被考虑
# 0.75 ~ 0.8 是一个通常代表"驾驶质量良好"的分数线
CURIOUS_PDMS_THRESHOLD = 0.8 

# ==============================================================================
# =============================== 脚本主体 (SCRIPT BODY) ===============================
# ==============================================================================

def filter_tokens_by_threshold(curious_csv_path, qwen_csv_path, output_path, top_k, threshold):
    print(f"Starting filtering process...")
    print(f"Criteria: Curious > Qwen AND Curious >= {threshold}")

    # --- 1. 加载CSV ---
    try:
        df_curious = pd.read_csv(curious_csv_path)
        df_qwen = pd.read_csv(qwen_csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV files: {e}")
        return

    # --- 2. 合并数据 ---
    # 仅保留两个模型都跑过的token
    merged_df = pd.merge(
        df_curious[['token', 'pdms_mean']], 
        df_qwen[['token', 'pdms_mean']], 
        on='token', 
        suffixes=('_curious', '_qwen'),
        how='inner'
    )
    print(f"Total common tokens: {len(merged_df)}")

    # --- 3. 应用筛选逻辑 ---
    
    # 条件 A: Curious 必须比 Qwen 好
    condition_better = merged_df['pdms_mean_curious'] > merged_df['pdms_mean_qwen']
    
    # 条件 B: Curious 的表现必须足够好 (高于阈值)
    # 这排除了两个模型都很差，或者Curious只是侥幸赢了一点点的低质量场景
    condition_high_quality = merged_df['pdms_mean_curious'] >= threshold
    
    filtered_df = merged_df[condition_better & condition_high_quality].copy()
    print(f"Tokens matching criteria: {len(filtered_df)}")

    if len(filtered_df) < top_k:
        print(f"[WARNING] Found only {len(filtered_df)} tokens matching criteria, which is less than requested {top_k}.")
    
    # --- 4. 计算差距并排序 ---
    # 我们依然按差距排序，差距越大越能体现模型优越性
    filtered_df['pdms_diff'] = filtered_df['pdms_mean_curious'] - filtered_df['pdms_mean_qwen']
    
    # 降序排列
    ranked_df = filtered_df.sort_values(by='pdms_diff', ascending=False)

    # --- 5. 选取Top-K ---
    top_tokens = ranked_df.head(top_k)['token'].tolist()

    # --- 6. 保存结果 ---
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for token in top_tokens:
            f.write(f"{token}\n")
            
    print(f"\nSuccessfully saved {len(top_tokens)} tokens to: {output_file}")
    
    # --- 7. 打印统计信息 ---
    print("\n--- Sample of Selected Tokens (Top 5) ---")
    print(ranked_df[['token', 'pdms_mean_curious', 'pdms_mean_qwen', 'pdms_diff']].head(5).to_string(index=False))
    
    print("\n--- Statistics of Selected Tokens ---")
    print(f"Avg Curious PDMS: {ranked_df.head(top_k)['pdms_mean_curious'].mean():.4f}")
    print(f"Avg Qwen PDMS:    {ranked_df.head(top_k)['pdms_mean_qwen'].mean():.4f}")
    print(f"Avg Diff:         {ranked_df.head(top_k)['pdms_diff'].mean():.4f}")

if __name__ == "__main__":
    filter_tokens_by_threshold(
        curious_csv_path=CURIOUS_VLA_CSV_PATH,
        qwen_csv_path=QWEN_VLA_CSV_PATH,
        output_path=OUTPUT_FILE_PATH,
        top_k=TOP_K,
        threshold=CURIOUS_PDMS_THRESHOLD
    )