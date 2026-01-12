# filter_tokens.py

import pandas as pd
from pathlib import Path

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# 1. 定义输入文件路径
CURIOUS_VLA_CSV_PATH = "/mnt/data/ccy/EasyR1/debug/policy_stats/outputs/k=8/norm_cot_text_88step_npu_scene_stats.csv"
QWEN_VLA_CSV_PATH = "/mnt/data/ccy/EasyR1/debug/policy_stats/outputs/k=8/navie_cot_text_scene_stats.csv"

# 2. 定义输出文件路径和筛选数量
OUTPUT_FILE_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/top_100_tokens.txt"
TOP_K = 100

# ==============================================================================
# =============================== 脚本主体 (SCRIPT BODY) ===============================
# ==============================================================================

def filter_and_rank_tokens(curious_csv_path: str, qwen_csv_path: str, output_path: str, top_k: int):
    """
    加载两个模型的评估结果，筛选出Curious-VLA表现更优的场景，
    并按优势大小排序，最后保存Top-K的token列表。
    """
    
    # --- 1. 检查输入文件是否存在 ---
    curious_csv = Path(curious_csv_path)
    qwen_csv = Path(qwen_csv_path)
    
    if not curious_csv.exists():
        print(f"[ERROR] Curious-VLA CSV file not found at: {curious_csv}")
        return
    if not qwen_csv.exists():
        print(f"[ERROR] QwenVL2.5 CSV file not found at: {qwen_csv}")
        return
        
    print("Input CSV files found. Starting the filtering process...")

    # --- 2. 加载CSV文件 ---
    try:
        df_curious = pd.read_csv(curious_csv)
        df_qwen = pd.read_csv(qwen_csv)
        print(f"Successfully loaded {len(df_curious)} records for Curious-VLA.")
        print(f"Successfully loaded {len(df_qwen)} records for QwenVL2.5.")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV files. Error: {e}")
        return

    # --- 3. 合并两个DataFrame ---
    # 我们使用 'inner' join 来找到两个文件中共有的token
    # 并为每个模型的列添加后缀以区分
    merged_df = pd.merge(
        df_curious, 
        df_qwen, 
        on='token', 
        suffixes=('_curious', '_qwen'),
        how='inner'
    )
    print(f"Found {len(merged_df)} common tokens between the two models.")
    
    if merged_df.empty:
        print("[ERROR] No common tokens found. Please check your CSV files.")
        return

    # --- 4. 应用筛选条件 ---
    # 条件: Curious-VLA的pdms_mean要大于QwenVL2.5的pdms_mean
    filtered_df = merged_df[merged_df['pdms_mean_curious'] > merged_df['pdms_mean_qwen']].copy()
    print(f"Found {len(filtered_df)} tokens where Curious-VLA performed better.")

    if len(filtered_df) < top_k:
        print(f"[Warning] Found fewer than {top_k} tokens that meet the criteria. Proceeding with {len(filtered_df)} tokens.")
        top_k = len(filtered_df)
        
    if filtered_df.empty:
        print("[INFO] No tokens met the filtering criteria. The output file will be empty.")
        # 创建一个空文件
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            pass
        return

    # --- 5. 计算PDMS分数差值并排序 ---
    # 计算差值作为排序指标
    filtered_df['pdms_diff'] = filtered_df['pdms_mean_curious'] - filtered_df['pdms_mean_qwen']
    
    # 按差值从大到小排序
    ranked_df = filtered_df.sort_values(by='pdms_diff', ascending=False)

    # --- 6. 选取Top-K的token ---
    top_tokens = ranked_df.head(top_k)['token'].tolist()
    
    # --- 7. 保存结果到文件 ---
    output_file = Path(output_path)
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            for token in top_tokens:
                f.write(f"{token}\n")
        print(f"\nSuccessfully saved the Top {len(top_tokens)} tokens to: {output_file}")
        
        # 打印一些示例信息以供验证
        print("\n--- Top 5 Tokens and their PDMS scores ---")
        print(ranked_df[['token', 'pdms_mean_curious', 'pdms_mean_qwen', 'pdms_diff']].head(5).to_string(index=False))

    except Exception as e:
        print(f"[ERROR] Failed to write the output file. Error: {e}")


if __name__ == "__main__":
    filter_and_rank_tokens(
        curious_csv_path=CURIOUS_VLA_CSV_PATH,
        qwen_csv_path=QWEN_VLA_CSV_PATH,
        output_path=OUTPUT_FILE_PATH,
        top_k=TOP_K
    )