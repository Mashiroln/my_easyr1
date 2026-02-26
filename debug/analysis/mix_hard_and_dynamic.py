# token集合运算
# Result = (A / (B' / B)) ∪ C
# A: original dynamic samples
# B': current log samples 当前记录下的sample
# B: current dynamic samples 当前记录中的多样性sample
# C: hard samples
# (A / (B' / B)): 从原始动态样本中去除当前记录下的【非多样性】样本


# if A is None
# Result = B ∪ C
# 即从当前多样性样本中加入困难样本

# 其中输出也是CSV文件，包含token列，供后续数据集过滤使用
# A和C同一token对应和行数值可能不同，保留A中的行数值

import pandas as pd
import numpy as np
import io
import os

exp_name = "norm_cot_text_130step"
out_root = f"/mnt/data/ccy/EasyR1/debug/analysis/{exp_name}"
# csv_path_origin_dynamic = "/mnt/data/ccy/EasyR1/debug/analysis/normalized_cot_text/group_stats_cot_norm_88step.csv"
# csv_path_current_log = os.path.join(out_root, "group_stats_cot_norm_88step.csv")

csv_path_origin_dynamic = None
csv_path_current_log = None

csv_path_current_dynamic = os.path.join(out_root, "group_stats_filtered_0.1.csv")
csv_path_hard = os.path.join(out_root, "group_stats_filtered_hard3k.csv")

csv_out_path = os.path.join(out_root, "group_stats_mixed_dynamic_hard_130step.csv")

# 1. 加载数据
if csv_path_origin_dynamic is not None and csv_path_current_log is not None:
    #  (A / (B' / B)):
    df_origin_dynamic = pd.read_csv(io.StringIO(open(csv_path_origin_dynamic, "r").read()))
    df_current_log = pd.read_csv(io.StringIO(open(csv_path_current_log, "r").read()))
    df_current_dynamic = pd.read_csv(io.StringIO(open(csv_path_current_dynamic, "r").read()))
    df_hard = pd.read_csv(io.StringIO(open(csv_path_hard, "r").read()))

    # 2. 提取token集合
    set_origin_dynamic = set(df_origin_dynamic['token'].tolist())
    set_current_dynamic = set(df_current_dynamic['token'].tolist())
    set_current_log = set(df_current_log['token'].tolist())
    set_hard = set(df_hard['token'].tolist())

    # 3. 计算结果集合
    set_non_diversity = set_current_log - set_current_dynamic
    set_result = (set_origin_dynamic - set_non_diversity).union(set_hard)
    print(f"原始动态样本数量: {len(set_origin_dynamic)}")
    print(f"当前记录样本数量: {len(set_current_log)}")
    print(f"当前动态样本数量: {len(set_current_dynamic)}")
    print(f"困难样本数量: {len(set_hard)}")

    print(f"结果样本数量: {len(set_result)}")

    # 4. 构建结果DataFrame
    # 如果token在origin_dynamic中，则保留origin_dynamic中的行；如果不在，则从hard中取行
    df_result = df_origin_dynamic[df_origin_dynamic['token'].isin(set_result)].copy()
    df_result = df_result._append(df_hard[df_hard['token'].isin(set_result) & ~df_hard['token'].isin(df_result['token'])])

    # 5. 保存结果到CSV文件

    df_result.to_csv(csv_out_path, index=False)
    print(f"结果数据已保存到 {csv_out_path}，行数: {len(df_result)}")
    print("--- 结果数据 ---")
    print(df_result)
else:
    df_origin_dynamic = None
    df_current_log = None
    # Result = B ∪ C
    df_current_dynamic = pd.read_csv(io.StringIO(open(csv_path_current_dynamic, "r").read()))
    df_hard = pd.read_csv(io.StringIO(open(csv_path_hard, "r").read()))
    # 2. 提取token集合
    set_current_dynamic = set(df_current_dynamic['token'].tolist())
    set_hard = set(df_hard['token'].tolist())
    
    # 3. 计算结果集合
    set_result = set_current_dynamic.union(set_hard)
    print(f"当前动态样本数量: {len(set_current_dynamic)}")
    print(f"困难样本数量: {len(set_hard)}")
    print(f"结果样本数量: {len(set_result)}")
    # 4. 构建结果DataFrame
    # 如果token在current_dynamic中，则保留current_dynamic中的行；如果不在，则从hard中取行
    df_result = df_current_dynamic[df_current_dynamic['token'].isin(set_result)].copy()
    df_result = df_result._append(df_hard[df_hard['token'].isin(set_result) & ~df_hard['token'].isin(df_result['token'])])
    
    # 5. 保存结果到CSV文件
    df_result.to_csv(csv_out_path, index=False)
    print(f"结果数据已保存到 {csv_out_path}，行数: {len(df_result)}")
    print("--- 结果数据 ---")
    print(df_result)
