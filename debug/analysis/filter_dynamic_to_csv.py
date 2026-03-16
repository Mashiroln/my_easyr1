import pandas as pd
import numpy as np
import io
import os

diversity_threshold = 0.1 # 阈值越小，要求混合程度越高

# csv_path = "full_navsim/group_stats.csv"
exp_name = "4B_navsim_ps01_120kv3_pn103kv3_selfnorm_stage2"
# out_root = f"/mnt/data/ccy/EasyR1/debug/analysis/{exp_name}"
out_root = f"/mnt/data/ccy/VLA_train/parallel_infer/output/{exp_name}"
csv_path = os.path.join(out_root, "generations_full.csv")
csv_out_path = os.path.join(out_root, f"group_stats_filtered_{diversity_threshold}.csv")
# read csv data from path
csv_data = open(csv_path, "r").read()
n_rollout = 8
# 1. 加载数据
df = pd.read_csv(io.StringIO(csv_data))
print(f"原始数据行数: {len(df)}")
print("--- 原始数据 ---")
print(df)

# 创建一个副本以进行操作
df_filtered = df.copy()

# 2. 初步过滤
std_threshold = 0.01
df_filtered = df_filtered[df_filtered['pdms_std'] > std_threshold].copy()
print(f"\n步骤1: std > {std_threshold} 过滤后，剩余行数: {len(df_filtered)}")

# 3. 估算比例 p
# 避免除以0的情况
df_filtered = df_filtered[df_filtered['pdms_range'] > 1e-6].copy()
# 核心假设: p_est ≈ mean / range
df_filtered['p_est'] = df_filtered['pdms_mean'] / df_filtered['pdms_range']

# 4. 计算多样性指标
diversity_metric = df_filtered['p_est']**n_rollout + (1 - df_filtered['p_est'])**n_rollout
df_filtered['diversity_metric'] = diversity_metric

# 5. 应用多样性过滤
# diversity_threshold = 0.5 # 阈值越小，要求混合程度越高
df_filtered = df_filtered[df_filtered['diversity_metric'] < diversity_threshold].copy()
print(f"步骤2: 多样性过滤 (metric < {diversity_threshold}) 后，剩余行数: {len(df_filtered)}")

# 6. 执行置信度检验
if not df_filtered.empty:
    group_size = 16
    # 估算高价值点的数量k
    k_est = (df_filtered['p_est'] * group_size).round()
    
    # 根据k和range预测标准差
    # variance = (k * (n-k) / (n * (n-1))) * range^2
    var_numerator = k_est * (group_size - k_est)
    var_denominator = group_size * (group_size - 1)
    predicted_std = np.sqrt(var_numerator / var_denominator) * df_filtered['pdms_range']
    
    df_filtered['predicted_std'] = predicted_std
    
    # 计算相对误差
    confidence_error = np.abs(df_filtered['predicted_std'] - df_filtered['pdms_std']) / df_filtered['pdms_std']
    df_filtered['confidence_error'] = confidence_error
    
    # 只保留误差在10%以内的结果
    confidence_threshold = 0.1
    final_df = df_filtered[df_filtered['confidence_error'] < confidence_threshold].copy()
    print(f"步骤3: 置信度检验 (error < {confidence_threshold}) 后，剩余行数: {len(final_df)}")
else:
    final_df = df_filtered # 如果前面已经过滤完了，直接赋值

# 7. 输出结果
print("\n--- 筛选过程中的详细数据 ---")
# 为了方便观察，我们打印出带有中间计算列的DataFrame
print(df_filtered.round(4))

print("\n--- 最终筛选结果 (输出为CSV格式) ---")
# 输出不包含我们中间计算的列
output_columns = df.columns.tolist()
final_csv_output = final_df[output_columns].to_csv(index=False)
# save to file
with open(csv_out_path, "w") as f_out:
    f_out.write(final_csv_output)
print("Saved filtered results to:", csv_out_path)

# Write token list txt (for data.token_filter_file)
txt_out_path = csv_out_path.replace(".csv", ".txt")
with open(txt_out_path, "w") as f_out:
    for token in final_df["token"]:
        f_out.write(f"{token}\n")
print(f"Token list: {txt_out_path}  ({len(final_df)} entries)")