import pandas as pd
import numpy as np
import io
import os


exp_name = "norm_cot_text_130step"
out_root = f"/mnt/data/ccy/EasyR1/debug/analysis/{exp_name}"
csv_path = os.path.join(out_root, "denorm_1111_policy_stats.csv")
csv_out_path = os.path.join(out_root, "group_stats_filtered_hard3k.csv")
# read csv data from path
csv_data = open(csv_path, "r").read()
# 1. 加载数据
df = pd.read_csv(io.StringIO(csv_data))
print(f"原始数据行数: {len(df)}")
print("--- 原始数据 ---")
print(df)

# 创建一个副本以进行操作
df_filtered = df.copy()

# 2. 排序，基于pdms_mean降序
df_filtered = df_filtered.sort_values(by='pdms_mean', ascending=False).copy()

# 3. 选择前1000个最难样本（pdms_mean>0的样本中，最低的1000个）
df_filtered = df_filtered[df_filtered['pdms_mean'] > 0].tail(3000).copy()

# 4. 保存过滤后的数据到新的CSV文件
df_filtered.to_csv(csv_out_path, index=False)
print(f"\n过滤后的数据已保存到 {csv_out_path}，行数: {len(df_filtered)}")
print("--- 过滤后数据 ---")
print(df_filtered)