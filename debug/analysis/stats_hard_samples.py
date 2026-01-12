import pandas as pd
import numpy as np
import io
import os


hard_a_file = "/mnt/data/ccy/EasyR1/debug/analysis/normalized_cot_text/group_stats_filtered_hard1k.csv"
hard_b_file = "/mnt/data/ccy/EasyR1/debug/analysis/normalized_traj_text/group_stats_filtered_hard1k.csv"

df_hard_a = pd.read_csv(io.StringIO(open(hard_a_file, "r").read()))
df_hard_b = pd.read_csv(io.StringIO(open(hard_b_file, "r").read()))

set_hard_a = set(df_hard_a['token'].tolist())
set_hard_b = set(df_hard_b['token'].tolist())

set_hard_common = set_hard_a.intersection(set_hard_b)
print(f"困难样本A数量: {len(set_hard_a)}")
print(f"困难样本B数量: {len(set_hard_b)}")
print(f"共同困难样本数量: {len(set_hard_common)}")