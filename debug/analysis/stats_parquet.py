#!/usr/bin/env python3
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm
import os

# ============ 配置 ============
exp_name = "4B_navsim_ps01_120kv3_pn103kv3_selfnorm_stage2"
# out_root = f"/mnt/data/ccy/EasyR1/debug/analysis/{exp_name}"
out_root = f"/mnt/data/ccy/VLA_train/parallel_infer/output/{exp_name}"
parquet_path = os.path.join(out_root, "generations_full.parquet")  # 你提取后的文件
tmp_csv_path = os.path.join(out_root, "tmp_filtered.csv")    # 中间文件
output_csv_path = os.path.join(out_root, "generations_full.csv")  # 输出
chunk_size = 500_000  # 每批处理的行数，可根据内存调整
# ==============================

# Step 1: 流式过滤 parsed_ok == true
dataset = ds.dataset(parquet_path, format="parquet")

with open(tmp_csv_path, "w") as f_out:
    header_written = False
    for batch in dataset.to_batches():
        df = batch.to_pandas()

        # 只保留 parsed_ok == True
        # df = df[df["parsed_ok"] == True]

        # 只保留必要列
        if "pdms_scaled" in df.columns and "pdms" in df.columns:
            df = df[["token", "pdms", "pdms_scaled"]]
        else:
            df = df[["token", "score"]]
            df["pdms"] = df["score"]  # 如果没有 pdms_scaled，就用 score 填充 pdms 列，保持兼容
            df["pdms_scaled"] = df["score"]  # 同样填充 pdms_scaled 列

        if not header_written:
            df.to_csv(f_out, index=False, header=True)
            header_written = True
        else:
            df.to_csv(f_out, index=False, header=False)

print(f"✅ 已提取并过滤写入: {tmp_csv_path}")

# Step 2: 一次性 groupby 得到精确统计
# （如果文件依然很大，也可以改成 dask/polars，但 pandas 对几百万行没问题）

df_all = pd.read_csv(tmp_csv_path)

grouped = df_all.groupby("token").agg(
    group_size=("token", "size"),
    pdms_mean=("pdms", "mean"),
    pdms_std=("pdms", "std"),
    pdms_range=("pdms", lambda x: x.max() - x.min()),
    pdms_scaled_mean=("pdms_scaled", "mean"),
    pdms_scaled_std=("pdms_scaled", "std"),
    pdms_scaled_range=("pdms_scaled", lambda x: x.max() - x.min())
).reset_index()

# Step 3: 写出结果
grouped.to_csv(output_csv_path, index=False)
print(f"📊 统计结果已写出: {output_csv_path}")

# 额外打印整体分布
print("📈 Group-level stats summary:")
print(grouped.describe())
