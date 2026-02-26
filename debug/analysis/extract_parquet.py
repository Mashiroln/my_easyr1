import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os

# ========= 配置 =========
out_root = "/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step"
input_path = "/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step/denorm_1111_policy_stats.jsonl"

os.makedirs(out_root, exist_ok=True)

extracted_parquet = os.path.join(out_root, "denorm_1111_policy_stats.parquet")
# summary_csv = "group_summary.csv"
chunk_size = 100000  # 每批多少条写入一次 parquet

# ========= 初始化 =========
buffer = []
group_stats = defaultdict(lambda: {
    "pdms": [],
    "pdms_scaled": [],
    "count": 0
})

# ========= 流式读取 =========
writer = None  # 全局 writer

with open(input_path, "r", encoding="utf-8") as f:
    for line_idx, line in tqdm(enumerate(f, 1), desc="Processing lines"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # if not data.get("parsed_ok"):
        #     continue

        token = data.get("token")
        poses = data.get("poses")
        pdms = data.get("pdms")
        pdms_scaled = data.get("pdms_scaled")

        if token is None or pdms is None or pdms_scaled is None:
            continue

        buffer.append({
            "token": token,
            "pdms": pdms,
            "pdms_scaled": pdms_scaled,
            "poses": json.dumps(poses)  # ✅ pose转字符串存更稳
        })

        stats = group_stats[token]
        stats["pdms"].append(pdms)
        stats["pdms_scaled"].append(pdms_scaled)
        stats["count"] += 1

        if len(buffer) >= chunk_size:
            df_chunk = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(extracted_parquet, table.schema)
            writer.write_table(table)
            buffer.clear()

# 写最后一批
if buffer:
    df_chunk = pd.DataFrame(buffer)
    table = pa.Table.from_pandas(df_chunk)
    if writer is None:
        writer = pq.ParquetWriter(extracted_parquet, table.schema)
    writer.write_table(table)
    buffer.clear()

if writer:
    writer.close()

print(f"✅ 抽取完成，已写入 {extracted_parquet}")
