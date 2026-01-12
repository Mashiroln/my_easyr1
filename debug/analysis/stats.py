import json
import numpy as np
from collections import defaultdict
from itertools import combinations
import pandas as pd
import os

# ====================
# 配置
# ====================
log_path = "generations_dynamic.jsonl"   # ✅ 日志路径
output_csv = "group_similarity_stats_dynamic_75step.csv"

# ====================
# 工具函数
# ====================

def pose_distance(p1, p2):
    """
    计算两个 8x3 pose 序列之间的 L2 平均距离。
    p1, p2: list of list
    """
    arr1 = np.array(p1)
    arr2 = np.array(p2)
    return np.mean(np.linalg.norm(arr1 - arr2, axis=1))

# ====================
# 1. 读取与过滤
# ====================
groups = defaultdict(list)

n_samples = 10000  # 只分析最后 10000 条记录
def tail_lines(file_path, n=n_samples, chunk_size=8192):
    """仅从文件末尾取出最后 n 行"""
    lines = []
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        buffer = b""
        pos = file_size
        while pos > 0 and len(lines) < n + 1:  # +1 是为了防止文件最后没换行的情况
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            buffer = f.read(read_size) + buffer
            lines = buffer.split(b"\n")
        return [l.decode("utf-8", errors="ignore") for l in lines[-n:] if l.strip()]

# 读取最后2000行
for line in tail_lines(log_path, n=n_samples):
    try:
        data = json.loads(line)
        if data.get("parsed_ok") is True:
            token = data.get("token")
            poses = data.get("poses")
            pdms = data.get("pdms", None)
            pdms_scaled = data.get("pdms_scaled", None)
            if poses is not None and len(poses) == 8:
                groups[token].append({
                    "poses": poses,
                    "pdms": pdms,
                    "pdms_scaled": pdms_scaled
                })
    except json.JSONDecodeError:
        continue
print(f"✅ 有效 group 数: {len(groups)}")

# ====================
# 2. 计算组内相似性
# ====================
records = []

for token, items in groups.items():
    n = len(items)
    if n <= 1:
        continue  # 单个样本不计算相似性

    # poses pairwise distance
    pose_distances = []
    for (i, a), (j, b) in combinations(enumerate(items), 2):
        d = pose_distance(a["poses"], b["poses"])
        pose_distances.append(d)

    # pdms stats
    pdms_values = [it["pdms"] for it in items if it["pdms"] is not None]
    pdms_mean = np.mean(pdms_values) if pdms_values else np.nan
    pdms_std = np.std(pdms_values) if pdms_values else np.nan
    pdms_range = (np.max(pdms_values) - np.min(pdms_values)) if len(pdms_values) > 1 else 0
    
    # pdms_scaled stats
    pdms_scaled_values = [it["pdms_scaled"] for it in items if it.get("pdms_scaled") is not None]
    pdms_scaled_mean = np.mean(pdms_scaled_values) if pdms_scaled_values else np.nan
    pdms_scaled_std = np.std(pdms_scaled_values) if pdms_scaled_values else np.nan
    pdms_scaled_range = (np.max(pdms_scaled_values) - np.min(pdms_scaled_values)) if len(pdms_scaled_values) > 1 else 0
    
    records.append({
        "token": token,
        "group_size": n,
        "pose_dist_mean": np.mean(pose_distances),
        "pose_dist_std": np.std(pose_distances),
        "pdms_mean": pdms_mean,
        "pdms_std": pdms_std,
        "pdms_range": pdms_range,
        "pdms_scaled_mean": pdms_scaled_mean,
        "pdms_scaled_std": pdms_scaled_std,
        "pdms_scaled_range": pdms_scaled_range,
    })

# ====================
# 3. 输出结果
# ====================
df = pd.DataFrame(records)
df = df.sort_values(by="pose_dist_mean", ascending=True)

print(df.head(20))  # 看前20个多样性最低的 group

df.to_csv(output_csv, index=False)
print(f"✅ 已输出统计结果至 {output_csv}")
