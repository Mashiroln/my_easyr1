import json
import numpy as np
import os
import orjson
from pathlib import Path

global means, stds
with open('/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/trajectory_stats_train.json', 'r', encoding='utf-8') as f: 
    data = json.load(f)
    means = np.array(data['mean'])
    stds = np.array(data['std'])


def denormalize(poses):
    result = np.array(poses) * stds + means
    return result.tolist()

def process_jsonl(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("wb") as fout:  # orjson 输出必须写二进制

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
                poses = obj.get("poses")
                if isinstance(poses, list):
                    obj["poses"] = denormalize(poses)
                fout.write(orjson.dumps(obj))
                fout.write(b"\n")
            except Exception as e:
                # 打印错误但不中断
                print(f"[WARN] Line {line_num} skipped: {e}")

if __name__ == "__main__":
    process_jsonl("/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step/1111_policy_stats.jsonl", "/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step/denorm_1111_policy_stats.jsonl")

