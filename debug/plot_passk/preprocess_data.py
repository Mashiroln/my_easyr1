# preprocess_data.py

import orjson
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle

# --- 核心 Navsim 库导入 ---
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

# ======================= 用户配置区 =======================
# --- 输入文件 ---
CURIOUS_VLA_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_88step_npu/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k_88step_policy_stats.jsonl"
QWEN_VLA_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/navie_cot_text/generations_policy_stats.jsonl"
GROUND_TRUTH_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl"
NAVSIM_LOGS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_logs/trainval/"
NAVSIM_SENSOR_BLOBS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_sensor_blobs/trainval/" 

# --- 输出目录 ---
PREPROCESS_OUTPUT_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/preprocessed_data/"
# =========================================================

def preprocess_trajectories(jsonl_path: str, output_path: str, use_orjson: bool = False):
    """加载轨迹和PDMS分数，保存为pickle文件。"""
    trajectories = defaultdict(list)
    print(f"Processing trajectories from {Path(jsonl_path).name}...")
    try:
        with open(jsonl_path, 'rb') as f:
            for line in tqdm(f, desc="Reading lines"):
                record = orjson.loads(line) if use_orjson else json.loads(line)
                token, poses, pdms = record.get("token"), record.get("poses"), record.get("pdms", -1.0)
                if token and poses:
                    # 保存 (轨迹, 分数) 元组
                    trajectories[token].append((np.array(poses, dtype=np.float32), pdms))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {jsonl_path}"); return
    
    with open(output_path, 'wb') as f:
        pickle.dump(dict(trajectories), f)
    print(f"Saved preprocessed trajectories to {output_path}")

def build_and_save_token_map(navsim_log_path: Path, sensor_blobs_path: Path, output_path: str):
    """一次性扫描所有log文件，建立 token -> log_name 的映射并保存。"""
    print("Building token to log_name map... (this may take over an hour, but only needs to be run once)")
    token_map = {}
    log_files = sorted(list(navsim_log_path.glob("*.pkl")))
    scene_filter_config = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    for log_file in tqdm(log_files, desc="Pre-scanning logs"):
        log_name = log_file.stem
        try:
            scene_filter_config.log_names = [log_name]
            temp_loader = SceneLoader(data_path=navsim_log_path, original_sensor_path=sensor_blobs_path, scene_filter=scene_filter_config, sensor_config=SensorConfig.build_no_sensors())
            for token in temp_loader.tokens:
                token_map[token] = log_name
        except Exception as e:
            print(f"Skipping log {log_name} due to error: {e}")
            continue
            
    with open(output_path, 'w') as f:
        json.dump(token_map, f)
    print(f"Token map built and saved to {output_path}")

def main():
    Path(PREPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. 预处理轨迹文件
    preprocess_trajectories(CURIOUS_VLA_JSONL, f"{PREPROCESS_OUTPUT_DIR}/curious_vla_trajs.pkl")
    preprocess_trajectories(QWEN_VLA_JSONL, f"{PREPROCESS_OUTPUT_DIR}/qwen_vla_trajs.pkl", use_orjson=True)
    preprocess_trajectories(GROUND_TRUTH_JSONL, f"{PREPROCESS_OUTPUT_DIR}/gt_trajs.pkl")
    
    # 2. 构建token->log映射
    build_and_save_token_map(Path(NAVSIM_LOGS_PATH), Path(NAVSIM_SENSOR_BLOBS_PATH), f"{PREPROCESS_OUTPUT_DIR}/token_to_log_map.json")

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()