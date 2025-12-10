import requests
import re
import json
import threading
import numpy as np
import random
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.reward_score.navsim.helper import parse_action_tokens, denormalize
from verl.utils.reward_score.navsim.pdms_logger import BatchJsonlLogger

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


time_str = datetime.now().strftime("%m%d%H%M")
log_file_path = f"/mnt/data/ccy/EasyR1/debug/analysis/generations_dynamic_{time_str}.jsonl"
log_lock = threading.Lock()

def log_to_jsonl(data: dict, file_path: str):
    with log_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

codebook_path = "/mnt/data/ccy/EasyR1/codebook/kmeans_codebook_v12.json"
with open(codebook_path, 'r') as f:
    json_data = json.load(f)
    codebook = np.array(json_data["codebook"])
    

'''
"answer": 
    {'gt': [[4.88, 0.0, 0.0], [9.51, 0.0, 0.0], [13.92, 0.0, 0.0], [18.12, 0.05, 0.0], 
            [22.12, 0.08, 0.0], [25.81, 0.07, 0.0], [29.05, 0.05, 0.0], [31.76, -0.08, -0.08]], 
    'token': 'bd2a4d57c04d50f1'}
'''

'''
PAYLOAD = {
    "token": "ffef12d9476e557b",
    "poses": [
        [2.6424, 0.2735, 0.2282], [5.3800, 1.0834, 0.3573],
        [8.1056, 2.2591, 0.4646], [10.9658, 3.7943, 0.5213],
        [13.8787, 5.5712, 0.5706], [16.8150, 7.4979, 0.5823],
        [19.8750, 9.5652, 0.5965], [23.1119, 11.7544, 0.5954]
    ],
    "verbose": False
}
'''

url_pool = ["http://0.0.0.0:8901/score"]
headers = {"Content-Type": "application/json"}
retries = 3
timeout = 120

EXPECTED_FIELDS = {
    "critical_objects": dict,
    "explanation": str,
    "meta_behaviour": dict,
    "future_trajectory": str,
}


def simulator_reward(token: str, poses: list[list[float]], verbose: bool):
    payload = {
        "token": token,
        "poses": poses,
        "verbose": verbose
    }
    # print(payload)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(random.choice(url_pool), data=json.dumps(payload), headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                pdms = data["pdms"]
                scaled_pdms = data["pdms_scaled"]
                return pdms, scaled_pdms
            else:
                logger.warning(f"[WARN] server error code: {resp.status_code}, try again {attempt}/{retries}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] request errer {e}, after tries {attempt}/{retries}")
            if attempt == retries: return 0.0, 0.0

def step_length_reward(indices, ground_truth):
    return int(len(indices) == len(ground_truth))



def format_reward(parsed):
    """
    计算格式奖励：
    - 成功解析JSON +0.4
    - 包含所有必需字段 +0.4
    - 字段类型正确 +0.1
    - 内容非空（至少 explanation 与 trajectory 有内容） +0.1
    """
    score = 0.0
    data = parsed.get("data")

    if not parsed["parsed_ok"] or not isinstance(data, dict):
        return score

    # 1. parse ok
    score += 0.4

    # 2. all present
    all_present = all(k in data for k in EXPECTED_FIELDS)
    if all_present:
        score += 0.4

    # 3. type ok
    type_ok = all(isinstance(data.get(k), t) for k, t in EXPECTED_FIELDS.items())
    if type_ok:
        score += 0.1

    # 4. non empty
    non_empty = (
        bool(data.get("explanation")) and
        bool(data.get("future_trajectory")) and
        len(data.get("critical_objects", {})) > 0
    )
    if non_empty:
        score += 0.1

    return score


def compute_score_fast(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for pdms reward function.")
    
    # 定义单个请求的处理函数
    def process_single_input(reward_input: Dict[str, Any]) -> Dict[str, float]:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        token = ground_truth["token"]

        _, poses = parse_action_tokens(response, codebook)
        pdms, scaled_pdms = simulator_reward(token, poses, False)
        # format_score = format_reward(parsed_dict)
        format_score = 1.0 if pdms >= 0.9 else 0.0 # 简化格式奖励为二值，因为当前是traj only，没有thinking，纯粹优化采样
        if pdms is None:
            pdms = 0.0
        if scaled_pdms is None:
            scaled_pdms = 0.0
            
        save_dict = {}
        save_dict['poses'] = poses
        save_dict["token"] = token
        save_dict["pdms"] = pdms
        save_dict["pdms_scaled"] = scaled_pdms
        save_dict["format_score"] = format_score
        save_dict["overall_score"] =(1 - format_weight) * pdms + format_weight * format_score
        log_to_jsonl(save_dict, log_file_path)
        # batch_logger.write(save_dict)
        
        return {
            "overall": (1 - format_weight) * pdms + format_weight * format_score,
            "format": format_score,
            "accuracy": pdms,
        }
    
    # 多线程并发处理，保持结果顺序与输入一致
    with ThreadPoolExecutor(max_workers=64) as executor:  # 可根据服务器承载能力调整
        # 提交所有任务并记录顺序
        future_to_index = {
            executor.submit(process_single_input, req): i 
            for i, req in enumerate(reward_inputs)
        }
        
        # 初始化结果列表，按原顺序填充
        scores = [None] * len(reward_inputs)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                scores[idx] = result
            except Exception as e:
                # 错误处理（根据需要调整，这里用默认值避免流程中断）
                scores[idx] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
                print(f"Error processing input {idx}: {str(e)}")
    
    return scores