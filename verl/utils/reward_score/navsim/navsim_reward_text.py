import requests
import re
import json
import threading
import numpy as np
import random
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.reward_score.navsim.helper import parse_text_waypoint, parse_text_waypoint_dict, denormalize, parse_trajectory_string_after_tag
from verl.utils.reward_score.navsim.pdms_logger import BatchJsonlLogger

import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REWARD_NAME = "navsim_span_grpo"
REWARD_TYPE = "batch"

time_str = datetime.now().strftime("%m%d%H%M")
log_file_path = f"/mnt/data/ccy/EasyR1/debug/analysis/generations_dynamic_{time_str}.jsonl"
log_lock = threading.Lock()

def log_to_jsonl(data: dict, file_path: str):
    with log_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
# batch_logger = BatchJsonlLogger(
#     file_path=log_file_path,
#     batch_size=100,
#     flush_interval=5
# )


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
url_pool_group = ["http://0.0.0.0:8901/score_group"]
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
    if len(poses) != 8:
        return 0.0, 0.0
    
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


# def navsim_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
#     token = extra_info["id"]
#     parsed_dict = parse_text_waypoint(solution_str)
#     pdms = simulator_reward(token, parsed_dict["poses"], False)
#     format_score = format_reward(parsed_dict)
#     if pdms is None:
#         pdms = 0.0
#     return pdms + 0.2 * format_score
    # return pdms

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
        
        # poses = parse_text_waypoint(response)
        poses = parse_trajectory_string_after_tag(response, "future_trajectory")
        poses = denormalize(poses)
        
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
            "accuracy": scaled_pdms,
            "pdms": pdms
        }
    
    # 多线程并发处理，保持结果顺序与输入一致
    with ThreadPoolExecutor(max_workers=96) as executor:
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
                scores[idx] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
                print(f"Error processing input {idx}: {str(e)}")
    
    return scores


def compute_score_group(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for pdms reward function.")

    # 1. 按 token 分组，同时保留原始索引
    token_groups = defaultdict(list)
    for i, req in enumerate(reward_inputs):
        try:
            token = req["ground_truth"]["token"]
            token_groups[token].append((i, req)) # 存储 (原始索引, 原始请求)
        except KeyError:
            print(f"[ERROR] Input {i} is missing 'ground_truth' or 'token'.")
            # 稍后我们将为这些错误索引填充
            pass

    # 2. 初始化结果列表
    scores = [None] * len(reward_inputs)
    
    # 3. 多线程并发处理 (按 token 批处理)
    with ThreadPoolExecutor(max_workers=192) as executor: # 192 也许可以调低，因为现在任务数 = token数
        
        # 跟踪 future 对应的所有原始索引，用于错误处理
        future_to_indices = {}
        
        for token, group in token_groups.items():
            future = executor.submit(process_token_batch, token, group, format_weight)
            # group 是 [(idx1, req1), (idx2, req2), ...]
            indices = [item[0] for item in group]
            future_to_indices[future] = indices

        # 4. 收集结果并按原始顺序填充
        for future in as_completed(future_to_indices):
            indices = future_to_indices[future]
            try:
                # future.result() 返回 [(idx1, res1), (idx2, res2), ...]
                results_with_indices = future.result()
                
                for idx, result in results_with_indices:
                    if scores[idx] is not None:
                        # 这是一个逻辑错误，不应该发生
                        print(f"[ERROR] Index {idx} is being overwritten. Check grouping logic.")
                    scores[idx] = result
                    
            except Exception as e:
                print(f"[ERROR] Error processing batch for indices {indices}: {str(e)}")
                # 如果整个批处理任务失败，为这个批次中的所有索引填充错误
                for idx in indices:
                    scores[idx] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 5. 检查并填充所有在分组阶段就失败的 None 值
    for i in range(len(scores)):
        if scores[i] is None:
            print(f"[WARN] Input {i} was not processed (likely missing token). Assigning default score.")
            scores[i] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            
    return scores

def simulator_reward_batch(token: str, all_poses: List[List[List[float]]], verbose: bool) -> List[Dict[str, float]]:
    """
    调用 /score_same 批量评估接口。
    
    Args:
        token: 相同的 token。
        all_poses: B*N*3 的 poses 列表 (B 是批量大小, N=8, 3=xyz)。
        verbose: 是否详细输出。

    Returns:
        一个字典列表，每个字典包含 "pdms" 和 "pdms_scaled"。
        如果请求失败，返回 B 个默认的 0 分字典。
    """
    if not all_poses:
        return []

    batch_size = len(all_poses)
    
    payload = {
        "token": token,
        "poses": all_poses, # 格式为 B*N*3
        "verbose": verbose
    }
    
    default_error_result = [{"pdms": 0.0, "pdms_scaled": 0.0}] * batch_size

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(random.choice(url_pool_group), data=json.dumps(payload), headers=headers, timeout=timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                # 关键假设：我们假设 /score_same 返回一个结果列表
                # 例如: [{"pdms": 0.9, "pdms_scaled": 0.8}, {"pdms": 0.7, "pdms_scaled": 0.6}, ...]
                # 并且返回的列表顺序与输入的 all_poses 顺序一致。
                results = data
                
                if isinstance(results, list) and len(results) == batch_size:
                    processed_results = [
                        {
                            "pdms": r.get("pdms", 0.0), 
                            "pdms_scaled": r.get("pdms_scaled", 0.0)
                        } 
                        for r in results
                    ]
                    return processed_results
                else:
                    print(f"[WARN] server response format error for token {token}. Expected list of size {batch_size}, got {results}")
                    return default_error_result

            else:
                print(f"[WARN] server error code: {resp.status_code}, try again {attempt}/{retries} for token {token}")
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] request error {e}, after tries {attempt}/{retries} for token {token}")
    
    # 所有重试失败
    return default_error_result

def process_token_batch(
    token: str, 
    group: List[Tuple[int, Dict[str, Any]]], 
    format_weight: float
) -> List[Tuple[int, Dict[str, float]]]:
    """
    处理共享同一个 token 的所有 reward_input。
    
    Args:
        token: 共享的 token。
        group: 一个元组列表, (原始索引, reward_input 字典)。
        format_weight: 格式权重。

    Returns:
        一个元组列表, (原始索引, 包含分数的字典)。
    """
    
    batch_poses_to_send = []
    indices_to_send = []
    results_for_invalid = []
    
    # 用于在获取分数后进行日志记录和计算
    parsed_data_map = {} 

    # 1. 预处理：解析 poses 并分离有效和无效的
    for original_index, reward_input in group:
        response = reward_input["response"]
        poses = parse_text_waypoint(response)
        poses = denormalize(poses)
        
        parsed_data_map[original_index] = {"poses": poses, "reward_input": reward_input}
        
        # 原始的 simulator_reward 会对 len != 8 的返回 0.0
        # 我们在这里模拟这个行为
        if len(poses) != 8:
            pdms = 0.0
            scaled_pdms = 0.0
            format_score = 0.0 # 因为 pdms < 0.9
            overall_score = 0.0 # (1-fw)*0.0 + fw*0.0
            
            # 记录日志 (即使是 0 分)
            save_dict = {
                'poses': poses,
                "token": token,
                "pdms": pdms,
                "pdms_scaled": scaled_pdms,
                "format_score": format_score,
                "overall_score": overall_score
            }
            log_to_jsonl(save_dict, log_file_path)
            
            # 添加到无效结果列表
            results_for_invalid.append((
                original_index,
                {"overall": overall_score, "format": format_score, "accuracy": pdms}
            ))
        else:
            # 这是一个有效的 pose，添加到待发送的批量中
            batch_poses_to_send.append(poses)
            indices_to_send.append(original_index)

    # 2. 批量请求：仅处理有效的 poses
    final_results = list(results_for_invalid) # 从无效结果开始
    
    if batch_poses_to_send:
        # 调用新的批量模拟器
        score_results = simulator_reward_batch(token, batch_poses_to_send, False)
        
        # 3. 处理批量返回的结果
        for i in range(len(indices_to_send)):
            original_index = indices_to_send[i]
            score_data = score_results[i] # 获取对应的分数
            
            pdms = score_data.get("pdms", 0.0)
            scaled_pdms = score_data.get("pdms_scaled", 0.0)
            
            # 从之前存储的 map 中获取 poses
            poses = parsed_data_map[original_index]["poses"]
            
            # 计算最终分数 (逻辑与原始代码一致)
            format_score = 1.0 if pdms >= 0.9 else 0.0
            overall_score = (1 - format_weight) * pdms + format_weight * format_score
            
            # 记录日志
            save_dict = {
                'poses': poses,
                "token": token,
                "pdms": pdms,
                "pdms_scaled": scaled_pdms,
                "format_score": format_score,
                "overall_score": overall_score
            }
            log_to_jsonl(save_dict, log_file_path)
            
            # 添加到最终结果列表
            final_results.append((
                original_index,
                {"overall": overall_score, "format": format_score, "accuracy": pdms}
            ))

    return final_results