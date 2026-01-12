import warnings
import re
import json
import math
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def decode_indices_to_trajectory(indices: list[int], codebook: list[tuple[float, float, float]]) -> list[list[float]]:
    """
    action token 索引序列解码为绝对坐标轨迹，返回 [[x, y, yaw], ...]
    """
    trajectory = []
    current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

    for index in indices:
        if index >= len(codebook):
            continue

        local_delta_x, local_delta_y, delta_theta = codebook[index]

        theta = current_pose[2]
        world_delta_x = local_delta_x * math.cos(theta) - local_delta_y * math.sin(theta)
        world_delta_y = local_delta_x * math.sin(theta) + local_delta_y * math.cos(theta)

        new_x = current_pose[0] + world_delta_x
        new_y = current_pose[1] + world_delta_y
        new_theta = normalize_angle(current_pose[2] + delta_theta)

        current_pose = np.array([new_x, new_y, new_theta])
        trajectory.append([float(new_x), float(new_y), float(new_theta)])

    return trajectory


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi] 范围内。"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def parse_action_tokens(text: str, codebook: list[tuple[float, float, float]]) -> list[list[float]]:
    """
    从模型输出中提取 Action Tokens，返回 [[x, y, yaw], ...]
    """
    try:
        action_indices_str = re.findall(r"<action_(\d+)>", text)
        if not action_indices_str:
            return None

        action_indices = [int(idx) for idx in action_indices_str]
        decoded_trajectory = decode_indices_to_trajectory(action_indices, codebook)
        return action_indices, decoded_trajectory[:8] if len(decoded_trajectory) > 8 else decoded_trajectory
    except Exception as e:
        logger.warning(f"An unexpected exception occurred during action token parsing: {e}")
        return None


def parse_text_waypoint(output_text):
    """
    解析模型输出文本，提取未来轨迹
    """
    result = []

    if not output_text or not isinstance(output_text, str):
        return result

    # Step 1. 提取 trajectory
    traj_str = output_text
    if isinstance(traj_str, str):
        matches = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", traj_str)
        if matches:
            result = [list(map(float, m)) for m in matches]

    return result
    
def parse_trajectory_string_after_tag(text: str, tag='"future_trajectory"'):
    """
    鲁棒解析 tag字段后的 [x, y, yaw] 数据。
    不依赖 <answer> 标签，仅依赖字段名顺序和方括号结构。
    """
    try:

        parts = text.split(tag)
        
        if len(parts) < 2:
            return None
        
        content_after_key = parts[-1]
        list_match = re.search(r'\[(.*?)\]', content_after_key, re.DOTALL)
        
        if not list_match:
            return None
            
        target_content = list_match.group(1) # 获取 [...] 内部的字符串
        coord_pattern = r'\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)'
        matches_3d = re.findall(coord_pattern, target_content)

        if matches_3d and len(matches_3d) >= 8:
            points = [(float(x), float(y), float(yaw)) for x, y, yaw in matches_3d[:8]]
            return points
        
        return None 

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during yaw trajectory parsing: {e}")
        return None

def parse_text_waypoint_dict(output_text):
    """
    解析模型输出文本，提取未来轨迹、解释内容等字段。
    """
    result = {
        "data": None,            # 解析后的完整JSON
        "poses": [],
        "explanation": "",
        "parsed_ok": False
    }

    if not output_text or not isinstance(output_text, str):
        return result

    # Step 1. 清洗并尝试解析JSON
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", output_text.strip(), flags=re.IGNORECASE).strip()
        data = json.loads(cleaned)
        result["data"] = data
        result["parsed_ok"] = True
    except Exception as e:
        logger.warning(f"[parse_text_waypoint] JSON解析失败: {e}")
        return result

    # Step 2. 提取 explanation
    if isinstance(data.get("explanation"), str):
        result["explanation"] = data["explanation"].strip()

    # Step 3. 提取 trajectory
    traj_str = data.get("future_trajectory", "")
    if isinstance(traj_str, str):
        matches = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", traj_str)
        if matches:
            result["poses"] = [list(map(float, m)) for m in matches]

    return result

# stat_path = '/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/trajectory_stats_train.json' # 103k
stat_path = '/mnt/data/ccy/datasets/golden_navtrain/expand_tools/trajectory_stats_refine_v4.json'  # refine 103k v4

global means, stds
with open(stat_path, 'r', encoding='utf-8') as f: 
    data = json.load(f)
    means = np.array(data['mean'])
    stds = np.array(data['std'])


def denormalize(poses):
    result = np.array(poses) * stds + means
    return result.tolist()


if __name__ == '__main__':
    text = "{\"critical_objects\": {\"nearby_vehicle\": \"yes\", \"conflicting_pedestrian\": \"no\", \"cyclist\": \"no\", \"construction\": \"no\", \"traffic_element\": \"no\", \"weather_condition\": \"no\", \"road_hazard\": \"no\", \"emergency_vehicle\": \"no\", \"animal\": \"no\", \"special_vehicle\": \"no\", \"conflicting_vehicle\": \"yes\", \"door_opening_vehicle\": \"no\"}, \"explanation\": \"The expert trajectory was executed to navigate around a nearby vehicle and a conflicting vehicle ahead. The nearby vehicle on the left requires the ego-vehicle to maintain a safe distance, while the conflicting vehicle ahead necessitates a slight deviation to the right to avoid a potential collision. The expert smoothly adjusts the path to ensure safe passage.\", \"meta_behaviour\": {\"speed\": \"keep\", \"command\": \"lane_follow\"}, \"future_trajectory\": \"[PT, (+3.88, +0.11, +0.03), (+7.81, +0.23, +0.02), (+11.80, +0.27, 0.00), (+15.84, +0.19, -0.04), (+19.94, 0.00, -0.07), (+24.09, -0.28, -0.09), (+28.27, -0.66, -0.11), (+32.48, -1.12, -0.12)]\"}"
    print(parse_text_waypoint(text))