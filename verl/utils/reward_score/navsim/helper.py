import warnings
import re
import json
import math
import os
import logging
import importlib
import numpy as np
from typing import Callable, List, Optional, Tuple

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

def parse_cpr_text_waypoint(output_text: str) -> List[List[float]]:
    """Same parsing logic, but strictly limited to the planning trajectory."""
    result: List[List[float]] = []

    if not output_text or not isinstance(output_text, str):
        return result

    # 1. 定位 planning trajectory 的起始位置
    keyword = "Here is the planning trajectory"
    idx = output_text.find(keyword)

    # 2. 如果找到了关键词，截取该关键词之后的所有内容
    if idx != -1:
        target_text = output_text[idx:]
    else:
        # 容错处理：如果 LLM 幻觉没有生成这句话，为了防止误提取 coarsed trajectory，直接返回空列表跳过此样本
        return result

    # 3. 仅对截取后的 target_text 进行坐标提取
    matches = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", target_text)
    if matches:
        result = [list(map(float, m)) for m in matches]

    return result

# ------------------------------
# Stats paths (configurable)
# ------------------------------
# Training scripts can override these via env vars:
# - NAVSIM_STAT_PATH
# - NAVSIM_STAT_PATH_SYN
_DEFAULT_STAT_PATH = "/mnt/data/ccy/EasyR1/verl/utils/reward_score/navsim/trajectory_stats_train.json"

stat_path = os.environ.get("NAVSIM_STAT_PATH", _DEFAULT_STAT_PATH)
stat_path_syn_env = os.environ.get("NAVSIM_STAT_PATH_SYN", "").strip()
stat_path_syn = stat_path_syn_env if stat_path_syn_env else None


def _load_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["mean"]), np.array(data["std"])


global means, stds, means_syn, stds_syn
means, stds = _load_stats(stat_path)
if stat_path_syn:
    means_syn, stds_syn = _load_stats(stat_path_syn)
else:
    means_syn, stds_syn = means, stds


_SYN_TOKEN_RE = re.compile(r"-00\d$")


def _use_syn_stats(token: Optional[str]) -> bool:
    return bool(token) and bool(_SYN_TOKEN_RE.search(str(token)))


def denormalize(poses: List[List[float]], token: Optional[str] = None) -> List[List[float]]:
    use_syn = _use_syn_stats(token)
    # print(f"Denormalizing poses for token={token} using {'synthetic' if use_syn else 'train'} stats.")
    m, s = (means_syn, stds_syn) if use_syn else (means, stds)
    result = np.array(poses) * s + m
    return result.tolist()


# ------------------------------
# Trajectory parsing (configurable)
# ------------------------------
# Training scripts can override these via env vars:
# - NAVSIM_TRAJ_PARSER_FUNC: import path "package.module:function"


def _load_callable_from_spec(spec: str) -> Optional[Callable[[str], List[List[float]]]]:
    """Load a callable from a spec like 'a.b.c:func'. Returns None if invalid."""

    s = (spec or "").strip()
    if not s or ":" not in s:
        return None
    mod_name, func_name = s.split(":", 1)
    mod_name, func_name = mod_name.strip(), func_name.strip()
    if not mod_name or not func_name:
        return None
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name, None)
        if callable(fn):
            return fn  # type: ignore[return-value]
    except Exception:
        return None
    return None


def get_trajectory_parser(parser: Optional[str] = None) -> Callable[[str], List[List[float]]]:
    """Return a callable that parses output_text -> poses.

    This indirection lets the launcher script choose parsing behavior.
    """

    # Only supported mechanism: direct callable import path.
    fn_spec = (os.environ.get("NAVSIM_TRAJ_PARSER_FUNC", "") or "").strip()
    fn = _load_callable_from_spec(fn_spec)
    if fn is None:
        raise ValueError(
            "NAVSIM_TRAJ_PARSER_FUNC is required (expected 'package.module:function'). "
            f"Got: {fn_spec!r}"
        )
    return fn


if __name__ == '__main__':
    text = "{\"critical_objects\": {\"nearby_vehicle\": \"yes\", \"conflicting_pedestrian\": \"no\", \"cyclist\": \"no\", \"construction\": \"no\", \"traffic_element\": \"no\", \"weather_condition\": \"no\", \"road_hazard\": \"no\", \"emergency_vehicle\": \"no\", \"animal\": \"no\", \"special_vehicle\": \"no\", \"conflicting_vehicle\": \"yes\", \"door_opening_vehicle\": \"no\"}, \"explanation\": \"The expert trajectory was executed to navigate around a nearby vehicle and a conflicting vehicle ahead. The nearby vehicle on the left requires the ego-vehicle to maintain a safe distance, while the conflicting vehicle ahead necessitates a slight deviation to the right to avoid a potential collision. The expert smoothly adjusts the path to ensure safe passage.\", \"meta_behaviour\": {\"speed\": \"keep\", \"command\": \"lane_follow\"}, \"future_trajectory\": \"[PT, (+3.88, +0.11, +0.03), (+7.81, +0.23, +0.02), (+11.80, +0.27, 0.00), (+15.84, +0.19, -0.04), (+19.94, 0.00, -0.07), (+24.09, -0.28, -0.09), (+28.27, -0.66, -0.11), (+32.48, -1.12, -0.12)]\"}"
    print(parse_text_waypoint(text))