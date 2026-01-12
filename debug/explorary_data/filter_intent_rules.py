import orjson
import sys
import math
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from cubic_spline import calc_curvatures

# ==============================================================================

def _has_curvature_sign_change(curvatures: List[float]) -> bool:
    """
    检查曲率是否存在符号变化（S形转弯）
    (这个函数我们保留，因为它只依赖 'curvatures')
    """
    for i in range(1, len(curvatures)):
        if curvatures[i] * curvatures[i-1] < 0:
            return True
    return False

def calculate_heading_change_from_poses_list(positions: List[List[float]]) -> float:
    """
    从 poses 列表 (包含 [x, y, heading]) 中计算航向角变化（单位：度）。
    """
    if len(positions) < 2:
        return 0.0

    # 假设 positions[i][2] 是以 *弧度 (radians)* 为单位的航向角。
    # (如果它是角度, 请移除 math.degrees() 的调用)
    try:
        start_yaw = math.degrees(positions[0][2])
        end_yaw = math.degrees(positions[-1][2])
    except IndexError:
        return 0.0

    # 计算 180 度内的最小角度差
    heading_change = abs(end_yaw - start_yaw)
    if heading_change > 180:
        heading_change = 360 - heading_change
        
    return heading_change

# ==============================================================================
# (!!!) 核心方向标注逻辑 (已修改) (!!!)
# ==============================================================================

def annotate_direction_simple(
    positions: List[List[float]], 
    thresholds: Dict[str, float]
) -> str:
    """
    根据你的简化规则，仅标注方向。
    (!!!) 仅依赖 Poses [x, y, heading] 和 'calc_curvatures' (!!!)
    """
    
    mid_idx = len(positions) // 2
    if len(positions) < 2:
        return "other" 

    curvatures = calc_curvatures([pos[0] for pos in positions], [pos[1] for pos in positions])
    if not curvatures:
        return "other" 

    curvature = curvatures[mid_idx]
    
    heading_change = calculate_heading_change_from_poses_list(positions)
    curvature_sign_change = _has_curvature_sign_change(curvatures)

    # --- (!!!) 新的决策逻辑 (!!!) ---

    # 规则 2 (车道变换 -> 直行)
    # 近似: S形曲线 + 航向角变化小 = 车道变换
    if curvature_sign_change and heading_change < thresholds["min_heading_change"]:
        return "straight"

    # 规则 1 (直行)
    # 近似: 曲率小 或 航向角变化小 = 直行
    if (abs(curvature) < thresholds["min_curvature"] or 
        heading_change < thresholds["min_heading_change"]):
        return "straight"

    # 规则 3 (转向判断)
    # 必须不是 S 形曲线
    if (thresholds["min_curvature"] <= abs(curvature) <= thresholds["max_curvature"] and
        heading_change >= thresholds["min_heading_change"] and
        not curvature_sign_change):
        
        # (!!!) 关键检查点 (!!!)
        # 检查你的 calc_curvatures 约定：
        # 如果 curvature > 0 对应的是 *右转*, 
        # 请将下面这行改为:
        # direction = "right" if curvature > 0 else "left"
        direction = "left" if curvature > 0 else "right"
        
        return f"{direction} turn" 

    # 规则 4 (掉头判断 -> 其他)
    # 必须不是 S 形曲线
    if (abs(curvature) > thresholds["max_curvature"] and
        abs(heading_change - 180) <= thresholds["u_turn_heading_range"] and
        not curvature_sign_change):
        return "other" 
    
    # 规则 5: 其他所有情况 (例如 S形 + 大转弯)
    return "other"

# ==============================================================================
# 过滤器 (已修改)
# ==============================================================================

def filter_by_kinematic_consistency(sample_data: dict, thresholds: dict) -> bool:
    """
    检查 'intent' 字段是否与运动学计算出的 'direction' 一致
    (!!!) 最终版 (!!!)
    """
    
    intent = sample_data.get("intent", "unknown")
    if intent == "unknown":
        return True

    if "poses" not in sample_data:
        return False

    poses = sample_data.get("poses")
    
    if not poses or len(poses) < 2: 
        return False # 数据不足

    try:
        calculated_direction = annotate_direction_simple(
            positions=poses,
            thresholds=thresholds
        )
    except Exception as e:
        print(f"Skipping token {sample_data.get('token')} due to calculation error: {e}", file=sys.stderr)
        return False
        
    # (!!!) 最小修改：修复 'go straight' 的 key (!!!)
    intent_map = {
        "go straight": "straight",      # <--- 修复
        "turn left": "left turn",
        "turn right": "right turn",
        
        # (保留这些，以防你的数据里有)
        "lane change left": "straight",
        "lane change right": "straight",
        "left U-turn": "other",
        "right U-turn": "other"
    }
    
    if intent in intent_map:
        return calculated_direction == intent_map[intent]
    else:
        # 如果是 'stop' 等其他我们没定义的 intent，过滤掉
        return False

# ==============================================================================
# 主文件处理流程
# ==============================================================================

def main_filter_process(input_file, output_file, thresholds):
    """
    读取 input_file, 应用运动学过滤器, 写入 output_file
    """
    total_lines = 0
    kept_lines = 0
    error_lines = 0
    
    print(f"开始处理 {input_file}...")
    print("应用运动学一致性筛选 (FINAL - 仅 Poses[x,y,heading] 和 Curvature)...")
    
    try:
        with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
            pbar = tqdm(fin, desc=f"Filtering {input_file}", unit=" lines", unit_scale=True)

            for line in pbar:
                total_lines += 1
                try:
                    data = orjson.loads(line)
                    
                    if filter_by_kinematic_consistency(data, thresholds):
                        fout.write(orjson.dumps(data) + b"\n")
                        kept_lines += 1
                        
                except orjson.JSONDecodeError:
                    print(f"Skipping line {total_lines} due to JSON decode error.", file=sys.stderr)
                    error_lines += 1
                
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"发生意外错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- 筛选完成 ---")
    print(f"总共处理行数: {total_lines}")
    print(f"保留行数 (Kept): {kept_lines}")
    print(f"过滤行数 (Filtered): {total_lines - kept_lines - error_lines}")
    print(f"格式错误行数 (Errors): {error_lines}")
    print(f"成功写入到: {output_file}")

if __name__ == "__main__":
    
    # 1. 从 FlexVLAAnnotator 类中提取所有阈值
    THRESHOLDS = {
        "min_curvature": 0.015,
        "max_curvature": 0.25,
        "lateral_offset_threshold": 0.3,
        "lane_change_min_displacement": 1.75,
        "min_heading_change": 5,
        "u_turn_heading_range": 15,
    }

    # 2. 定义文件名
    # 确保这个文件有 'poses', 'rotations', 'intent'
    INPUT_JSONL = "/mnt/data/ccy/EasyR1/debug/explorary_data/11110423/augmented_recog_multi_intent.jsonl"
    OUTPUT_JSONL = "/mnt/data/ccy/EasyR1/debug/explorary_data/11110423/augmented_recog_multi_intent_filter_intent_rule_1.jsonl"

    # 3. 运行
    main_filter_process(INPUT_JSONL, OUTPUT_JSONL, THRESHOLDS)