# generate_visualizations_fixed.py

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines
import cv2
import orjson # 高效处理大JSONL文件
import json
from collections import defaultdict

# --- 核心 Navsim 库导入 ---
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Camera
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.camera import _transform_pcs_to_images

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# --- 输入文件 ---
TOKEN_LIST_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/selected_tokens.txt"
CURIOUS_VLA_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_88step_npu/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k_88step_policy_stats.jsonl"
QWEN_VLA_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/navie_cot_text/generations_policy_stats.jsonl"
GROUND_TRUTH_JSONL = "/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl"

# --- Navsim 数据集路径 ---
NAVSIM_LOGS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_logs/trainval/"
NAVSIM_SENSOR_BLOBS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_sensor_blobs/trainval/" 

# --- 输出配置 ---
OUTPUT_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/v2"
# 【测试模式】设置为5，跑通后再设置为None来处理所有token
TOKENS_TO_PROCESS = None

# ==============================================================================
# =============================== 核心函数 (CORE FUNCTIONS) ===============================
# ==============================================================================

def load_trajectories_from_jsonl(jsonl_path: str, use_orjson: bool = False) -> defaultdict:
    # (此函数与您的版本相同，保持不变)
    trajectories = defaultdict(list)
    print(f"Loading trajectories from {Path(jsonl_path).name}...")
    try:
        with open(jsonl_path, 'rb') as f:
            for line in tqdm(f, desc="Reading lines"):
                record = orjson.loads(line) if use_orjson else json.loads(line)
                token, poses = record.get("token"), record.get("poses")
                if token and poses:
                    trajectories[token].append(np.array(poses, dtype=np.float32))
    except FileNotFoundError:
        print(f"[ERROR] Trajectory file not found: {jsonl_path}")
        return None
    print(f"Loaded {len(trajectories)} unique tokens with a total of {sum(len(v) for v in trajectories.values())} trajectories.")
    return trajectories

def project_trajectory_to_image_official(trajectory_poses: np.ndarray, camera: Camera) -> np.ndarray:
    # (此函数与上一版本相同，保持不变)
    if trajectory_poses.shape[0] == 0: return np.array([])
    num_points = trajectory_poses.shape[0]
    points_3d = np.zeros((3, num_points)); points_3d[0, :] = trajectory_poses[:, 0]; points_3d[1, :] = trajectory_poses[:, 1]; points_3d[2, :] = 0 
    projected_points, fov_mask = _transform_pcs_to_images(
        lidar_pc=np.vstack([points_3d, np.zeros((3, num_points))]),
        sensor2lidar_rotation=camera.sensor2lidar_rotation, sensor2lidar_translation=camera.sensor2lidar_translation,
        intrinsic=camera.intrinsics, img_shape=None)
    return projected_points[fov_mask]

def build_token_to_log_map(navsim_log_path: Path, sensor_blobs_path: Path) -> dict:
    """
    【性能优化】: 一次性扫描所有log文件，建立 token -> log_name 的映射。
    """
    print("Building token to log_name map for fast lookups... (this may take a minute)")
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
        except Exception:
            continue
    print(f"Token map built. Found {len(token_map)} total unique tokens.")
    return token_map

def generate_plots_for_model(scene, token, gt_poses, model_poses_list, model_name, output_dir):
    # (此函数与您的版本相同，保持不变)
    current_frame_idx = scene.scene_metadata.num_history_frames - 1; current_frame = scene.frames[current_frame_idx]
    fig_bev, ax_bev = plt.subplots(figsize=(10, 10)); add_configured_bev_on_ax(ax_bev, scene.map_api, current_frame)
    ax_bev.plot(gt_poses[:, 1], gt_poses[:, 0], color="#59a14f", linewidth=2.5, zorder=11, ls='-')
    for model_poses in model_poses_list: ax_bev.plot(model_poses[:, 1], model_poses[:, 0], color="#DE7061", linewidth=1.5, zorder=10, ls='-', alpha=0.7)
    configure_bev_ax(ax_bev); configure_ax(ax_bev)
    human_legend = mlines.Line2D([], [], color="#59a14f", ls='-', label='Ground Truth (Human)')
    agent_legend = mlines.Line2D([], [], color="#DE7061", ls='-', label=f'{model_name} Prediction (k={len(model_poses_list)})')
    ax_bev.legend(handles=[human_legend, agent_legend], fontsize=12); ax_bev.set_title(f"BEV - {model_name}\nToken: {token}", fontsize=14)
    plt.tight_layout(); bev_filename = output_dir / f"{model_name}_bev.png"; plt.savefig(bev_filename, dpi=150); plt.close(fig_bev)
    
    front_camera = current_frame.cameras.cam_f0
    if front_camera.image is not None:
        image_bgr = cv2.cvtColor(front_camera.image, cv2.COLOR_RGB2BGR); color_gt = (79, 161, 89); color_model = (97, 112, 222)
        gt_points_2d = project_trajectory_to_image_official(gt_poses, front_camera)
        if gt_points_2d.shape[0] > 1: cv2.polylines(image_bgr, [gt_points_2d.astype(np.int32)], isClosed=False, color=color_gt, thickness=4, lineType=cv2.LINE_AA)
        for model_poses in model_poses_list:
            model_points_2d = project_trajectory_to_image_official(model_poses, front_camera)
            if model_points_2d.shape[0] > 1: cv2.polylines(image_bgr, [model_points_2d.astype(np.int32)], isClosed=False, color=color_model, thickness=3, lineType=cv2.LINE_AA)
        cam_filename = output_dir / f"{model_name}_cam.png"; cv2.imwrite(str(cam_filename), image_bgr)
    else: print(f"Warning: Camera image not found for token {token}")

# ==============================================================================
# ================================== 主流程 ===================================
# ==============================================================================

def main():
    # --- 1. 加载所有轨迹数据到内存 ---
    gt_trajs = load_trajectories_from_jsonl(GROUND_TRUTH_JSONL)
    curious_trajs = load_trajectories_from_jsonl(CURIOUS_VLA_JSONL)
    qwen_trajs = load_trajectories_from_jsonl(QWEN_VLA_JSONL, use_orjson=True)
    if not all([gt_trajs, curious_trajs, qwen_trajs]): print("Aborting due to missing trajectory files."); return

    # --- 2. 加载待处理的token列表 ---
    try:
        with open(TOKEN_LIST_PATH, 'r') as f: tokens_to_visualize = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tokens_to_visualize)} tokens to visualize from {TOKEN_LIST_PATH}")
    except FileNotFoundError: print(f"[ERROR] Token list file not found: {TOKEN_LIST_PATH}"); return

    if TOKENS_TO_PROCESS: tokens_to_visualize = tokens_to_visualize[:TOKENS_TO_PROCESS]; print(f"--- RUNNING IN TEST MODE: Processing first {len(tokens_to_visualize)} tokens ---")

    # --- 3. 【性能优化】预先构建 token -> log_name 映射 ---
    token_to_log_map = build_token_to_log_map(Path(NAVSIM_LOGS_PATH), Path(NAVSIM_SENSOR_BLOBS_PATH))

    # --- 4. 【核心修正】创建只请求前置摄像头的 SensorConfig ---
    sensor_config_cam_f0_only = SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    # --- 5. 循环处理每个token ---
    for token in tqdm(tokens_to_visualize, desc="Visualizing Tokens"):
        token_output_dir = Path(OUTPUT_DIR) / token; token_output_dir.mkdir(parents=True, exist_ok=True)
        if token not in gt_trajs or token not in curious_trajs or token not in qwen_trajs: print(f"\nWarning: Skipping token {token} (missing from trajectory files)."); continue
        
        gt_poses, curious_poses_list, qwen_poses_list = gt_trajs[token][0], curious_trajs[token], qwen_trajs[token]
        
        log_name = token_to_log_map.get(token) # 【性能优化】快速查找
        if log_name is None: print(f"\nWarning: Could not find log file for token {token}. Skipping."); continue
        
        try:
            scene_filter = SceneFilter(log_names=[log_name], tokens=[token], num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
            scene_loader = SceneLoader(
                data_path=Path(NAVSIM_LOGS_PATH), 
                original_sensor_path=Path(NAVSIM_SENSOR_BLOBS_PATH), 
                scene_filter=scene_filter, 
                sensor_config=sensor_config_cam_f0_only # 【核心修正】使用自定义的config
            )
            scene = scene_loader.get_scene_from_token(token)
        except Exception as e: print(f"Error loading scene for token {token}: {e}. Skipping."); continue
            
        generate_plots_for_model(scene, token, gt_poses, curious_poses_list, "curious_vla", token_output_dir)
        generate_plots_for_model(scene, token, gt_poses, qwen_poses_list, "qwen2_5_vl", token_output_dir)
        
    print("\nVisualization process completed.")

if __name__ == "__main__":
    main()