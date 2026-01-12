# generate_comparison_plots.py

import os
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines
import cv2
import orjson
from collections import defaultdict

# --- 核心 Navsim 库导入 ---
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Camera
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.camera import _transform_pcs_to_images

# ======================= 用户配置区 =======================
# 1. 使用经过筛选的token列表
TOKEN_LIST_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/top_200_tokens_threshold.txt"
PREPROCESSED_DATA_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/preprocessed_data/"

# 2. 数据集路径
NAVSIM_LOGS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_logs/trainval/"
NAVSIM_SENSOR_BLOBS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_sensor_blobs/trainval/" 

# 3. 输出总目录
OUTPUT_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/v4"

# 4. 设置为 None 来处理所有token
TOKENS_TO_PROCESS = None
# =========================================================

def project_trajectory_to_image_official(trajectory_poses, camera):
    # (此函数与上一版本完全相同，保持不变)
    if trajectory_poses.shape[0] == 0: return np.array([])
    num_points = trajectory_poses.shape[0]
    points_3d = np.zeros((3, num_points)); points_3d[0, :] = trajectory_poses[:, 0]; points_3d[1, :] = trajectory_poses[:, 1]; points_3d[2, :] = 0 
    projected_points, fov_mask = _transform_pcs_to_images(
        lidar_pc=np.vstack([points_3d, np.zeros((3, num_points))]),
        sensor2lidar_rotation=camera.sensor2lidar_rotation, sensor2lidar_translation=camera.sensor2lidar_translation,
        intrinsic=camera.intrinsics, img_shape=None)
    return projected_points[fov_mask]

def select_trajectory_subset(trajs_with_pdms: list, num_each: int = 4) -> list:
    """
    【核心变更】: 根据PDMS分数，只筛选中等和低两个档次的轨迹子集。
    """
    if not trajs_with_pdms:
        return []

    # 按PDMS分数降序排序
    sorted_trajs = sorted(trajs_with_pdms, key=lambda x: x[1], reverse=True)
    
    total_count = len(sorted_trajs)
    if total_count < num_each * 2: # 如果轨迹太少，就返回全部
        return sorted_trajs

    # 选取分数最低的轨迹
    low_pdms = sorted_trajs[-num_each:]
    
    # 选取分数中位的轨迹
    mid_index = total_count // 2
    mid_start_index = max(0, mid_index - (num_each // 2))
    mid_pdms = sorted_trajs[mid_start_index : mid_start_index + num_each]
    
    # 合并中等和低分轨迹
    subset = mid_pdms + low_pdms
    
    return subset

def generate_plots_for_model(scene, token, gt_poses, model_trajs_with_pdms, model_name, output_dir):
    """为单个模型生成BEV和Camera两张可视化图"""
    current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    
    # --- 【核心变更】: 只筛选中等和低分轨迹进行可视化 ---
    subset_with_pdms = select_trajectory_subset(model_trajs_with_pdms)
    subset_poses = [t[0] for t in subset_with_pdms]

    # --- 1. 生成 BEV 图 ---
    fig_bev, ax_bev = plt.subplots(figsize=(10, 10))
    add_configured_bev_on_ax(ax_bev, scene.map_api, current_frame)
    
    # 绘制 GT 轨迹 (绿色，更粗以突出)
    ax_bev.plot(gt_poses[:, 1], gt_poses[:, 0], color="#59a14f", linewidth=3, zorder=11, ls='-')
    
    # 绘制筛选出的模型轨迹 (全部为红色)
    for poses in subset_poses:
        ax_bev.plot(poses[:, 1], poses[:, 0], color="#DE7061", linewidth=2, zorder=10, ls='-', alpha=0.8)

    configure_bev_ax(ax_bev); configure_ax(ax_bev)
    human_legend = mlines.Line2D([], [], color="#59a14f", ls='-', label='Ground Truth')
    agent_legend = mlines.Line2D([], [], color="#DE7061", ls='-', label=f'{model_name}')
    ax_bev.legend(handles=[human_legend, agent_legend], fontsize=12)
    ax_bev.set_title(f"BEV - {model_name}\nToken: {token}", fontsize=14)
    plt.tight_layout()
    # 【核心变更】: 简化文件名
    bev_filename = output_dir / f"{model_name}_bev.png"
    plt.savefig(bev_filename, dpi=150)
    plt.close(fig_bev)

    # --- 2. 生成 Camera 图 ---
    front_camera = current_frame.cameras.cam_f0
    if front_camera.image is not None:
        image_bgr = cv2.cvtColor(front_camera.image, cv2.COLOR_RGB2BGR)
        color_gt_bgr = (79, 161, 89)   # Green
        color_model_bgr = (97, 112, 222) # Red
        
        # 投影并绘制 GT 轨迹 (绿色，更粗)
        gt_points = project_trajectory_to_image_official(gt_poses, front_camera)
        if gt_points.shape[0] > 1:
            cv2.polylines(image_bgr, [gt_points.astype(np.int32)], False, color_gt_bgr, 5, cv2.LINE_AA)

        # 投影并绘制筛选出的模型轨迹 (全部为红色)
        for poses in subset_poses:
            points = project_trajectory_to_image_official(poses, front_camera)
            if points.shape[0] > 1:
                cv2.polylines(image_bgr, [points.astype(np.int32)], False, color_model_bgr, 3, cv2.LINE_AA, )
        
        # 【核心变更】: 简化文件名
        cam_filename = output_dir / f"{model_name}_cam.png"
        cv2.imwrite(str(cam_filename), image_bgr)
    else:
        print(f"Warning: Camera image not found for token {token}")

def main():
    # --- 1. 加载预处理数据 ---
    print("Loading preprocessed data...")
    with open(f"{PREPROCESSED_DATA_DIR}/gt_trajs.pkl", 'rb') as f: gt_trajs = pickle.load(f)
    with open(f"{PREPROCESSED_DATA_DIR}/curious_vla_trajs.pkl", 'rb') as f: curious_trajs = pickle.load(f)
    with open(f"{PREPROCESSED_DATA_DIR}/qwen_vla_trajs.pkl", 'rb') as f: qwen_trajs = pickle.load(f)
    with open(f"{PREPROCESSED_DATA_DIR}/token_to_log_map.json", 'r') as f: token_to_log_map = json.load(f)

    # --- 2. 加载待处理的token列表 ---
    with open(TOKEN_LIST_PATH, 'r') as f: tokens_to_visualize = [line.strip() for line in f if line.strip()]
    if TOKENS_TO_PROCESS: tokens_to_visualize = tokens_to_visualize[:TOKENS_TO_PROCESS]

    # --- 3. 创建 SensorConfig ---
    sensor_config = SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    # --- 4. 主循环 ---
    for token in tqdm(tokens_to_visualize, desc="Visualizing Tokens"):
        token_output_dir = Path(OUTPUT_DIR) / token
        token_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not all(token in d for d in [gt_trajs, curious_trajs, qwen_trajs]): continue
        gt_poses = gt_trajs[token][0][0]
        
        log_name = token_to_log_map.get(token)
        if log_name is None: continue
        
        try:
            scene_filter = SceneFilter(log_names=[log_name], tokens=[token], num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
            scene_loader = SceneLoader(
                data_path=Path(NAVSIM_LOGS_PATH), 
                original_sensor_path=Path(NAVSIM_SENSOR_BLOBS_PATH), 
                scene_filter=scene_filter, 
                sensor_config=sensor_config
            )
            scene = scene_loader.get_scene_from_token(token)
        except Exception as e: 
            print(f"\n[ERROR] Failed to load scene for token {token}. Error: {e}. Skipping.")
            continue
            
        generate_plots_for_model(scene, token, gt_poses, curious_trajs[token], "curious_vla", token_output_dir)
        generate_plots_for_model(scene, token, gt_poses, qwen_trajs[token], "qwen2_5_vl", token_output_dir)
        
    print("\nVisualization process completed.")

if __name__ == "__main__":
    main()