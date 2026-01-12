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
import random

# --- 核心 Navsim 库导入 ---
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Camera
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.camera import _transform_pcs_to_images
from matplotlib import font_manager as fm

font_path = "/mnt/data/ccy/EasyR1/COMIC.TTF"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Comic Sans MS'

# ======================= 用户配置区 =======================
# 1. 使用经过高级筛选的token列表
TOKEN_LIST_PATH = "/mnt/data/ccy/EasyR1/debug/plot_passk/selected_tokens.txt"
PREPROCESSED_DATA_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/preprocessed_data/"

# 2. 数据集路径
NAVSIM_LOGS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_logs/trainval/"
NAVSIM_SENSOR_BLOBS_PATH = "/mnt/data/ccy/datasets/navsim/trainval_sensor_blobs/trainval/" 

# 3. 输出总目录
OUTPUT_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/v3"

os.mkdir(OUTPUT_DIR) if not os.path.exists(OUTPUT_DIR) else None

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

def smart_bad_traj(trajs_with_pdms: list, num_each: int = 4) -> list:
    """
    智能筛选 "差" 轨迹 (中+低)。
    - 如果PDMS全为0 (例如模型完全失败)，则随机抽选8条返回，避免排序失去意义。
    - 如果存在PDMS不为0，则按 select_trajectory_subset (中+低) 函数返回。
    """
    if not trajs_with_pdms:
        return []

    # 检查PDMS是否全为0
    all_zero = all(pdms == 0.0 for _, pdms in trajs_with_pdms)
    
    if all_zero:
        # PDMS全为0，排序无意义，改为随机采样
        num_to_sample = min(len(trajs_with_pdms), num_each * 2)
        return random.sample(trajs_with_pdms, num_to_sample)
    else:
        # 至少有一个PDMS不为0，使用标准的中+低筛选
        return select_trajectory_subset(trajs_with_pdms, num_each)

# ==============================================================================
# 3. 新函数: smart_good_traj (智能筛选好轨迹)
# ==============================================================================

def smart_good_traj(trajs_with_pdms: list, num_each: int = 4) -> list:
    """
    智能筛选 "好" 轨迹 (中+高)，并保证轨迹两两不相同。
    - 1. 首先对所有输入轨迹进行去重。
    - 2. 选取PDMS分数最高的 'num_each' 条。
    - 3. 选取PDMS分数中位的 'num_each' 条。
    - 4. 合并两者，确保最终列表中的轨迹是唯一的。
    """
    if not trajs_with_pdms:
        return []

    # --- 1. 强制去重 (满足“两两不相同”约束) ---
    # (如果多条相同轨迹有不同PDMS，保留分数最高的那个)
    unique_trajs_map = {}
    try:
        # 优先使用 .tobytes() 作为key
        for traj, pdms in trajs_with_pdms:
            key = traj.tobytes()
            if key not in unique_trajs_map or pdms > unique_trajs_map[key][1]:
                unique_trajs_map[key] = (traj, pdms)
    except AttributeError:
        # 降级方案
        print("Warning: Trajectory is not a numpy array, using str() for uniqueness.")
        for traj, pdms in trajs_with_pdms:
            key = str(traj)
            if key not in unique_trajs_map or pdms > unique_trajs_map[key][1]:
                unique_trajs_map[key] = (traj, pdms)
                
    deduped_list = list(unique_trajs_map.values())

    # --- 2. 按PDMS分数降序排序 ---
    sorted_trajs = sorted(deduped_list, key=lambda x: x[1], reverse=True)

    total_count = len(sorted_trajs)
    num_to_select = num_each * 2

    if total_count <= num_to_select: # 如果轨迹太少，就返回全部
        return sorted_trajs

    # --- 3. 选取切片 ---
    
    # 选取分数最高的轨迹 (高位)
    high_pdms = sorted_trajs[:num_each]
    
    # 选取分数中位的轨迹
    mid_index = total_count // 2
    mid_start_index = max(0, mid_index - (num_each // 2))
    mid_pdms = sorted_trajs[mid_start_index : mid_start_index + num_each]

    # --- 4. 合并 (确保高位轨迹的优先权) ---
    # (如果中位和高位切片重叠，字典会保留高位版本)
    subset_map = {}
    
    # 先添加中位
    for traj, pdms in mid_pdms:
        subset_map[traj.tobytes()] = (traj, pdms)
        
    # 再添加高位 (如果key已存在，高位会覆盖中位，确保我们保留的是高位)
    for traj, pdms in high_pdms:
        subset_map[traj.tobytes()] = (traj, pdms)

    return list(subset_map.values())

def generate_plots_for_model(scene, token, gt_poses, model_poses_list, model_name, output_dir):
    """
    [已更新] 结合了方案 1 (Jitter) 和 方案 2 (Alpha Blending) 的绘图函数。
    """
    
    # --- [方案 1: 重合检测与 Jitter] ---
    # print("pose list: ", model_poses_list)
    subset_with_pdms = smart_bad_traj(model_poses_list) if model_name == 'qwen2_5_vl' else smart_good_traj(model_poses_list)
    
    processed_poses_list = [np.copy(poses[0]) for poses in subset_with_pdms]
    
    
    
    # 2. 检查是否 "完全重合"
    is_collapsed = False
    if len(processed_poses_list) > 2: # 至少要有两条轨迹才能比较
        is_collapsed = True
        first_pose_set = processed_poses_list[0]
        # 检查后续所有轨迹是否与第一条完全相同
        for other_pose_set in processed_poses_list[1:]:
            if not np.array_equal(first_pose_set, other_pose_set):
                is_collapsed = False
                break

    # 3. 如果 "完全重合"，则应用 Jitter
    if is_collapsed:
        # print(f"Info: Detected collapsed trajectories for {model_name} on token {token}. Adding slight jitter for visualization.")
        
        # 定义噪声 (例如：15cm)
        # 噪声只加到 (x, y) 维度，即 [0] 和 [1]
        noise_vec_1 = np.zeros_like(processed_poses_list[0][0]) # [0.0, 0.0, 0.0, ...]
        noise_vec_1[0] = 0.05 + random.gauss(0, 0.025) # +15cm on X
        noise_vec_1[1] = 0.05 + random.gauss(0, 0.025)  # +15cm on Y

        noise_vec_2 = np.zeros_like(processed_poses_list[1][0])
        noise_vec_2[0] = -0.05 + random.gauss(0, 0.025)# -15cm on X
        noise_vec_2[1] = -0.05 + random.gauss(0, 0.025)# -15cm on Y

        # 将噪声应用到前两条轨迹上
        # (np.newaxis) 是为了让 (3,) 的噪声能和 (N, 3) 的轨迹相加
        processed_poses_list[0] = processed_poses_list[0] + noise_vec_1[np.newaxis, :]
        processed_poses_list[1] = processed_poses_list[1] + noise_vec_2[np.newaxis, :]
        
        # 其他 K-2 条轨迹保持不变 (它们将与 0 和 1 重叠)
    
    # --- [方案 2: 使用 Alpha Blending 进行绘图] ---
    # 接下来，我们使用 'processed_poses_list' (可能已被Jitter) 进行绘图
    
    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_frame_idx]

    # --- BEV 绘图 ---
    fig_bev, ax_bev = plt.subplots(figsize=(10, 10))
    add_configured_bev_on_ax(ax_bev, scene.map_api, current_frame)
    
    # 1. 绘制 GT (保持不变)
    ax_bev.plot(gt_poses[:, 1], gt_poses[:, 0], color="#59a14f", linewidth=5, zorder=11, ls='-')

    # 2. [方案 2] 绘制模型轨迹 (使用低 Alpha)
    for model_poses in processed_poses_list: # 使用处理过的列表
        ax_bev.plot(
            model_poses[:, 1], 
            model_poses[:, 0], 
            color="#DE7061", 
            linewidth=4.5,  # 可以稍微加粗
            zorder=10, 
            ls='-', 
            alpha=0.15      # [核心] 使用低透明度
        )
        
    configure_bev_ax(ax_bev); configure_ax(ax_bev)
    human_legend = mlines.Line2D([], [], color="#59a14f", ls='-', label='Human Ground Truth')
    model_name_str = "Qwen2.5-VL" if model_name == 'qwen2_5_vl' else "Curious-VLA"
    
    agent_legend = mlines.Line2D([], [], color="#DE7061", ls='-', label=f'{model_name_str} (k={len(processed_poses_list)})')
    ax_bev.legend(handles=[human_legend, agent_legend], fontsize=12)
    # ax_bev.set_title(f"BEV - {model_name}\nToken: {token}", fontsize=14)
    ax_bev.set_title("")
    plt.tight_layout(); bev_filename = output_dir / f"{model_name}_bev.png"; plt.savefig(bev_filename, dpi=150); plt.close(fig_bev)
    
    # --- Camera 绘图 ---
    front_camera = current_frame.cameras.cam_f0
    if front_camera.image is not None:
        
        # 1. 准备基础图像和颜色
        image_bgr = cv2.cvtColor(front_camera.image, cv2.COLOR_RGB2BGR)
        color_gt = (79, 161, 89)      # 绿色 (B,G,R)
        color_model = (97, 112, 222)  # 红色/蓝色 (B,G,R)

        # 2. 绘制 Ground Truth
        # 我们在一个 "最终图像" 上操作
        final_image = image_bgr.copy()
        gt_points_2d = project_trajectory_to_image_official(gt_poses, front_camera)
        if gt_points_2d.shape[0] > 1:
            cv2.polylines(final_image, [gt_points_2d.astype(np.int32)], isClosed=False, color=color_gt, thickness=8, lineType=cv2.LINE_AA)

        # --- [核心修复] ---
        # 3. 创建一个 "Alpha 热图"
        
        # 3.1: 创建一个 2D 的 float 数组来累加透明度
        alpha_map = np.zeros(final_image.shape[:2], dtype=np.float32)
        
        # [回答 "太淡了"]：在这里调整单条线的 Alpha (例如 0.25)
        # 4 条线重叠就会达到 1.0 (饱和)
        alpha_per_line = 0.25 
        
        for model_poses in processed_poses_list: # 使用处理过的列表
            model_points_2d = project_trajectory_to_image_official(model_poses, front_camera)
            if model_points_2d.shape[0] > 1:
                
                # a. 创建一个空白的覆盖层 (overlay)
                single_line_overlay = np.zeros_like(final_image)
                
                # b. 在这个空白层上绘制 *单条* 轨迹
                cv2.polylines(single_line_overlay, 
                              [model_points_2d.astype(np.int32)], 
                              isClosed=False, 
                              color=(255, 255, 255), # 用纯白色画
                              thickness=6,  # <-- 线条粗细
                              lineType=cv2.LINE_AA)
                
                # c. 创建 2D 掩码
                mask_2d = cv2.cvtColor(single_line_overlay, cv2.COLOR_BGR2GRAY) > 0
                
                # d. 在 Alpha Map 上 "累加" 透明度
                alpha_map[mask_2d] = alpha_map[mask_2d] + alpha_per_line

        # 3.2: 裁剪 Alpha Map，使最大透明度为 1.0
        alpha_map_clipped = np.clip(alpha_map, 0.0, 1.0)
        
        # 3.3: 将 2D Alpha Map 扩展到 3D (H, W, 3) 以便混合
        alpha_3d = np.stack([alpha_map_clipped]*3, axis=-1)

        # 4. 创建纯色的 "模型图层"
        # model_color_layer 是一张纯色的图，例如 (97, 112, 222)
        model_color_layer = np.zeros_like(final_image, dtype=np.uint8)
        model_color_layer[:] = color_model

        # 5. [核心] 进行最终的、一次性的 Alpha 混合
        #    公式: final = (foreground * alpha) + (background * (1.0 - alpha))
        
        # 转换 BGR uint8 图像为 float32
        final_image_f = final_image.astype(np.float32)
        model_color_layer_f = model_color_layer.astype(np.float32)
        
        # 执行混合
        blended_f = (model_color_layer_f * alpha_3d) + (final_image_f * (1.0 - alpha_3d))
        
        # 转回 uint8
        final_image = blended_f.astype(np.uint8)

        # 6. 保存最终混合后的图像
        cam_filename = output_dir / f"{model_name}_origin.png"
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