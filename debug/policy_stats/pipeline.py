import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
import orjson
from contextlib import redirect_stdout


def calculate_min_fde_BATCH(k_predicted_trajs, gt_trajectory):
    final_preds_xy = k_predicted_trajs[..., -1, :2]
    final_gt_xy = gt_trajectory[..., -1, :2]
    final_gt_xy_expanded = final_gt_xy.unsqueeze(1)
    all_fdes = torch.norm(final_preds_xy - final_gt_xy_expanded, p=2, dim=-1)
    min_fdes_per_scene, _ = torch.min(all_fdes, dim=1)
    return min_fdes_per_scene

def calculate_avg_pfde_BATCH(k_predicted_trajs):
    B, k, T, D = k_predicted_trajs.shape
    if k < 2:
        return torch.zeros(B, device=k_predicted_trajs.device, dtype=torch.float32)
    final_preds_xy = k_predicted_trajs[..., -1, :2]
    t1 = final_preds_xy.unsqueeze(2)
    t2 = final_preds_xy.unsqueeze(1)
    diffs = t1 - t2
    pairwise_dists = torch.norm(diffs, p=2, dim=-1)
    sum_dists_per_scene = torch.sum(pairwise_dists.triu(diagonal=1), dim=[1, 2])
    num_pairs = (k * (k - 1)) / 2.0
    avg_pfde_per_scene = sum_dists_per_scene / num_pairs
    return avg_pfde_per_scene




class MetricCalculator(nn.Module):
    """
    将指标计算封装为 nn.Module 以便使用 DataParallel.
    一次性高效计算 minADE, minFDE, avg-pADE, avg-pFDE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, k_predicted_trajs, gt_trajectory):
        # k_predicted_trajs: (B, k, 8, 3)
        # gt_trajectory:     (B, 8, 3)
        
        k = k_predicted_trajs.shape[1]
        
        # --- 1. minADE 和 minFDE 计算 ---
        
        # (B, k, 8, 2)
        preds_xy = k_predicted_trajs[..., :2]
        # (B, 1, 8, 2)
        gt_xy_expanded = gt_trajectory[..., :2].unsqueeze(1) 
        
        # 计算所有时间步的 L2 距离
        # (B, k, 8, 2) - (B, 1, 8, 2) => (B, k, 8, 2)
        diffs_all_steps = preds_xy - gt_xy_expanded
        dists_all_steps = torch.norm(diffs_all_steps, p=2, dim=-1) # (B, k, 8)
        
        # FDE: 只取最后一个时间步的距离 (B, k)
        all_fdes = dists_all_steps[..., -1]
        # minFDE: (B,)
        min_fde, _ = torch.min(all_fdes, dim=1)
        
        # ADE: 取所有时间步的平均距离 (B, k)
        all_ades = torch.mean(dists_all_steps, dim=-1)
        # minADE: (B,)
        min_ade, _ = torch.min(all_ades, dim=1)

        # --- 2. avg-pADE 和 avg-pFDE 计算 ---
        
        # 如果 k < 2, 无法计算成对距离
        if k < 2:
            avg_pfde = torch.zeros_like(min_fde)
            avg_pade = torch.zeros_like(min_ade)
        else:
            # 唯一的配对数量
            num_pairs = (k * (k - 1)) / 2.0
            
            # avg-pFDE (成对最终距离)
            # (B, k, 2)
            final_preds_xy = preds_xy[..., -1, :] 
            t1_final = final_preds_xy.unsqueeze(2) # (B, k, 1, 2)
            t2_final = final_preds_xy.unsqueeze(1) # (B, 1, k, 2)
            # (B, k, k)
            pairwise_dists_final = torch.norm(t1_final - t2_final, p=2, dim=-1)
            # (B,)
            avg_pfde = torch.sum(pairwise_dists_final.triu(diagonal=1), dim=[1, 2]) / num_pairs
            
            # avg-pADE (成对平均距离)
            # (B, k, 1, 8, 2)
            t1_all = preds_xy.unsqueeze(2)
            # (B, 1, k, 8, 2)
            t2_all = preds_xy.unsqueeze(1) 
            # (B, k, k, 8)
            pairwise_dists_all_steps = torch.norm(t1_all - t2_all, p=2, dim=-1)
            # (B, k, k) - 在时间维度上求平均
            avg_pairwise_dists = torch.mean(pairwise_dists_all_steps, dim=-1)
            # (B,)
            avg_pade = torch.sum(avg_pairwise_dists.triu(diagonal=1), dim=[1, 2]) / num_pairs

        # 返回所有四个指标
        return min_ade, min_fde, avg_pade, avg_pfde


class GroupedKDataset(Dataset):
    """
    一个 Dataset，用于存储 K 值相同的 "一个分组" 的所有数据。
    """
    def __init__(self, data_dict):
        self.gt_list = data_dict["gt_list"]
        self.pred_list = data_dict["pred_list"]
        self.pdms_list = data_dict["pdms_list"]
        self.token_list = data_dict["token_list"]
        
    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        # 返回 CPU Tensors 和 token 字符串
        return (
            self.gt_list[idx],       # (8, 3)
            self.pred_list[idx],     # (k, 8, 3)
            self.pdms_list[idx],     # (k,)
            self.token_list[idx]     # string
        )

def custom_collate(batch):
    """
    将来自 GroupedKDataset 的 (gt, pred, pdms, token) 元组列表
    正确地堆叠成批次。
    """
    gt_batch = torch.stack([item[0] for item in batch])
    pred_batch = torch.stack([item[1] for item in batch])
    pdms_batch = torch.stack([item[2] for item in batch])
    token_batch = [item[3] for item in batch]
    
    return gt_batch, pred_batch, pdms_batch, token_batch

# ==============================================================================
# 数据加载 (不变)
# ==============================================================================

def load_gt_data(gt_jsonl_path):
    gt_data = {}
    print(f"Loading GT data from {gt_jsonl_path}...")
    try:
        with open(gt_jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                token = data['token']
                poses = torch.tensor(data['poses'], dtype=torch.float32, device='cpu')
                if poses.shape == (8, 3):
                    gt_data[token] = {"poses": poses}
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_jsonl_path}")
        exit(1)
    
    print(f"Loaded {len(gt_data)} GT scenarios.")
    return gt_data

def load_predictions_data(pred_jsonl_path, max_k):
    pred_data = defaultdict(lambda: {"poses_list": [], "pdms_list": []})
    with open(pred_jsonl_path, 'r') as f:
        pbar = tqdm(f, 
                        total=None, 
                        unit=" lines", 
                        desc=f"Loading {os.path.basename(pred_jsonl_path)}",
                        leave=False, 
                        position=1)
        for line in pbar:
            try:
                data = orjson.loads(line)
                token = data['token']
                if len(pred_data[token]["poses_list"]) < max_k:
                    poses = torch.tensor(data['poses'], dtype=torch.float32, device='cpu')
                    if poses.shape == (8, 3):
                        pred_data[token]["poses_list"].append(poses)
                        pred_data[token]["pdms_list"].append(data['pdms'])
            except Exception:
                pass
    return dict(pred_data)

# ==============================================================================
# 主评估流程 (!! 重构 !!)
# ==============================================================================

def evaluate_experiments(exp_list, gt_jsonl_path, output_dir, max_k=16, batch_size=1024, num_workers=8):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Running on CPU. Multi-GPU optimization will not apply.")
        num_workers = 0 # DataLoader 在 CPU 模式下 0 worker 通常更快
    
    # 1. 设置 Multi-GPU 计算器
    calculator = MetricCalculator().to(device)
    if torch.cuda.device_count() > 1 and device == 'cuda':
        print(f"--- Using {torch.cuda.device_count()} GPUs for calculation! (Batch Size per GPU: ~{batch_size // torch.cuda.device_count()}) ---")
        calculator = nn.DataParallel(calculator)
    else:
         print(f"--- Using 1 {device.upper()} (Batch Size: {batch_size}) ---")
    
    # 2. 统一加载 GT 数据 (到 CPU)
    gt_data = load_gt_data(gt_jsonl_path)
    if not gt_data: return

    all_reports = []

    # 3. 循环处理每个实验目录 (全局进度条 1/2)
    for exp_dir in tqdm(exp_list, desc="Overall Experiments", position=0):
        if not os.path.isdir(exp_dir): continue

        pred_jsonl_path = next((os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith("policy_stats.jsonl")), 
                                 os.path.join(exp_dir, "POLICY_STATS_FILE_NOT_FOUND"))
        exp_name = os.path.basename(exp_dir)
        if not os.path.exists(pred_jsonl_path): continue
        
        start_time = time.time()
        tqdm.write(f"\n--- 正在处理实验: {exp_name} ---")
        
        # 4. 加载预测 (到 CPU)
        pred_data = load_predictions_data(pred_jsonl_path, max_k)
        
        # 5. 按 K 分组 (在 CPU 上)
        grouped_data = defaultdict(lambda: {"gt_list": [], "pred_list": [], "pdms_list": [], "token_list": []})
        
        for token, gt_entry in tqdm(gt_data.items(), desc=f"Grouping {exp_name}", leave=False, position=1):
            if token not in pred_data: continue
            pred_poses_list = pred_data[token]["poses_list"]
            pdms_list = pred_data[token]["pdms_list"]
            k = len(pred_poses_list)
            if k == 0: continue
            
            k_predicted_trajs_stacked = torch.stack(pred_poses_list)
            grouped_data[k]["gt_list"].append(gt_entry["poses"])
            grouped_data[k]["pred_list"].append(k_predicted_trajs_stacked)
            grouped_data[k]["pdms_list"].append(torch.tensor(pdms_list, dtype=torch.float32))
            grouped_data[k]["token_list"].append(token)

        if not grouped_data:
            tqdm.write(f"实验 {exp_name} 未找到与GT匹配的 token。")
            continue

        # 6. !! 核心优化: 对每个 K 分组使用 DataLoader 批量计算 !!
        all_scene_results = []
        
        with torch.no_grad():
            # 迭代 K=1, K=2, ... K=16
            for k, data in tqdm(grouped_data.items(), desc=f"Computing K-Groups for {exp_name}", leave=False, position=1):
                
                # 6.1 创建 Dataset 和 DataLoader
                dataset = GroupedKDataset(data)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=custom_collate,
                    pin_memory=True  # 加速 CPU -> GPU 传输
                )
                
                # 6.2 迭代该 K 分组的 mini-batches
                for gt_batch, k_preds_batch, pdms_batch, token_batch in loader:
                    
                    # 6.3 异步将数据移至 GPU
                    gt_batch = gt_batch.to(device, non_blocking=True)
                    k_preds_batch = k_preds_batch.to(device, non_blocking=True)
                    pdms_batch = pdms_batch.to(device, non_blocking=True) # (B, k)
                    
                    # 6.4 Multi-GPU 并行计算
                    # (B,), (B,)
                    min_ades_batch, min_fdes_batch, avg_pade_k_batch, avg_pfde_k_batch = calculator(k_preds_batch, gt_batch)
                    
                    # 6.5 CPU 计算 (PDMS)
                    # (B,)
                    pdms_mean_batch = torch.mean(pdms_batch, dim=1)
                    
                    # 6.6 收集结果 (移回 CPU)
                    min_ades_cpu = min_ades_batch.cpu().numpy()
                    min_fdes_cpu = min_fdes_batch.cpu().numpy()
                    avg_pade_k_cpu = avg_pade_k_batch.cpu().numpy()
                    avg_pfde_k_cpu = avg_pfde_k_batch.cpu().numpy()
                    pdms_mean_cpu = pdms_mean_batch.cpu().numpy()
                    
                    B_mini = len(token_batch)
                    for i in range(B_mini):
                        all_scene_results.append({
                            "token": token_batch[i],
                            "k": k,
                            "min_ade": min_ades_cpu[i],
                            "min_fde": min_fdes_cpu[i],
                            "avg_pade_k": avg_pade_k_cpu[i],
                            "avg_pfde_k": avg_pfde_k_cpu[i],
                            "pdms_mean": pdms_mean_cpu[i]
                        })

        # --- 7. 输出 1: CSV ---
        df = pd.DataFrame(all_scene_results)
        csv_filename = os.path.join(output_dir, f"{exp_name}_scene_stats.csv")
        df.to_csv(csv_filename, index=False, float_format='%.4f')
        tqdm.write(f"成功保存场景统计数据到 {csv_filename}")

        # --- 8. 输出 2: 总结报告 ---
        report = {
            "Experiment": exp_name,
            "Scenarios": len(df),
            "Avg. k": df["k"].mean(),
            "minADE@k": df["min_ade"].mean(),
            "minFDE@k": df["min_fde"].mean(),
            "avg-pADE@k": df["avg_pade_k"].mean(),
            "avg-pFDE@k": df["avg_pfde_k"].mean(),
            "avg-PDMS": df["pdms_mean"].mean()
        }
        all_reports.append(report)
        tqdm.write(f"处理完毕 {exp_name} (用时: {time.time() - start_time:.2f}s)")

    # 9. 打印最终总结报告
    if not all_reports:
        print("\n未处理任何有效的实验。")
        return

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("\n" + "="*80)
            print(" " * 25 + "最终实验总结报告")
            print("="*80)
            summary_df = pd.DataFrame(all_reports)
            pd.set_option('display.width', 1000)
            pd.set_option('display.colheader_justify', 'left')
            pd.set_option('display.float_format', '{:,.4f}'.format)
            print(summary_df.to_string(index=False))
            print("="*80)