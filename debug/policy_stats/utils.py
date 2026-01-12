import torch

def calculate_min_fde_torch(k_predicted_trajs, gt_trajectory):
    """
    在 GPU 上批量计算 minFDE@k.

    参数:
    k_predicted_trajs (torch.Tensor): shape 为 (B, k, 8, 3)
    gt_trajectory (torch.Tensor):     shape 为 (B, 8, 3)

    返回:
    torch.Tensor: 批次中所有场景的平均 minFDE (一个标量).
    """
    # 1. 提取最终 [x, y] 坐标
    # shape: (B, k, 2)
    final_preds_xy = k_predicted_trajs[..., -1, :2]
    
    # 2. 提取 GT 最终 [x, y] 坐标
    # shape: (B, 2)
    final_gt_xy = gt_trajectory[..., -1, :2]
    
    # 3. 扩展 GT 坐标以便广播
    # shape: (B, 1, 2)
    final_gt_xy_expanded = final_gt_xy.unsqueeze(1)
    
    # 4. 批量计算 k 个 FDE (利用广播)
    #    (B, k, 2) - (B, 1, 2) => (B, k, 2)
    #    torch.norm 在最后一个维度 (dim=-1) 上计算L2范数
    # shape: (B, k)
    all_fdes = torch.norm(final_preds_xy - final_gt_xy_expanded, p=2, dim=-1)
    
    # 5. 沿着 k 那个维度找到最小值
    # shape: (B,)
    min_fdes_per_scene, _ = torch.min(all_fdes, dim=1)
    
    # 6. 返回批次中的平均 minFDE
    return torch.mean(min_fdes_per_scene)

def calculate_avg_pfde_torch(k_predicted_trajs):
    """
    在 GPU 上批量计算 avg-pFDE@k.

    参数:
    k_predicted_trajs (torch.Tensor): shape 为 (B, k, 8, 3)

    返回:
    torch.Tensor: 批次中所有场景的平均 avg-pFDE (一个标量).
    """
    k = k_predicted_trajs.shape[1]
    
    # 如果 k < 2, 无法计算, 直接返回 0
    if k < 2:
        return torch.tensor(0.0, device=k_predicted_trajs.device)
        
    # 1. 提取最终 [x, y] 坐标
    # shape: (B, k, 2)
    final_preds_xy = k_predicted_trajs[..., -1, :2]
    
    # 2. 准备广播以计算 (B, k, k, 2) 的差异矩阵
    # shape: (B, k, 1, 2)
    t1 = final_preds_xy.unsqueeze(2)
    # shape: (B, 1, k, 2)
    t2 = final_preds_xy.unsqueeze(1)
    
    # 3. 批量计算所有 (k, k) 对之间的差异
    # shape: (B, k, k, 2)
    diffs = t1 - t2
    
    # 4. 计算 L2 距离, 得到 (B, k, k) 的距离矩阵
    # shape: (B, k, k)
    pairwise_dists = torch.norm(diffs, p=2, dim=-1)
    
    # 5. 我们只需要上三角 (不包括对角线) 的值
    #    首先获取 k x k 的上三角索引
    triu_indices = torch.triu_indices(k, k, offset=1)
    
    # 6. 提取每个场景的上三角距离
    # pairwise_dists[:, triu_indices[0], triu_indices[1]] 
    # -> shape (B, num_pairs)
    #    其中 num_pairs = k * (k - 1) / 2
    num_pairs = (k * (k - 1)) / 2
    
    # 7. 计算每个场景的平均 pFDE
    #    我们直接对所有场景的所有配对求和，然后除以总数
    #    (注意: .sum() / num_pairs 会得到 B 个场景的总和,
    #     所以我们用 .mean() 得到 B 个场景的平均值)
    # shape: (B,)
    avg_pfde_per_scene = torch.sum(pairwise_dists[:, triu_indices[0], triu_indices[1]], dim=1) / num_pairs

    # 8. 返回批次中的平均 avg-pFDE
    return torch.mean(avg_pfde_per_scene)