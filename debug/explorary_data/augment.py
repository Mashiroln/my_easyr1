# 文件名: augment.py
import torch
import os
import orjson
from tqdm import tqdm
from pipeline import load_gt_data, load_predictions_data # 复用你的加载器

def compute_ade(pred_traj, gt_traj):
    """计算两条轨迹之间的 ADE"""
    # 确保输入是 (T, D)
    diff = pred_traj[..., :2] - gt_traj[..., :2]
    dists = torch.norm(diff, p=2, dim=-1)
    return torch.mean(dists)

def select_diverse_augmentations(pred_poses_list, pdms_list, intent_list, gt_poses, config):
    """
    为你描述的筛选逻辑的核心实现
    
    规则 1: PDMS > 0.95
    规则 2a: GT ADE > 阈值
    规则 2b: 选出的样本间 ADE > 阈值 (使用最远点采样)
    """
    
    # --- 规则 1: 筛选高 PDMS (高可行性) ---
    # 注意：我们假设 "pred PDMS - GT PDMS > 0.3" 规则无法实现，
    # 因为 GT .jsonl 文件中没有 PDMS。我们严格执行 PDMS > 0.95。
    
    high_quality_candidates = []
    for poses, pdms, intent in zip(pred_poses_list, pdms_list, intent_list):
        if pdms > config["PDMS_THRESHOLD"]:
            # --- 规则 2a: 筛选高 GT ADE (高新颖性) ---
            ade_vs_gt = compute_ade(poses, gt_poses)
            
            if ade_vs_gt > config["MIN_ADE_FROM_GT"]:
                high_quality_candidates.append({
                    "poses": poses, 
                    "pdms": pdms, 
                    "ade_vs_gt": ade_vs_gt.item(),
                    "intent": intent
                })

    if not high_quality_candidates:
        return []

    # --- 规则 2b: 贪心最远点采样 (保证内部多样性) ---
    
    # 1. 按 "与GT的距离" 降序排序，最远的优先
    high_quality_candidates.sort(key=lambda x: x["ade_vs_gt"], reverse=True)
    
    # 2. 选择第一个 (离GT最远的) 作为种子
    selected_augs = [high_quality_candidates[0]]
    remaining_cands = high_quality_candidates[1:]

    # 3. 迭代选择, 直到达到最大数量或没有符合条件的
    while len(selected_augs) < config["MAX_AUGMENT_PER_TOKEN"] and remaining_cands:
        
        best_next_cand_obj = None # 用于存储对象
        best_next_cand_idx = -1   # 用于存储索引
        max_min_dist = -1 
        
        # 使用 enumerate() 来获取索引
        for idx, cand in enumerate(remaining_cands):
            # 计算该候选点到 *所有已选点* 的最小 ADE
            min_dist_to_selected = min(
                [compute_ade(cand["poses"], sel["poses"]) for sel in selected_augs]
            )
            
            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_next_cand_obj = cand # 存储对象
                best_next_cand_idx = idx  # 存储索引
        
        # 4. 检查这个最佳候选点是否满足 "样本间最小距离" 阈值
        if max_min_dist > config["MIN_INTER_ADE"]:
            selected_augs.append(best_next_cand_obj)
            # 使用 .pop(index) 替代 .remove(object)
            remaining_cands.pop(best_next_cand_idx)
        else:
            # 如果距离最大的点都不满足阈值，说明剩下的都太近了，停止采样
            break
            
    return selected_augs


def process_experiment(exp_dir, gt_jsonl_path, output_jsonl_path, config):
    """
    主处理函数：加载数据、循环、调用筛选器、保存结果
    """
    
    # 1. 加载 GT (到 CPU)
    print(f"Loading GT data from {gt_jsonl_path}...")
    gt_data = load_gt_data(gt_jsonl_path)
    if not gt_data: return

    # 2. 加载 预测 (到 CPU)
    pred_jsonl_path = next((os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith("policy_stats.jsonl")), None)
    if not pred_jsonl_path:
        print(f"Error: 找不到 {exp_dir} 中的 policy_stats.jsonl")
        return

    exp_name = os.path.basename(exp_dir)
    print(f"Loading predictions for {exp_name}...")
    pred_data = load_predictions_data(pred_jsonl_path, config["MAX_K"])

    # 3. 循环处理每个 Token
    total_augmented = 0
    print(f"Processing scenes and writing to {output_jsonl_path}...")
    
    # 使用 'wb' 模式和 orjson 以获得最大写入速度
    with open(output_jsonl_path, "wb") as f:
        for token, gt_entry in tqdm(gt_data.items(), desc=f"Augmenting {exp_name}"):
            
            # 检查 GT 和 Pred 是否匹配
            if token not in pred_data: continue
            
            gt_poses = gt_entry["poses"].cpu() # 确保在 CPU
            pred_entry = pred_data[token]
            
            # 从 dict 转为 Tensors 列表 (确保在 CPU)
            pred_poses_list = [p.cpu() for p in pred_entry["poses_list"]]
            pdms_list = pred_entry["pdms_list"] # 已经是 float 列表
            intent_list = pred_entry.get("intent_list", ["default"] * len(pred_poses_list))

            if not pred_poses_list: continue

            # --- 调用核心筛选逻辑 ---
            selected = select_diverse_augmentations(
                pred_poses_list, 
                pdms_list,
                intent_list, 
                gt_poses, 
                config
            )

            # 4. 写入文件
            for i, aug in enumerate(selected):
                output_data = {
                    "token": token,
                    "aug_id": i, # 标记这是该 token 的第几个扩增数据
                    "poses": aug["poses"].tolist(), # 转回 list 以便 json 序列化
                    "pdms": aug["pdms"],
                    "ade_vs_gt": aug["ade_vs_gt"], # 保存这个有用的元数据
                    "intent": aug["intent"]
                }
                f.write(orjson.dumps(output_data) + b"\n")
                total_augmented += 1

    print(f"\n--- 任务完成 ---")
    print(f"实验: {exp_name}")
    print(f"总计生成 {total_augmented} 条扩增轨迹。")
    print(f"输出文件: {output_jsonl_path}")