import torch
import os
import orjson
import json
from tqdm import tqdm
from datetime import datetime

# 从你现有的 pipeline 中导入预测加载器
# 我们不能用 load_gt_data，因为它不加载 GT PDMS
from pipeline import load_predictions_data

def load_gt_data_with_pdms(gt_jsonl_path):
    """
    一个*新*的 GT 加载器，专门用于加载 GT 的 PDMS 分数。
    """
    gt_data = {}
    print(f"Loading GT data (with PDMS) from {gt_jsonl_path}...")
    try:
        with open(gt_jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line) # 使用标准 json，与你的原始 pipeline 保持一致
                    token = data['token']
                    
                    # (!!!) 关键: 必须检查 GT 文件中是否有 'pdms'
                    if 'pdms' not in data:
                        # print(f"Warning: Token {token} in GT file has no 'pdms' key. Skipping.")
                        continue
                        
                    pdms = data['pdms']
                    poses = torch.tensor(data['poses'], dtype=torch.float32, device='cpu')
                    
                    if poses.shape == (8, 3):
                        gt_data[token] = {"poses": poses, "pdms": pdms}
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # 跳过格式错误的行
                    pass
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_jsonl_path}")
        return None
    
    if not gt_data:
         print("Error: 未找到任何包含 'pdms' 键的有效 GT 数据。")
         print(f"请检查文件: {gt_jsonl_path}")
         return None

    print(f"Loaded {len(gt_data)} GT scenarios (with PDMS).")
    return gt_data

def find_pdms_gaps(exp_dir, gt_jsonl_path, output_jsonl_path, config):
    """
    主处理函数：加载数据、循环、执行筛选、保存结果
    """
    
    # 1. 加载 GT (使用我们新的加载器)
    gt_data = load_gt_data_with_pdms(gt_jsonl_path)
    if not gt_data:
        return

    # 2. 加载 预测 (复用你的 pipeline 函数)
    pred_jsonl_path = next((os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith("policy_stats.jsonl")), None)
    if not pred_jsonl_path:
        print(f"Error: 找不到 {exp_dir} 中的 policy_stats.jsonl")
        return

    exp_name = os.path.basename(exp_dir)
    print(f"Loading predictions for {exp_name}...")
    pred_data = load_predictions_data(pred_jsonl_path, config["MAX_K"])

    # 3. 循环处理, 筛选, 写入
    total_filtered = 0
    print(f"Processing scenes and writing to {output_jsonl_path}...")
    
    # 使用 'wb' 模式和 orjson 以获得最大写入速度
    with open(output_jsonl_path, "wb") as f:
        
        # 迭代所有 GT token
        for token, gt_entry in tqdm(gt_data.items(), desc=f"Filtering {exp_name}"):
            
            # 检查模型是否也处理了此 token
            if token not in pred_data:
                continue
                
            gt_pdms = gt_entry["pdms"]
            pred_entry = pred_data[token]
            
            # 迭代该 token 的 K 条预测轨迹
            for pred_poses, pred_pdms in zip(pred_entry["poses_list"], pred_entry["pdms_list"]):
                
                # --- (!!!) 你的核心筛选逻辑 (!!!) ---
                if pred_pdms > gt_pdms:
                    
                    # (可选) 增加你上次提到的 "0.3" 差距阈值
                    pdms_gap = pred_pdms - gt_pdms
                    if pdms_gap < config["MIN_PDMS_GAP"]:
                        continue

                    # 筛选成功，准备写入
                    output_data = {
                        "token": token,
                        "pred_poses": pred_poses.cpu().tolist(), # 转为 list 写入 json
                        "pred_pdms": pred_pdms,
                        "gt_poses": gt_entry["poses"].cpu().tolist(),
                        "gt_pdms": gt_pdms,
                        "pdms_gap": pdms_gap # 保存这个 gap 信息
                    }
                    f.write(orjson.dumps(output_data) + b"\n")
                    total_filtered += 1

    print(f"\n--- 任务完成 ---")
    print(f"实验: {exp_name}")
    print(f"总计筛选出 {total_filtered} 条 PDMS gap > {config['MIN_PDMS_GAP']} 的轨迹。")
    print(f"输出文件: {output_jsonl_path}")

# ==============================================================================
# 执行入口
# ==============================================================================
if __name__ == "__main__":
    time_str = datetime.now().strftime("%m%d%H%M")
    
    # --- 1. 路径配置 ---
    EXP_ROOT = "/mnt/data/ccy/EasyR1/debug/analysis"
    # 换一个新的输出目录
    OUTPUT_ROOT = f"/mnt/data/ccy/EasyR1/debug/explorary_data/pdms_gap"
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 使用你上次指定的实验
    exp_name = "88step_augment_filter"
    exp_dir = os.path.join("/mnt/data/ccy/EasyR1/debug/explorary_data", exp_name)
    
    # GT (Ground Truth) 文件路径
    gt_jsonl_path = "/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl" 

    # --- 2. 筛选超参数配置 ---
    CONFIG = {
        # 根据你上一个 prompt: "pred PDMS - GT PDMS > 0.3"
        # 如果你只想满足 "pred > gt"，把这个值设为 0.0
        "MIN_PDMS_GAP": 0.0, 
        
        # 加载数据时使用 (与你 main.py 一致)
        "MAX_K": 8 
    }
    
    # --- 3. 运行筛选 ---
    output_path = os.path.join(OUTPUT_ROOT, f"pdms_gap_filtered_{exp_name}.jsonl")
    
    find_pdms_gaps(
        exp_dir=exp_dir,
        gt_jsonl_path=gt_jsonl_path,
        output_jsonl_path=output_path,
        config=CONFIG
    )