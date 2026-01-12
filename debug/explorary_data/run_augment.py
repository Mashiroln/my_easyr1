# 文件名: run_augment.py
import os
from augment import process_experiment
from datetime import datetime

time_str = datetime.now().strftime("%m%d%H%M")

if __name__ == "__main__":
    # --- 1. 路径配置 ---
    EXP_ROOT = "/mnt/data/ccy/EasyR1/debug/analysis"
    OUTPUT_ROOT = f"/mnt/data/ccy/EasyR1/debug/explorary_data/{time_str}"
    
    # 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 你指定的实验
    exp_name = "recog_multi_intent"
    exp_dir = os.path.join(EXP_ROOT, exp_name)
    
    # GT (Ground Truth) 文件路径
    gt_jsonl_path = "/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl" 

    # --- 2. 筛选超参数配置 ---
    
    # 基于你的先验:
    # avg-PDMS: 0.9155 -> 0.95 是个合理的
    # minADE@k: 0.2689 -> 1.0 以上算 "显著不同"
    # avg-pADE@k: 0.6407 -> 0.8 以上算 "彼此不同"
    
    CONFIG = {
        # 规则 1: PDMS 阈值
        "PDMS_THRESHOLD": 0.99,
        
        # 规则 2a: 离 GT "足够远" 的 ADE 阈值
        "MIN_ADE_FROM_GT": 0.7, 
        
        # 规则 2b: 样本之间 "足够远" 的 ADE 阈值
        "MIN_INTER_ADE": 0.5, 
        
        # 规则 2b: 每个 token 最多采样几条
        "MAX_AUGMENT_PER_TOKEN": 3,
        
        # 加载数据时使用 (与你 main.py 一致)
        "MAX_K": 8 
    }
    
    # --- 3. 运行扩增 ---
    output_path = os.path.join(OUTPUT_ROOT, f"augmented_{exp_name}.jsonl")
    
    process_experiment(
        exp_dir=exp_dir,
        gt_jsonl_path=gt_jsonl_path,
        output_jsonl_path=output_path,
        config=CONFIG
    )