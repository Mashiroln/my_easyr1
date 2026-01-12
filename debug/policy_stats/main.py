import os
from pipeline import evaluate_experiments
from datetime import datetime

time_str = datetime.now().strftime("%m%d%H%M")

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    EXP_ROOT = "/mnt/data/ccy/EasyR1/debug/analysis"
    OUTPUT_ROOT = f"/mnt/data/ccy/EasyR1/debug/policy_stats/outputs/{time_str}"
    
    # 创建目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    exp_list = [
        os.path.join(EXP_ROOT, "navie_cot_text"),
        # os.path.join(EXP_ROOT, "normalized_cot_text"),
        # os.path.join(EXP_ROOT, "recog"),
        # os.path.join(EXP_ROOT, "norm_cot_text_88step_npu"),
        # os.path.join(EXP_ROOT, "norm_cot_text_130step")
    ]
    
    BATCH_SIZE = 1024
    NUM_WORKERS = 8

    # 2. GT (Ground Truth) 文件路径
    gt_jsonl_path = "/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl" 

    # 3. K 的最大值 (截断)
    MAX_K = 49
    
    # --- 运行评估 ---
    print("开始评估...")
    
    evaluate_experiments(
            exp_list, 
            gt_jsonl_path, 
            OUTPUT_ROOT,
            MAX_K,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )