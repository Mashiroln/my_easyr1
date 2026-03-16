import pandas as pd
from datasets import load_dataset
import os
import shutil

# --- 1. 请在这里修改您的文件路径 ---
# 包含原始数据的Hugging Face数据集文件夹路径
dataset_path = "/mnt/data/ccy/EasyR1/data/qwen3vl_4b_prn103kv4_norm_cot_full/"
# 包含有效token的CSV文件路径
csv_path = "/mnt/data/ccy/VLA_train/parallel_infer/output/qwen3_vl_4b_ctr_segment_openvit_sapo/group_stats_filtered_0.5.csv"  # <-- 请修改为您的CSV文件路径
# 新的过滤后数据集的保存路径
out_name = "qwen3vl_4b_prn103kv4_norm_cot_ssapo_adas1x_2k"
output_path = f"/mnt/data/ccy/EasyR1/data/{out_name}/data/train.parquet" # <-- 请修改为您想保存的路径
test_output_path = f"/mnt/data/ccy/EasyR1/data/{out_name}/data/test.parquet"
# --- 2. 从CSV文件中加载tokens ---
print(f"Loading tokens from {csv_path}...")
# 使用pandas读取CSV
token_df = pd.read_csv(csv_path)
# 将'token'列转换为一个集合(set)，这样查找会非常快
valid_tokens = set(token_df['token'].tolist())
print(f"Successfully loaded {len(valid_tokens)} unique tokens.")

# --- 3. 加载Hugging Face数据集 ---
print(f"Loading dataset from {dataset_path}...")
dataset = load_dataset(dataset_path, split="train")
print("Original dataset loaded. Number of records:", len(dataset))

# --- 4. 定义筛选函数并执行筛选 ---
# 这个函数会检查每个样本的 'answer' 字段中的 'token' 是否在我们的有效token集合中
def is_token_valid(example):
    return example['answer']['token'] in valid_tokens

print("Filtering dataset...")
# 使用 .filter() 方法高效地筛选数据集
# num_proc 可以设置用于并行处理的CPU核心数，以加快速度
filtered_dataset = dataset.filter(is_token_valid, num_proc=4) 
print("Filtering complete. Number of records in new dataset:", len(filtered_dataset))

# --- 5. 保存新的数据集为Parquet文件 ---
# 确保输出目录存在
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

print(f"Saving filtered dataset to {output_path}...")
# 使用 to_parquet() 方法保存
filtered_dataset.to_parquet(output_path)
shutil.copyfile(os.path.join(dataset_path, "data/test.parquet"), test_output_path)
print("Done!")