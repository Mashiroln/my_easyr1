import pandas as pd
from datasets import load_dataset
import os
import shutil

# --- 1. 请在这里修改您的文件路径 ---
dataset_path = "/mnt/data/ccy/EasyR1/data/navsim_cot1view_full_easyr1/"

# 现在是 txt 文件，每行一个 token
txt_path = "human_balanced/test100.txt"

out_name = "navsim_cot1view_full_easyr1_test100"
output_path = f"/mnt/data/ccy/EasyR1/data/{out_name}/data/test100.parquet"
test_output_path = f"/mnt/data/ccy/EasyR1/data/{out_name}/data/test.parquet"

# --- 2. 从 TXT 文件中加载 tokens ---
print(f"Loading tokens from {txt_path}...")

with open(txt_path, "r", encoding="utf-8") as f:
    # strip() 去掉换行与空白，过滤空行
    valid_tokens = {line.strip() for line in f if line.strip()}

print(f"Successfully loaded {len(valid_tokens)} unique tokens.")

# --- 3. 加载 Hugging Face 数据集 ---
print(f"Loading dataset from {dataset_path}...")
dataset = load_dataset(dataset_path, split="train")
print("Original dataset loaded. Number of records:", len(dataset))

# --- 4. 定义筛选函数并执行筛选 ---
def is_token_valid(example):
    return example["answer"]["token"] in valid_tokens

print("Filtering dataset...")
filtered_dataset = dataset.filter(is_token_valid, num_proc=4)
print("Filtering complete. Number of records in new dataset:", len(filtered_dataset))

# --- 5. 保存新的数据集为 Parquet 文件 ---
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

print(f"Saving filtered dataset to {output_path}...")
filtered_dataset.to_parquet(output_path)

shutil.copyfile(
    os.path.join(dataset_path, "data/test.parquet"),
    test_output_path
)

print("Done!")
