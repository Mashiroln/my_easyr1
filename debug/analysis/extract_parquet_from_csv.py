import pandas as pd
import os
from pathlib import Path

# 1. 设置参数
# 替换为你的CSV文件所在文件夹路径（绝对路径或相对路径均可）
exp_name = "4B_navsim_ps01_120kv3_pn103kv3_selfnorm_stage2"
folder_path = f"/mnt/data/ccy/VLA_train/parallel_infer/output/{exp_name}"
# 合并后Parquet文件的保存路径和名称
output_parquet = os.path.join(folder_path, "generations_full.parquet")

# 2. 读取并筛选文件夹中的CSV文件
# 获取文件夹内所有文件路径
file_paths = [Path(folder_path) / f for f in os.listdir(folder_path)]
# 筛选出后缀为.csv的文件
csv_files = [f for f in file_paths if f.suffix.lower() == ".csv"]

if not csv_files:
    raise FileNotFoundError("指定文件夹中未找到CSV文件，请检查路径是否正确。")

metric_cols = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
    "score",
    "no_ec_epdms",
]

# 3. 批量读取CSV并合并
df_list = []
for csv_file in csv_files:
    # 读取单个CSV（header=0表示第一行为表头，encoding根据文件实际编码调整）
    df = pd.read_csv(
        csv_file,
        header=0,
        encoding="utf-8"  # 若文件为GBK编码，可改为 encoding="gbk"
    )
    df_list.append(df)
    print(f"已读取：{csv_file.name}，数据行数：{len(df)}")

# 合并所有DataFrame（忽略索引，避免重复）
# merged_df = pd.concat(df_list, ignore_index=True)
merged_df = pd.concat(df_list, ignore_index=True)
original_count = len(merged_df)

# =======================================================
# 筛选有效行 (Filter Valid Rows)
# =======================================================
if 'valid' in merged_df.columns:
    print(f"\n正在根据 'valid' 列进行筛选...")
    
    # 逻辑：
    # 1. .astype(str): 强制转为字符串，防止混合类型报错
    # 2. .str.lower(): 转为小写，处理 'True', 'true', 'TRUE'
    # 3. .str.strip(): 去除可能存在的空格
    # 4. == 'true': 只保留值为 true 的行
    condition = merged_df['valid'].astype(str).str.lower().str.strip() == 'true'
    
    merged_df = merged_df[condition].copy()
    
    # 可选：既然筛选后全是 True，为了 Parquet 存储规范，强制将该列设为布尔值 True
    merged_df['valid'] = True

    print(f"筛选前行数: {original_count}")
    print(f"筛选后行数: {len(merged_df)}")
    print(f"已丢弃坏行: {original_count - len(merged_df)}")
else:
    print("\n警告：未在数据中找到 'valid' 列，跳过筛选步骤。")

for col in metric_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(
            merged_df[col],
            errors="coerce"   # 非法值 → NaN
        )
        
# 4. 导出为Parquet格式
# engine="pyarrow"是Parquet的常用引擎，需提前安装pyarrow库
merged_df.to_parquet(output_parquet, engine="pyarrow", index=False)

# 输出结果信息
print(f"\n合并完成！")
print(f"合并后总数据行数：{len(merged_df)}")
print(f"Parquet文件保存路径：{Path(output_parquet).absolute()}")