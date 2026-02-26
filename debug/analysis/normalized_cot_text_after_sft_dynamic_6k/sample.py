import pandas as pd

def main():
    input_csv = "group_stats_filtered_0.1.csv"    # 原始 CSV 路径
    output_csv = "group_stats_filtered_0.1_50percent.csv"  # 输出 CSV 路径
    sample_frac = 0.5          # 采样比例

    df = pd.read_csv(input_csv)
    df_sampled = df.sample(frac=sample_frac, random_state=42)
    df_sampled.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
