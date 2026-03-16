#!/usr/bin/env python3
"""Step 4: 根据多样性和置信度筛选动态样本"""
import pandas as pd
import numpy as np
import io
import os
import argparse

def main(csv_path: str, output_csv: str = None,
         diversity_threshold: float = 0.1,
         n_rollout: int = 8,
         group_size: int = 16,
         group_size_mode: str = "fixed",
         std_threshold: float = 0.01,
         confidence_threshold: float = 0.1):

    out_root = os.path.dirname(csv_path)
    if output_csv is None:
        output_csv = os.path.join(out_root, f"group_stats_filtered_{diversity_threshold}.csv")

    csv_data = open(csv_path, "r").read()
    df = pd.read_csv(io.StringIO(csv_data))
    print(f"原始数据行数: {len(df)}")

    # If present, we are in prefill mode and the true identify is (token, prefill_meta_id)
    use_prefill_key = "prefill_meta_id" in df.columns

    df_filtered = df.copy()

    # 初步过滤
    df_filtered = df_filtered[df_filtered['pdms_std'] > std_threshold].copy()
    print(f"\n步骤1: std > {std_threshold} 过滤后，剩余行数: {len(df_filtered)}")

    # 估算比例 p
    df_filtered = df_filtered[df_filtered['pdms_range'] > 1e-6].copy()
    df_filtered['p_est'] = df_filtered['pdms_mean'] / df_filtered['pdms_range']

    # 计算多样性指标
    diversity_metric = df_filtered['p_est']**n_rollout + (1 - df_filtered['p_est'])**n_rollout
    df_filtered['diversity_metric'] = diversity_metric

    # 应用多样性过滤
    df_filtered = df_filtered[df_filtered['diversity_metric'] < diversity_threshold].copy()
    print(f"步骤2: 多样性过滤 (metric < {diversity_threshold}) 后，剩余行数: {len(df_filtered)}")

    # 执行置信度检验
    if not df_filtered.empty:
        if group_size_mode not in {"fixed", "from_csv"}:
            raise ValueError("group_size_mode must be 'fixed' or 'from_csv'")

        if group_size_mode == "from_csv":
            if "group_size" not in df_filtered.columns:
                raise ValueError("group_size_mode='from_csv' 但输入CSV缺少 group_size 列")
            n = pd.to_numeric(df_filtered["group_size"], errors="coerce").fillna(0).astype(int)
        else:
            n = pd.Series(group_size, index=df_filtered.index, dtype=int)

        # n<=1 时方差公式无意义，直接置为 NaN，后续会被 confidence_error 过滤掉
        n_safe = n.where(n > 1, other=np.nan)

        k_est = (df_filtered['p_est'] * n_safe).round()
        var_numerator = k_est * (n_safe - k_est)
        var_denominator = n_safe * (n_safe - 1)
        predicted_std = np.sqrt(var_numerator / var_denominator) * df_filtered['pdms_range']
        df_filtered['predicted_std'] = predicted_std

        confidence_error = np.abs(df_filtered['predicted_std'] - df_filtered['pdms_std']) / df_filtered['pdms_std']
        df_filtered['confidence_error'] = confidence_error

        final_df = df_filtered[df_filtered['confidence_error'] < confidence_threshold].copy()
        print(f"步骤3: 置信度检验 (error < {confidence_threshold}) 后，剩余行数: {len(final_df)}")
    else:
        final_df = df_filtered

    print("\n--- 筛选过程中的详细数据 ---")
    print(df_filtered.round(4))

    output_columns = df.columns.tolist()
    final_csv_output = final_df[output_columns].to_csv(index=False)
    with open(output_csv, "w") as f_out:
        f_out.write(final_csv_output)
    print("Saved filtered results to:", output_csv)

    # 同时输出 key 列表 txt
    txt_path = output_csv.replace(".csv", ".txt")
    with open(txt_path, "w") as f:
        if use_prefill_key:
            for _, row in final_df[["token", "prefill_meta_id"]].iterrows():
                f.write(f"{row['token']}\t{int(row['prefill_meta_id'])}\n")
            print(f"Key list saved to: {txt_path} (token\\tprefill_meta_id)")
        else:
            for token in final_df['token']:
                f.write(f"{token}\n")
            print(f"Token list saved to: {txt_path}")

    return output_csv, txt_path, len(final_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--diversity_threshold", type=float, default=0.1)
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument(
        "--group_size_mode",
        type=str,
        default="fixed",
        choices=["fixed", "from_csv"],
        help="fixed: 使用 --group_size 常量；from_csv: 使用输入统计CSV的 group_size 列（prefill 推荐）",
    )
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    args = parser.parse_args()
    main(args.csv_path, args.output_csv, args.diversity_threshold,
         args.n_rollout, args.group_size, args.group_size_mode, args.std_threshold, args.confidence_threshold)
