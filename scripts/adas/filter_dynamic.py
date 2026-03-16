#!/usr/bin/env python3
"""Filter dynamic (high-variance) samples from group-level statistics.

Uses a three-stage filter:
  1. Standard deviation threshold – remove near-zero variance scenarios.
  2. Diversity metric – binomial model selects scenarios with mixed outcomes.
  3. Confidence check – validate observed std against binomial prediction.

Outputs:
  - A filtered CSV with passing rows.
  - A ``.txt`` file listing selected tokens (one per line), consumed by
    ``verl`` dataloader via ``data.token_filter_file``.
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd


def main(
    csv_path: str,
    output_csv: str | None = None,
    diversity_threshold: float = 0.1,
    n_rollout: int = 8,
    group_size: int = 16,
    group_size_mode: str = "fixed",
    std_threshold: float = 0.01,
    confidence_threshold: float = 0.1,
) -> tuple[str, str, int]:
    """Run the three-stage dynamic-sample filter.

    Returns:
        ``(output_csv_path, txt_path, count)``
    """
    out_root = os.path.dirname(csv_path)
    if output_csv is None:
        output_csv = os.path.join(out_root, f"group_stats_filtered_{diversity_threshold}.csv")

    df = pd.read_csv(csv_path)
    print(f"Input rows: {len(df)}")

    use_prefill_key = "prefill_meta_id" in df.columns
    df_filtered = df.copy()

    # Stage 1: std filter
    df_filtered = df_filtered[df_filtered["pdms_std"] > std_threshold].copy()
    print(f"  Stage 1 (std > {std_threshold}): {len(df_filtered)} remaining")

    # Stage 2: diversity filter
    df_filtered = df_filtered[df_filtered["pdms_range"] > 1e-6].copy()
    df_filtered["p_est"] = df_filtered["pdms_mean"] / df_filtered["pdms_range"]

    diversity_metric = df_filtered["p_est"] ** n_rollout + (1 - df_filtered["p_est"]) ** n_rollout
    df_filtered["diversity_metric"] = diversity_metric
    df_filtered = df_filtered[df_filtered["diversity_metric"] < diversity_threshold].copy()
    print(f"  Stage 2 (diversity < {diversity_threshold}): {len(df_filtered)} remaining")

    # Stage 3: confidence check
    if not df_filtered.empty:
        if group_size_mode == "from_csv":
            if "group_size" not in df_filtered.columns:
                raise ValueError("group_size_mode='from_csv' requires a group_size column")
            n = pd.to_numeric(df_filtered["group_size"], errors="coerce").fillna(0).astype(int)
        else:
            n = pd.Series(group_size, index=df_filtered.index, dtype=int)

        n_safe = n.where(n > 1, other=np.nan)
        k_est = (df_filtered["p_est"] * n_safe).round()
        predicted_std = np.sqrt(k_est * (n_safe - k_est) / (n_safe * (n_safe - 1))) * df_filtered["pdms_range"]
        df_filtered["predicted_std"] = predicted_std

        confidence_error = np.abs(predicted_std - df_filtered["pdms_std"]) / df_filtered["pdms_std"]
        df_filtered["confidence_error"] = confidence_error

        final_df = df_filtered[df_filtered["confidence_error"] < confidence_threshold].copy()
        print(f"  Stage 3 (confidence < {confidence_threshold}): {len(final_df)} remaining")
    else:
        final_df = df_filtered

    # Write filtered CSV
    output_columns = df.columns.tolist()
    final_df[output_columns].to_csv(output_csv, index=False)
    print(f"Filtered CSV: {output_csv}")

    # Write token list txt
    txt_path = output_csv.replace(".csv", ".txt")
    with open(txt_path, "w") as f:
        if use_prefill_key:
            for _, row in final_df[["token", "prefill_meta_id"]].iterrows():
                f.write(f"{row['token']}\t{int(row['prefill_meta_id'])}\n")
        else:
            for token in final_df["token"]:
                f.write(f"{token}\n")
    print(f"Token list: {txt_path}  ({len(final_df)} entries)")

    return output_csv, txt_path, len(final_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dynamic samples from group statistics")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--diversity_threshold", type=float, default=0.1)
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--group_size_mode", type=str, default="fixed", choices=["fixed", "from_csv"])
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    args = parser.parse_args()
    main(
        args.csv_path, args.output_csv, args.diversity_threshold,
        args.n_rollout, args.group_size, args.group_size_mode,
        args.std_threshold, args.confidence_threshold,
    )
