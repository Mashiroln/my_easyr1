#!/usr/bin/env python3
"""Compute per-scenario (or per-scenario-per-prefill) group-level statistics
from the merged scorer Parquet.

Outputs a CSV with columns: token, [prefill_meta_id], group_size,
pdms_mean, pdms_std, pdms_range, pdms_scaled_mean, pdms_scaled_std, pdms_scaled_range.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds


def main(parquet_path: str, output_csv: Optional[str] = None) -> str:
    out_root = os.path.dirname(parquet_path)
    parquet_stem = Path(parquet_path).stem
    tmp_csv_path = os.path.join(out_root, f"{parquet_stem}.tmp_filtered.csv")
    if output_csv is None:
        output_csv = os.path.join(out_root, f"{parquet_stem}.csv")

    dataset = ds.dataset(parquet_path, format="parquet")

    use_prefill_key = "prefill_meta_id" in dataset.schema.names
    group_cols = ["token", "prefill_meta_id"] if use_prefill_key else ["token"]

    # Streaming accumulators keyed by group identifier
    cnt: dict[object, int] = {}
    sum_pdms: dict[object, float] = {}
    sumsq_pdms: dict[object, float] = {}
    min_pdms: dict[object, float] = {}
    max_pdms: dict[object, float] = {}
    sum_pdms_scaled: dict[object, float] = {}
    sumsq_pdms_scaled: dict[object, float] = {}
    min_pdms_scaled: dict[object, float] = {}
    max_pdms_scaled: dict[object, float] = {}

    with open(tmp_csv_path, "w") as f_out:
        header_written = False
        for batch in dataset.to_batches():
            df = batch.to_pandas()
            if "pdms_scaled" in df.columns and "pdms" in df.columns:
                df = df[group_cols + ["pdms", "pdms_scaled"]]
            else:
                df = df[group_cols + ["score"]]
                df["pdms"] = df["score"]
                df["pdms_scaled"] = df["score"]

            df = df[df["token"].astype(str) != "average_all_frames"].copy()

            if use_prefill_key:
                df["prefill_meta_id"] = pd.to_numeric(df["prefill_meta_id"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["prefill_meta_id"])

            if not header_written:
                df.to_csv(f_out, index=False, header=True)
                header_written = True
            else:
                df.to_csv(f_out, index=False, header=False)

            df["pdms"] = pd.to_numeric(df["pdms"], errors="coerce")
            df["pdms_scaled"] = pd.to_numeric(df["pdms_scaled"], errors="coerce")
            df = df.dropna(subset=["pdms", "pdms_scaled", "token"])
            if df.empty:
                continue

            df["pdms_sq"] = df["pdms"] * df["pdms"]
            df["pdms_scaled_sq"] = df["pdms_scaled"] * df["pdms_scaled"]

            g = df.groupby(group_cols, sort=False).agg(
                n=("pdms", "size"),
                pdms_sum=("pdms", "sum"),
                pdms_sumsq=("pdms_sq", "sum"),
                pdms_min=("pdms", "min"),
                pdms_max=("pdms", "max"),
                pdms_scaled_sum=("pdms_scaled", "sum"),
                pdms_scaled_sumsq=("pdms_scaled_sq", "sum"),
                pdms_scaled_min=("pdms_scaled", "min"),
                pdms_scaled_max=("pdms_scaled", "max"),
            )

            for key, row in g.iterrows():
                if use_prefill_key:
                    token, meta_id = key
                    group_key: object = (str(token), int(meta_id))
                else:
                    group_key = str(key)

                n = int(row["n"])
                if n <= 0:
                    continue

                if group_key not in cnt:
                    cnt[group_key] = 0
                    sum_pdms[group_key] = 0.0
                    sumsq_pdms[group_key] = 0.0
                    min_pdms[group_key] = float(row["pdms_min"])
                    max_pdms[group_key] = float(row["pdms_max"])
                    sum_pdms_scaled[group_key] = 0.0
                    sumsq_pdms_scaled[group_key] = 0.0
                    min_pdms_scaled[group_key] = float(row["pdms_scaled_min"])
                    max_pdms_scaled[group_key] = float(row["pdms_scaled_max"])

                cnt[group_key] += n
                sum_pdms[group_key] += float(row["pdms_sum"])
                sumsq_pdms[group_key] += float(row["pdms_sumsq"])
                min_pdms[group_key] = min(min_pdms[group_key], float(row["pdms_min"]))
                max_pdms[group_key] = max(max_pdms[group_key], float(row["pdms_max"]))
                sum_pdms_scaled[group_key] += float(row["pdms_scaled_sum"])
                sumsq_pdms_scaled[group_key] += float(row["pdms_scaled_sumsq"])
                min_pdms_scaled[group_key] = min(min_pdms_scaled[group_key], float(row["pdms_scaled_min"]))
                max_pdms_scaled[group_key] = max(max_pdms_scaled[group_key], float(row["pdms_scaled_max"]))

    print(f"Intermediate CSV written: {tmp_csv_path}")

    def _std(sum_x: float, sumsq_x: float, n: int) -> float:
        if n <= 1:
            return float("nan")
        numerator = max(sumsq_x - (sum_x * sum_x) / n, 0.0)
        return math.sqrt(numerator / (n - 1))

    rows = []
    for k in cnt:
        n = cnt[k]
        row_out: dict = {}
        if use_prefill_key:
            token, meta_id = k  # type: ignore[misc]
            row_out["token"] = str(token)
            row_out["prefill_meta_id"] = int(meta_id)
        else:
            row_out["token"] = str(k)
        row_out.update({
            "group_size": n,
            "pdms_mean": sum_pdms[k] / n,
            "pdms_std": _std(sum_pdms[k], sumsq_pdms[k], n),
            "pdms_range": max_pdms[k] - min_pdms[k],
            "pdms_scaled_mean": sum_pdms_scaled[k] / n,
            "pdms_scaled_std": _std(sum_pdms_scaled[k], sumsq_pdms_scaled[k], n),
            "pdms_scaled_range": max_pdms_scaled[k] - min_pdms_scaled[k],
        })
        rows.append(row_out)

    grouped = pd.DataFrame(rows)
    if use_prefill_key:
        grouped["prefill_meta_id"] = pd.to_numeric(grouped["prefill_meta_id"], errors="coerce").astype("Int64")
        grouped = grouped.sort_values(["token", "prefill_meta_id"]).reset_index(drop=True)
    grouped.to_csv(output_csv, index=False)
    print(f"Stats written: {output_csv}  ({len(grouped)} groups, {sum(cnt.values())} total rows)")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute group-level PDMS statistics from scorer Parquet")
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()
    main(args.parquet_path, args.output_csv)
