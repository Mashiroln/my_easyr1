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
  - (Prefill mode) A ``.jsonl`` filter spec with denorm coarse centers,
    consumed by ``verl`` dataloader via ``data.token_filter_file``.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _load_center_denorm_from_meta(
    meta_jsonl: str,
    needed_tokens: set[tuple[str, int]],
) -> dict[tuple[str, int], list[list[float]]]:
    """Load center_denorm for (token, prefill_meta_id) pairs from prefill_meta.jsonl.

    Only raw denorm centers are returned — normalization is handled by the
    dataloader's TrajectoryNormalizer, keeping this pipeline norm-agnostic.

    Returns:
        {(token, prefill_meta_id): center_denorm_8x3}
    """
    import json
    from collections import defaultdict

    # Step 1: Load prefill_meta.jsonl → {prefill_meta_id: (centers_jsonl, centers_row, center_rank)}
    meta_by_id = {}
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta_id = int(obj["prefill_meta_id"])
            meta_by_id[meta_id] = {
                "centers_jsonl": str(obj["centers_jsonl"]),
                "centers_row": int(obj["centers_row"]),
                "center_rank": int(obj["center_rank"]),
            }

    # Step 2: Collect which (centers_jsonl, centers_row) we actually need
    centers_needed = defaultdict(set)  # {centers_jsonl: {centers_row}}
    token_to_meta = {}  # {(token, prefill_meta_id): meta_dict}

    for token, meta_id in needed_tokens:
        if meta_id not in meta_by_id:
            continue
        meta = meta_by_id[meta_id]
        centers_needed[meta["centers_jsonl"]].add(meta["centers_row"])
        token_to_meta[(token, meta_id)] = meta

    # Step 3: Load centers from centers_jsonl files
    centers_cache = {}  # {(centers_jsonl, centers_row): [[x,y,z], ...]}
    for centers_jsonl, rows in centers_needed.items():
        with open(centers_jsonl, "r", encoding="utf-8") as f:
            for row_i, line in enumerate(f):
                if row_i not in rows:
                    continue
                obj = json.loads(line)
                centers = obj.get("coarse_centers", [])
                centers_cache[(centers_jsonl, row_i)] = centers

    # Step 4: Build final lookup (denorm only, no normalization)
    lookup = {}
    for (token, meta_id), meta in token_to_meta.items():
        centers = centers_cache.get((meta["centers_jsonl"], meta["centers_row"]))
        if centers is None or meta["center_rank"] >= len(centers):
            continue

        center_denorm = centers[meta["center_rank"]]
        if not isinstance(center_denorm, list) or len(center_denorm) != 8:
            continue

        lookup[(token, meta_id)] = center_denorm

    return lookup


def main(
    csv_path: str,
    output_csv: str | None = None,
    diversity_threshold: float = 0.1,
    n_rollout: int = 8,
    group_size: int = 16,
    group_size_mode: str = "fixed",
    std_threshold: float = 0.01,
    confidence_threshold: float = 0.1,
    prefill_meta_jsonl: str | None = None,
) -> tuple[str, str, int]:
    """Run the three-stage dynamic-sample filter.

    Returns:
        ``(output_csv_path, txt_path, count)``
    """
    out_root = os.path.dirname(csv_path)
    if output_csv is None:
        stem = Path(csv_path).stem
        output_csv = os.path.join(out_root, f"{stem}_filtered_{diversity_threshold}.csv")

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

    # Write token list txt (backward compatible)
    txt_path = output_csv.replace(".csv", ".txt")
    with open(txt_path, "w") as f:
        if use_prefill_key:
            for _, row in final_df[["token", "prefill_meta_id"]].iterrows():
                f.write(f"{row['token']}\t{int(row['prefill_meta_id'])}\n")
        else:
            for token in final_df["token"]:
                f.write(f"{token}\n")
    print(f"Token list: {txt_path}  ({len(final_df)} entries)")

    # Write JSONL filter spec (self-contained, with denorm coarse centers for prefill)
    jsonl_path = output_csv.replace(".csv", ".jsonl")
    center_lookup = None
    if use_prefill_key and prefill_meta_jsonl:
        needed_tokens = {
            (str(row["token"]), int(row["prefill_meta_id"]))
            for _, row in final_df[["token", "prefill_meta_id"]].iterrows()
        }
        print(f"Loading center_denorm from prefill_meta: {prefill_meta_jsonl} ({len(needed_tokens)} needed)")
        center_lookup = _load_center_denorm_from_meta(prefill_meta_jsonl, needed_tokens)
        print(f"  Loaded {len(center_lookup)} center entries")

    with open(jsonl_path, "w") as f:
        for _, row in final_df.iterrows():
            entry = {"token": str(row["token"])}
            if use_prefill_key:
                meta_id = int(row["prefill_meta_id"])
                entry["prefill_meta_id"] = meta_id
                if center_lookup is not None:
                    center = center_lookup.get((str(row["token"]), meta_id))
                    if center is not None:
                        entry["coarse_center"] = center
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Filter spec JSONL: {jsonl_path}  ({len(final_df)} entries)")

    return output_csv, txt_path, len(final_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dynamic samples from group statistics")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("-p", "--diversity_threshold", type=float, default=0.1)
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--group_size_mode", type=str, default="fixed", choices=["fixed", "from_csv"])
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--conf", "--confidence_threshold", type=float, default=0.1,
                        dest="confidence_threshold")
    parser.add_argument("--prefill_meta_jsonl", type=str, default=None,
                        help="Prefill meta JSONL for loading coarse center trajectories")
    args = parser.parse_args()
    main(
        args.csv_path, args.output_csv, args.diversity_threshold,
        args.n_rollout, args.group_size, args.group_size_mode,
        args.std_threshold, args.confidence_threshold,
        prefill_meta_jsonl=args.prefill_meta_jsonl,
    )
