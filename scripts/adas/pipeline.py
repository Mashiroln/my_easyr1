#!/usr/bin/env python3
"""ADAS single-round pipeline: scorer CSVs -> token list for training.

Runs three steps:
  1. Merge scorer CSVs into a single Parquet.
     (1.5) Optionally enrich prefill Parquet with coarse trajectory data.
  2. Compute per-scenario group-level statistics.
  3. Filter dynamic (high-variance) samples and output a ``.txt`` token list.

The output ``.txt`` file is consumed directly by the verl dataloader via
``data.token_filter_file``, eliminating the need to pre-build a filtered
Parquet dataset.

ADAS outer loop (multi-round) is driven manually:
  Round N: run inference -> run this pipeline -> train with token_filter_file
  Round N+1: use new checkpoint, re-run inference, re-run pipeline, ...
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from scripts/adas/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_scorer_csv import main as merge_csv
from compute_stats import main as compute_stats
from filter_dynamic import main as filter_dynamic
from enrich_prefill import enrich_parquet as enrich_prefill_parquet
from enrich_prefill import _find_single_file


def run_pipeline(
    infer_folder: str,
    csv_mode: str = "all",
    # Filtering hyperparameters
    diversity_threshold: float = 0.1,
    n_rollout: int = 8,
    group_size: int = 16,
    std_threshold: float = 0.01,
    confidence_threshold: float = 0.1,
    # Target sample count for auto-tuning
    target_min: int = 3000,
    target_max: int = 6000,
    auto_tune: bool = True,
    # Optional overrides
    group_size_mode: str | None = None,
    enrich_prefill: bool | None = None,
    prefill_meta_jsonl: str | None = None,
    output_dir: str | None = None,
    include_glob: str | None = None,
    exclude_glob: str | None = None,
) -> str:
    """Run one round of ADAS filtering.

    Returns:
        Path to the output ``.txt`` token list file.
    """
    if group_size_mode is None:
        group_size_mode = "from_csv" if csv_mode == "prefill" else "fixed"
    if enrich_prefill is None:
        enrich_prefill = (csv_mode == "prefill")

    exp_name = os.path.basename(os.path.normpath(infer_folder))
    print("=" * 60)
    print(f"ADAS Pipeline - {exp_name}")
    print("=" * 60)

    # Step 1: Merge scorer CSVs into Parquet
    print("\n[Step 1] Merging scorer CSVs into Parquet...")
    parquet_path = merge_csv(
        infer_folder,
        output_parquet=None,
        csv_mode=csv_mode,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
    )

    # Step 1.5: Enrich prefill Parquet (conditional)
    if csv_mode == "prefill" and enrich_prefill:
        print("\n[Step 1.5] Enriching prefill Parquet...")
        meta_jsonl = prefill_meta_jsonl or _find_single_file(infer_folder, ".prefill_meta.jsonl")
        enriched_path = os.path.splitext(parquet_path)[0] + "_enriched.parquet"
        parquet_path = enrich_prefill_parquet(
            input_parquet=parquet_path,
            meta_jsonl=meta_jsonl,
            output_parquet=enriched_path,
        )

    # Step 2: Compute group-level statistics
    print("\n[Step 2] Computing group-level statistics...")
    stats_csv = compute_stats(parquet_path)

    # Step 3: Filter dynamic samples (with optional auto-tuning)
    print("\n[Step 3] Filtering dynamic samples...")

    if auto_tune:
        thresholds = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        best_threshold = diversity_threshold
        best_count = 0

        for thresh in thresholds:
            _, _, count = filter_dynamic(
                stats_csv, None, thresh, n_rollout, group_size,
                group_size_mode, std_threshold, confidence_threshold,
            )
            print(f"  threshold={thresh}: {count} samples")
            if target_min <= count <= target_max:
                best_threshold = thresh
                best_count = count
                print(f"  -> Selected: threshold={thresh}, count={count}")
                break
            elif count < target_min and count > best_count:
                best_threshold = thresh
                best_count = count

        diversity_threshold = best_threshold
        print(f"\nUsing diversity_threshold={diversity_threshold}")

    output_csv, txt_path, final_count = filter_dynamic(
        stats_csv, None, diversity_threshold, n_rollout, group_size,
        group_size_mode, std_threshold, confidence_threshold,
    )

    if final_count < target_min:
        print(f"WARNING: sample count {final_count} below target minimum {target_min}")
    elif final_count > target_max:
        print(f"WARNING: sample count {final_count} above target maximum {target_max}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Token list: {txt_path}")
    print(f"  Sample count: {final_count}")
    print(f"\nTo train with these samples:")
    print(f"  data.token_filter_file={txt_path}")
    print("=" * 60)

    return txt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADAS pipeline: scorer CSVs -> token list for training"
    )
    parser.add_argument("--infer_folder", type=str, required=True,
                        help="Directory containing scorer CSV output from parallel inference")
    parser.add_argument("--csv_mode", type=str, default="all",
                        choices=["all", "prefill", "no_prefill"])
    parser.add_argument("--diversity_threshold", type=float, default=0.1)
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    parser.add_argument("--target_min", type=int, default=3000)
    parser.add_argument("--target_max", type=int, default=6000)
    parser.add_argument("--no_auto_tune", action="store_true")
    parser.add_argument("--group_size_mode", type=str, default=None, choices=["fixed", "from_csv"])
    parser.add_argument("--include_glob", type=str, default=None)
    parser.add_argument("--exclude_glob", type=str, default=None)
    parser.add_argument("--prefill_meta_jsonl", type=str, default=None)

    args = parser.parse_args()
    run_pipeline(
        infer_folder=args.infer_folder,
        csv_mode=args.csv_mode,
        diversity_threshold=args.diversity_threshold,
        n_rollout=args.n_rollout,
        group_size=args.group_size,
        std_threshold=args.std_threshold,
        confidence_threshold=args.confidence_threshold,
        target_min=args.target_min,
        target_max=args.target_max,
        auto_tune=not args.no_auto_tune,
        group_size_mode=args.group_size_mode,
        include_glob=args.include_glob,
        exclude_glob=args.exclude_glob,
        prefill_meta_jsonl=args.prefill_meta_jsonl,
    )
