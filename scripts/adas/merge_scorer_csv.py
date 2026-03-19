#!/usr/bin/env python3
"""Merge scorer CSV files from parallel inference into a single Parquet file.

Each CSV row represents one trajectory rollout scored by the simulator.
Invalid rows (valid != True) are dropped during merging.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _is_prefill_csv(path: Path) -> bool:
    return "prefill" in path.name.lower()


def main(
    folder_path: str,
    output_parquet: Optional[str] = None,
    csv_mode: str = "all",
    include_glob: Optional[str] = None,
    exclude_glob: Optional[str] = None,
    include_bak: bool = False,
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
) -> str:
    """Merge scorer CSVs under *folder_path* into one Parquet file.

    Args:
        folder_path: directory containing one-trajectory-per-row scorer CSVs.
        output_parquet: output path. Auto-named if None.
        csv_mode: ``"standard"`` (default) excludes prefill CSVs;
            ``"prefill"`` selects only prefill CSVs.
            Legacy aliases ``"all"`` and ``"no_prefill"`` map to ``"standard"``.
        include_glob / exclude_glob: optional filename globs applied to basename.
        include_bak: keep ``*_bak.csv`` files (default: skip).
        chunksize: rows per pandas chunk when streaming CSVs.
        max_rows: debug – stop after this many rows total.

    Returns:
        Absolute path to the written Parquet file.
    """
    # Normalize legacy mode names
    if csv_mode in ("all", "no_prefill"):
        csv_mode = "standard"

    if output_parquet is None:
        suffix = "_prefill" if csv_mode == "prefill" else ""
        output_parquet = os.path.join(folder_path, f"generations_full{suffix}.parquet")

    file_paths = [Path(folder_path) / f for f in os.listdir(folder_path)]
    csv_files = [f for f in file_paths if f.suffix.lower() == ".csv"]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    def keep_csv(p: Path) -> bool:
        name = p.name
        lower = name.lower()

        # Skip pipeline-generated intermediate files
        if lower.endswith(".tmp_filtered.csv") or ".tmp_filtered." in lower or lower.startswith("tmp_filtered"):
            return False
        if lower.startswith("group_stats") or "group_stats" in lower:
            return False
        if lower.startswith("generations_full") and ("result" not in lower) and not lower.endswith("_rpc.csv"):
            return False

        if not include_bak and (lower.endswith("_bak.csv") or "_bak" in lower or lower.endswith(".bak.csv")):
            return False

        if csv_mode == "prefill" and not _is_prefill_csv(p):
            return False
        if csv_mode == "standard" and _is_prefill_csv(p):
            return False

        if include_glob and not fnmatch.fnmatch(name, include_glob):
            return False
        if exclude_glob and fnmatch.fnmatch(name, exclude_glob):
            return False

        return True

    csv_files = [p for p in csv_files if keep_csv(p)]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files remain after filtering (csv_mode={csv_mode}, "
            f"include_glob={include_glob}, exclude_glob={exclude_glob})"
        )

    metric_cols = [
        "no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision_within_bound", "lane_keeping",
        "history_comfort", "two_frame_extended_comfort", "score",
        "no_ec_epdms",
    ]

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total_read = 0
    total_written = 0
    total_dropped = 0

    def _align_columns(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
        for c in target_cols:
            if c not in df.columns:
                df[c] = pd.NA
        extra = [c for c in df.columns if c not in target_cols]
        if extra:
            df = df.drop(columns=extra)
        return df[target_cols]

    for csv_file in csv_files:
        file_written = 0
        print(f"Reading: {csv_file.name}")

        for chunk in pd.read_csv(
            csv_file, header=0, encoding="utf-8",
            chunksize=int(chunksize), low_memory=False,
        ):
            if max_rows is not None and total_read >= int(max_rows):
                break

            total_read += len(chunk)
            if max_rows is not None and total_read > int(max_rows):
                overflow = total_read - int(max_rows)
                if overflow > 0:
                    chunk = chunk.iloc[:-overflow].copy()
                    total_read = int(max_rows)

            original_n = len(chunk)
            if "valid" in chunk.columns:
                cond = chunk["valid"].astype(str).str.lower().str.strip() == "true"
                chunk = chunk[cond].copy()
                chunk["valid"] = True
                total_dropped += original_n - len(chunk)

            for col in metric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            if writer is None:
                if chunk.empty:
                    continue
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(output_parquet, schema)
                writer.write_table(table)
                total_written += len(chunk)
                file_written += len(chunk)
            else:
                assert schema is not None
                chunk = _align_columns(chunk, schema.names)
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
                total_written += len(chunk)
                file_written += len(chunk)

            if max_rows is not None and total_read >= int(max_rows):
                break

        print(f"  Done: {csv_file.name}, rows written: {file_written}")
        if max_rows is not None and total_read >= int(max_rows):
            break

    if writer is not None:
        writer.close()
    else:
        raise ValueError("All CSV chunks were empty or filtered out – no Parquet written")

    print(f"\nMerge complete: {total_written} rows written ({total_dropped} invalid dropped)")
    print(f"  Parquet: {Path(output_parquet).absolute()}")
    return output_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge scorer CSVs into Parquet")
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--csv_mode", type=str, default="standard", choices=["standard", "prefill", "all", "no_prefill"])
    parser.add_argument("--include_glob", type=str, default=None)
    parser.add_argument("--exclude_glob", type=str, default=None)
    parser.add_argument("--include_bak", action="store_true")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    main(
        args.folder_path, args.output_parquet,
        csv_mode=args.csv_mode, include_glob=args.include_glob,
        exclude_glob=args.exclude_glob, include_bak=args.include_bak,
        chunksize=args.chunksize, max_rows=args.max_rows,
    )
