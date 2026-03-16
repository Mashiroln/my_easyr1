#!/usr/bin/env python3
"""Step 2: 从CSV文件夹合并并提取parquet"""
from __future__ import annotations

import pandas as pd
import os
from pathlib import Path
import fnmatch
import argparse
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

def _is_prefill_csv(path: Path) -> bool:
    name = path.name.lower()
    return "prefill" in name


def main(
    folder_path: str,
    output_parquet: str = None,
    csv_mode: str = "all",
    include_glob: str | None = None,
    exclude_glob: str | None = None,
    include_bak: bool = False,
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
):
    """Merge scorer CSVs under a folder into one parquet.

    Args:
        folder_path: directory containing one-trajectory-per-row scorer CSV(s)
        output_parquet: output parquet path. If None, auto-named.
        csv_mode: one of {"all", "prefill", "no_prefill"}
        include_glob: optional filename glob to include (matched against basename)
        exclude_glob: optional filename glob to exclude (matched against basename)
        include_bak: whether to include *_bak.csv files
    """
    if output_parquet is None:
        suffix = "" if csv_mode == "all" else f"_{csv_mode}"
        output_parquet = os.path.join(folder_path, f"generations_full{suffix}.parquet")

    file_paths = [Path(folder_path) / f for f in os.listdir(folder_path)]
    csv_files = [f for f in file_paths if f.suffix.lower() == ".csv"]

    if not csv_files:
        raise FileNotFoundError("指定文件夹中未找到CSV文件，请检查路径是否正确。")

    def keep_csv(p: Path) -> bool:
        name = p.name
        lower = name.lower()

        # Exclude pipeline-generated intermediate CSVs to avoid accidentally
        # merging them back into scorer parquet on subsequent runs.
        # These files are not one-trajectory-per-row scorer outputs.
        if lower.endswith(".tmp_filtered.csv") or ".tmp_filtered." in lower:
            return False
        if lower.startswith("group_stats") or "group_stats" in lower:
            return False
        # stats_parquet outputs e.g. generations_full_*.csv (token-level group stats)
        if lower.startswith("generations_full") and ("result" not in lower) and not lower.endswith("_rpc.csv"):
            return False

        if not include_bak and (lower.endswith("_bak.csv") or "_bak" in lower or lower.endswith(".bak.csv")):
            return False

        if csv_mode == "prefill" and not _is_prefill_csv(p):
            return False
        if csv_mode == "no_prefill" and _is_prefill_csv(p):
            return False

        if include_glob and not fnmatch.fnmatch(name, include_glob):
            return False
        if exclude_glob and fnmatch.fnmatch(name, exclude_glob):
            return False

        return True

    csv_files = [p for p in csv_files if keep_csv(p)]
    if not csv_files:
        raise FileNotFoundError(
            "根据 csv_mode/include_glob/exclude_glob 过滤后未找到任何CSV文件，请检查参数。"
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
        print(f"开始流式读取：{csv_file.name}")

        for chunk in pd.read_csv(
            csv_file,
            header=0,
            encoding="utf-8",
            chunksize=int(chunksize),
            low_memory=False,
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
            if 'valid' in chunk.columns:
                cond = chunk['valid'].astype(str).str.lower().str.strip() == 'true'
                chunk = chunk[cond].copy()
                chunk['valid'] = True
                total_dropped += (original_n - len(chunk))

            for col in metric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # Build schema from first non-empty chunk
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
                # Align columns to schema names to keep writer stable
                chunk = _align_columns(chunk, schema.names)
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
                total_written += len(chunk)
                file_written += len(chunk)

            if file_written and file_written % (int(chunksize) * 5) == 0:
                print(f"  已写入 {csv_file.name}: {file_written} 行 (累计写入 {total_written})")

            if max_rows is not None and total_read >= int(max_rows):
                break

        print(f"完成：{csv_file.name}，写入行数：{file_written}")

        if max_rows is not None and total_read >= int(max_rows):
            break

    if writer is not None:
        writer.close()
    else:
        raise ValueError("所有CSV分块均为空（或全部被 valid 过滤丢弃），未写出 parquet")

    print("\n合并完成！")
    print(f"  - 总读取行数: {total_read}")
    print(f"  - 总写入行数: {total_written}")
    if total_dropped:
        print(f"  - valid 过滤丢弃行: {total_dropped}")
    print(f"  - Parquet: {Path(output_parquet).absolute()}")
    return output_parquet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument(
        "--csv_mode",
        type=str,
        default="all",
        choices=["all", "prefill", "no_prefill"],
        help="选择合并哪些CSV：all/prefill/no_prefill（按文件名是否包含 'prefill' 判断）",
    )
    parser.add_argument(
        "--include_glob",
        type=str,
        default=None,
        help="仅包含匹配该 glob 的 CSV basename（例如 '*_rpc.csv'）",
    )
    parser.add_argument(
        "--exclude_glob",
        type=str,
        default=None,
        help="排除匹配该 glob 的 CSV basename（例如 '*_bak.csv'）",
    )
    parser.add_argument(
        "--include_bak",
        action="store_true",
        help="是否包含 *_bak.csv（默认会忽略，防止重复合并）",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="分块读取 CSV 的行数（越大越快但更占内存）",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="调试用：最多处理前 N 行",
    )
    args = parser.parse_args()
    main(
        args.folder_path,
        args.output_parquet,
        csv_mode=args.csv_mode,
        include_glob=args.include_glob,
        exclude_glob=args.exclude_glob,
        include_bak=args.include_bak,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
