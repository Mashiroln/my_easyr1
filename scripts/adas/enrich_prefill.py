#!/usr/bin/env python3
"""Enrich prefill scorer Parquet with normalized coarse-center trajectory
and prefill text fields.

Only needed when ``csv_mode=prefill``.  For each scored trajectory row the
script adds the coarse center used as the prefill seed, both in denormalized
and normalized form, plus the formatted ``prefill_text`` string.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


@dataclass(frozen=True)
class PrefillMeta:
    prefill_meta_id: int
    centers_jsonl: str
    centers_row: int
    center_rank: int
    stats_path: str
    prefill_decimals: int
    cluster_size: Optional[int] = None


def _normalize_traj(center_denorm: List[List[float]], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(center_denorm, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"unexpected center shape {arr.shape}, expected (8,3)")
    return (arr - mean) / std


def _format_trajectory_for_prefill(poses_norm: np.ndarray, prefix: str = "PT", decimals: int = 2) -> str:
    parts: List[str] = []
    fmt = f"{{:+.{int(decimals)}f}}"

    def _snap0(v: float) -> float:
        if abs(round(float(v), int(decimals))) <= 1e-2:
            return 0.0
        return float(v)

    for p in poses_norm:
        x, y, z = _snap0(p[0]), _snap0(p[1]), _snap0(p[2])
        parts.append(f"({fmt.format(x)}, {fmt.format(y)}, {fmt.format(z)})")
    return f"[{prefix}, " + ", ".join(parts) + "]"


def build_prefill_text_from_center(
    center_denorm: List[List[float]], mean: np.ndarray, std: np.ndarray, decimals: int
) -> Tuple[List[List[float]], str]:
    traj_norm = _normalize_traj(center_denorm, mean, std)
    traj_norm_list: List[List[float]] = traj_norm.astype(float).tolist()
    traj_str = _format_trajectory_for_prefill(traj_norm, prefix="PT", decimals=decimals)
    prefill_text = '{\n  "coarse_trajectory": "<answer>' + traj_str + '</answer>",'
    return traj_norm_list, prefill_text


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_prefill_meta(meta_jsonl: str) -> Dict[int, PrefillMeta]:
    meta_by_id: Dict[int, PrefillMeta] = {}
    for obj in _read_jsonl(meta_jsonl):
        try:
            meta_id = int(obj["prefill_meta_id"])
            cluster_size = obj.get("cluster_size")
            if cluster_size is not None:
                try:
                    cluster_size = int(cluster_size)
                except Exception:
                    cluster_size = None
            meta_by_id[meta_id] = PrefillMeta(
                prefill_meta_id=meta_id,
                centers_jsonl=str(obj["centers_jsonl"]),
                centers_row=int(obj["centers_row"]),
                center_rank=int(obj["center_rank"]),
                stats_path=str(obj["stats_path"]),
                prefill_decimals=int(obj.get("prefill_decimals", 2)),
                cluster_size=cluster_size,
            )
        except Exception:
            continue
    if not meta_by_id:
        raise ValueError(f"No valid meta parsed from {meta_jsonl}")
    return meta_by_id


def load_stats_mean_std(stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(stats_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return np.asarray(obj["mean"], dtype=np.float32), np.asarray(obj["std"], dtype=np.float32)


def load_centers_rows(centers_jsonl: str, needed_rows: Iterable[int]) -> Dict[int, List[List[List[float]]]]:
    needed = set(int(x) for x in needed_rows)
    if not needed:
        return {}
    rows_to_centers: Dict[int, List[List[List[float]]]] = {}
    with open(centers_jsonl, "r", encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            if row_i not in needed:
                continue
            try:
                centers = json.loads(line).get("coarse_centers")
                if isinstance(centers, list):
                    rows_to_centers[int(row_i)] = centers
            except Exception:
                continue
    missing = needed - set(rows_to_centers)
    if missing:
        raise ValueError(f"centers_jsonl missing rows: {sorted(list(missing)[:10])}")
    return rows_to_centers


def enrich_parquet(
    input_parquet: str,
    meta_jsonl: str,
    output_parquet: str,
    max_rows: Optional[int] = None,
) -> str:
    meta_by_id = load_prefill_meta(meta_jsonl)

    centers_rows_needed: Dict[str, set[int]] = {}
    stats_paths_needed: set[str] = set()
    for m in meta_by_id.values():
        centers_rows_needed.setdefault(m.centers_jsonl, set()).add(int(m.centers_row))
        stats_paths_needed.add(m.stats_path)

    stats_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sp in sorted(stats_paths_needed):
        stats_cache[sp] = load_stats_mean_std(sp)

    centers_cache: Dict[Tuple[str, int], List[List[List[float]]]] = {}
    for centers_path, rows in centers_rows_needed.items():
        for row_i, centers in load_centers_rows(centers_path, rows).items():
            centers_cache[(centers_path, int(row_i))] = centers

    computed: Dict[int, Dict[str, Any]] = {}

    def compute_for_meta_id(meta_id: int) -> Dict[str, Any]:
        if meta_id in computed:
            return computed[meta_id]
        m = meta_by_id.get(int(meta_id))
        if m is None:
            out = {k: None for k in [
                "prefill_centers_jsonl", "prefill_stats_path", "prefill_decimals",
                "prefill_cluster_size", "prefill_center_denorm", "prefill_center_norm",
                "prefill_center_denorm_json", "prefill_center_norm_json", "prefill_text",
            ]}
            computed[meta_id] = out
            return out
        center_denorm = centers_cache[(m.centers_jsonl, int(m.centers_row))][int(m.center_rank)]
        mean, std = stats_cache[m.stats_path]
        traj_norm_list, prefill_text = build_prefill_text_from_center(center_denorm, mean, std, int(m.prefill_decimals))
        out = {
            "prefill_centers_jsonl": m.centers_jsonl,
            "prefill_stats_path": m.stats_path,
            "prefill_decimals": int(m.prefill_decimals),
            "prefill_cluster_size": m.cluster_size,
            "prefill_center_denorm": center_denorm,
            "prefill_center_norm": traj_norm_list,
            "prefill_center_denorm_json": json.dumps(center_denorm, ensure_ascii=False),
            "prefill_center_norm_json": json.dumps(traj_norm_list, ensure_ascii=False),
            "prefill_text": prefill_text,
        }
        computed[meta_id] = out
        return out

    dataset = ds.dataset(input_parquet, format="parquet")
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    fixed_schema: Optional[pa.Schema] = None
    seen_rows = 0

    for batch in dataset.to_batches():
        df = batch.to_pandas()
        if "prefill_meta_id" not in df.columns:
            raise ValueError("Input parquet missing prefill_meta_id column")

        for col in ["prefill_meta_id", "prefill_centers_row", "prefill_center_rank"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        if max_rows is not None:
            remaining = int(max_rows) - int(seen_rows)
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.iloc[:remaining].copy()

        meta_ids_series = pd.to_numeric(df["prefill_meta_id"], errors="coerce")
        unique_ids = pd.unique(meta_ids_series.dropna().astype(np.int64))
        payload_by_id: Dict[int, Dict[str, Any]] = {}
        for mid in unique_ids:
            payload_by_id[int(mid)] = compute_for_meta_id(int(mid))

        def _map(mid: Any, key: str):
            try:
                return payload_by_id.get(int(mid), {}).get(key)
            except Exception:
                return None

        for enrichment_key in [
            "prefill_centers_jsonl", "prefill_stats_path", "prefill_decimals",
            "prefill_cluster_size", "prefill_center_denorm", "prefill_center_norm",
            "prefill_center_denorm_json", "prefill_center_norm_json", "prefill_text",
        ]:
            df[enrichment_key] = df["prefill_meta_id"].map(lambda x, k=enrichment_key: _map(x, k))

        for col in ["prefill_decimals", "prefill_cluster_size"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        if writer is None:
            table0 = pa.Table.from_pandas(df, preserve_index=False)
            fixed_schema = table0.schema
            writer = pq.ParquetWriter(output_parquet, fixed_schema)
            writer.write_table(table0)
        else:
            assert fixed_schema is not None
            table = pa.Table.from_pandas(df, schema=fixed_schema, preserve_index=False)
            writer.write_table(table)

        seen_rows += len(df)
        if max_rows is not None and seen_rows >= int(max_rows):
            break

    if writer is not None:
        writer.close()

    print(f"Enriched parquet written: {output_parquet}")
    return output_parquet


def _find_single_file(folder: str, suffix: str) -> str:
    matches = sorted(str(p) for p in Path(folder).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No file matching *{suffix} under {folder}")
    if len(matches) > 1:
        matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return matches[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich prefill scorer Parquet")
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--meta_jsonl", type=str, default=None)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    folder = str(Path(args.input_parquet).parent)
    meta = args.meta_jsonl or _find_single_file(folder, ".prefill_meta.jsonl")
    out = args.output_parquet or str(Path(args.input_parquet).with_name(
        Path(args.input_parquet).stem + "_enriched.parquet"))

    enrich_parquet(args.input_parquet, meta, out, max_rows=args.max_rows)
