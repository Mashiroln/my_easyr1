#!/usr/bin/env python3
"""Step 5: 从筛选的token列表构造新数据集"""
from __future__ import annotations

import pandas as pd
import os
import shutil
import argparse
from typing import Iterable, Set

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def _read_key_list(txt_path: str):
    """Return either a set of tokens, or a set of (token, meta_id) pairs."""
    tokens: set[str] = set()
    pairs: set[tuple[str, int]] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                tokens.add(parts[0])
            elif len(parts) >= 2:
                tok = parts[0]
                try:
                    mid = int(parts[1])
                except Exception:
                    # fall back to token-only
                    tokens.add(tok)
                    continue
                pairs.add((tok, mid))
            else:
                continue
    return tokens, pairs


def main(
    dataset_path: str,
    txt_path: str,
    output_name: str,
    output_root: str = None,
    enriched_parquet: str | None = None,
):
    if output_root is None:
        output_root = "/mnt/data/ccy/EasyR1/data"

    output_path = os.path.join(output_root, output_name, "data/train.parquet")
    test_output_path = os.path.join(output_root, output_name, "data/test.parquet")

    print(f"Loading keys from {txt_path}...")
    token_set, pair_set = _read_key_list(txt_path)
    use_pairs = len(pair_set) > 0
    if use_pairs:
        valid_tokens = {t for (t, _) in pair_set}
        print(f"Successfully loaded {len(pair_set)} unique (token, prefill_meta_id) pairs. tokens={len(valid_tokens)}")
    else:
        valid_tokens = set(token_set)
        print(f"Successfully loaded {len(valid_tokens)} unique tokens.")

    # Prefer HuggingFace datasets if available; otherwise fall back to pure pyarrow processing.
    try:
        from datasets import load_dataset  # type: ignore

        print(f"Loading dataset from {dataset_path} via datasets...")
        dataset = load_dataset(dataset_path, split="train")
        print("Original dataset loaded. Number of records:", len(dataset))

        def is_token_valid(example):
            return example["answer"]["token"] in valid_tokens

        print("Filtering dataset...")
        filtered_dataset = dataset.filter(is_token_valid, num_proc=4)
        print("Filtering complete. Number of records in new dataset:", len(filtered_dataset))

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving filtered dataset to {output_path}...")
        filtered_dataset.to_parquet(output_path)
    except ModuleNotFoundError:
        # datasets not installed. Assume local parquet layout: <dataset_path>/data/train.parquet
        train_parquet = os.path.join(dataset_path, "data", "train.parquet")
        if not os.path.exists(train_parquet):
            raise FileNotFoundError(
                f"datasets 未安装，且未找到本地 parquet: {train_parquet}. "
                "请安装 datasets 或确认 dataset_path 指向含 data/train.parquet 的目录。"
            )

        print(f"datasets 未安装，改用 pyarrow 处理：{train_parquet}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = ds.dataset(train_parquet, format="parquet")
        schema = dataset.schema
        if schema.get_field_index("answer") < 0:
            raise ValueError("train.parquet 缺少 answer 列，无法按 answer.token 处理")
        answer_idx = schema.get_field_index("answer")

        token_values = pa.array(sorted(valid_tokens), type=pa.string())

        if not use_pairs:
            # Token-only filtering: write matched rows directly
            scanner = dataset.scanner()
            writer: pq.ParquetWriter | None = None
            written = 0
            for batch in scanner.to_batches():
                answer_arr = batch.column(answer_idx)
                token_arr = answer_arr.field("token")
                mask = pc.is_in(token_arr, value_set=token_values)
                mask = pc.fill_null(mask, False)
                filtered_batch = batch.filter(mask)
                if filtered_batch.num_rows == 0:
                    continue
                table = pa.Table.from_batches([filtered_batch])
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                written += table.num_rows
            if writer is not None:
                writer.close()
            print("Filtering complete. Number of records in new dataset:", written)
        else:
            # Pair-mode: build one row per (token, prefill_meta_id) and attach prefill-specific fields.
            if enriched_parquet is None:
                raise ValueError("txt 为 (token, prefill_meta_id) 时必须提供 enriched_parquet")

            # 1) Load per-key prefill payload from enriched parquet
            target_keys = set(pair_set)
            token_filter_values = pa.array(sorted(valid_tokens), type=pa.string())
            enrich_ds = ds.dataset(enriched_parquet, format="parquet")
            enrich_scanner = enrich_ds.scanner(
                columns=["token", "prefill_meta_id", "prefill_text", "prefill_center_norm"],
                filter=ds.field("token").isin(token_filter_values),
            )

            key_payload: dict[tuple[str, int], dict] = {}
            for b in enrich_scanner.to_batches():
                # Convert just the small subset to python
                tbl = pa.Table.from_batches([b])
                for row in tbl.to_pylist():
                    tok = row.get("token")
                    mid = row.get("prefill_meta_id")
                    if tok is None or mid is None:
                        continue
                    k = (str(tok), int(mid))
                    if k not in target_keys:
                        continue
                    if k in key_payload:
                        continue
                    key_payload[k] = {
                        "prefill_meta_id": int(mid),
                        "prefill_text": row.get("prefill_text"),
                        "prefill_center_norm": row.get("prefill_center_norm"),
                    }

            missing = len(target_keys) - len(key_payload)
            if missing:
                print(f"⚠️ enriched parquet 中未找到 {missing} 个 target key；将跳过这些 key")

            metas_by_token: dict[str, list[int]] = {}
            for (tok, mid) in target_keys:
                if (tok, mid) not in key_payload:
                    continue
                metas_by_token.setdefault(tok, []).append(mid)

            # 2) Scan source dataset, pick rows whose token is selected, then duplicate per meta
            source_scanner = dataset.scanner()
            out_records: list[dict] = []

            for batch in source_scanner.to_batches():
                answer_arr = batch.column(answer_idx)
                token_arr = answer_arr.field("token")
                mask = pc.is_in(token_arr, value_set=token_values)
                mask = pc.fill_null(mask, False)
                filtered = batch.filter(mask)
                if filtered.num_rows == 0:
                    continue

                tbl = pa.Table.from_batches([filtered])
                for row in tbl.to_pylist():
                    tok = row.get("answer", {}).get("token")
                    if tok is None:
                        continue
                    tok = str(tok)
                    mids = metas_by_token.get(tok)
                    if not mids:
                        continue
                    for mid in mids:
                        payload = key_payload.get((tok, mid))
                        if payload is None:
                            continue
                        new_row = dict(row)
                        new_row.update(payload)
                        out_records.append(new_row)

            table = pa.Table.from_pylist(out_records)
            pq.write_table(table, output_path)
            print("Constructed dataset rows:", table.num_rows)

    # 复制test集
    src_test = os.path.join(dataset_path, "data/test.parquet")
    if os.path.exists(src_test):
        shutil.copyfile(src_test, test_output_path)
        print(f"Test set copied to {test_output_path}")

    print("Done!")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="/mnt/data/ccy/EasyR1/data")
    parser.add_argument(
        "--enriched_parquet",
        type=str,
        default=None,
        help="(可选) prefill 模式下用于 join 的 *_enriched.parquet；当 txt 是 token\\tmeta_id 时必填",
    )
    args = parser.parse_args()
    main(args.dataset_path, args.txt_path, args.output_name, args.output_root, enriched_parquet=args.enriched_parquet)
