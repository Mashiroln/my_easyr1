#!/usr/bin/env python3
"""Select best poses per token from a JSONL file.

Rules:
- Any entry with pdms > 0.95 has highest priority. Once a token has a >0.95 entry, keep the first such entry (no replacement by even higher pdms values).
- If no >0.95 entry exists for a token, pick the entry with the highest pdms among the <=0.95 group.
- Output poses rounded to 2 decimals into a JSONL file.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PDMS_THRESHOLD = 0.95


def round_pose(poses: List[List[float]]) -> List[List[float]]:
    """Round pose coordinates to two decimals."""
    return [[round(x, 2) for x in pose] for pose in poses]


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}") from exc


def select_best_entries(records: Iterable[dict]) -> Tuple[Dict[str, dict], int]:
    """Select best entry per token per the pdms rules.

    Returns the best map and total lines processed.
    """
    best: Dict[str, dict] = {}
    total = 0
    for rec in records:
        total += 1
        token = rec.get("token")
        pdms = rec.get("pdms")
        poses = rec.get("poses")
        if token is None or pdms is None or poses is None:
            # skip malformed records silently
            continue

        current = best.get(token)
        has_high = pdms > PDMS_THRESHOLD

        if current is None:
            best[token] = {"pdms": pdms, "poses": poses}
            continue

        cur_pdms = current["pdms"]
        cur_high = cur_pdms > PDMS_THRESHOLD

        # If we already have a high priority one, keep it
        if cur_high and has_high:
            continue
        if cur_high and not has_high:
            continue
        if not cur_high and has_high:
            best[token] = {"pdms": pdms, "poses": poses}
            continue

        # Both non-high: take the larger pdms
        if pdms > cur_pdms:
            best[token] = {"pdms": pdms, "poses": poses}

    return best, total


def write_jsonl(path: Path, best: Dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for token, payload in best.items():
            rounded_poses = round_pose(payload["poses"])
            out = {
                "token": token,
                "poses": rounded_poses,
                "pdms": round(payload["pdms"], 4),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    input_path = Path("/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step/denorm_1111_policy_stats.jsonl")
    output_path = Path("/mnt/data/ccy/EasyR1/debug/analysis/norm_cot_text_130step/coarse_poses.jsonl")

    records = load_jsonl(input_path)
    best, total = select_best_entries(records)

    write_jsonl(output_path, best)

    print(f"Processed lines: {total}")
    print(f"Unique tokens: {len(best)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
