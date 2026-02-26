#!/usr/bin/env python3
"""Select coarse poses with pdms aimed near 0.90 and below 0.95.

Strategy (no token drops):
- Load recog and human_gt JSONL streams and keep *all* entries per token.
- If a token has any 0.85 <= pdms < 0.95 entries, pick the one closest to TARGET_PDMS (tie → lower pdms).
- Otherwise, if only below-band entries exist, pick the highest pdms below 0.85 (still closest to target).
- Otherwise (all are >= 0.95), pick the lowest pdms to keep the global mean under 0.95.
- Output poses rounded to two decimals, along with mean pdms statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

PDMS_LOW = 0.85
PDMS_HIGH = 0.95
TARGET_PDMS = 0.90


def round_pose(poses: List[List[float]]) -> List[List[float]]:
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
                raise ValueError(f"Failed to parse JSON on line {line_no} of {path}") from exc


def gather_candidates(paths: List[Path]) -> Dict[str, List[dict]]:
    candidates: Dict[str, List[dict]] = {}
    for path in paths:
        for rec in load_jsonl(path):
            token = rec.get("token")
            pdms = rec.get("pdms")
            poses = rec.get("poses")
            if token is None or pdms is None or poses is None:
                continue
            candidates.setdefault(token, []).append({"pdms": float(pdms), "poses": poses})
    return candidates


def select_per_token(candidates: Dict[str, List[dict]]) -> Dict[str, dict]:
    selected: Dict[str, dict] = {}
    for token, recs in candidates.items():
        in_band = [r for r in recs if PDMS_LOW <= r["pdms"] < PDMS_HIGH]
        below_band = [r for r in recs if r["pdms"] < PDMS_LOW]

        if in_band:
            # Closest to target within the desired band; prefer lower pdms on ties.
            best = min(
                in_band,
                key=lambda r: (abs(r["pdms"] - TARGET_PDMS), r["pdms"]),
            )
        elif below_band:
            # Raise toward target by picking the highest pdms below the band.
            best = max(
                below_band,
                key=lambda r: (r["pdms"], -abs(r["pdms"] - TARGET_PDMS)),
            )
        else:
            # All candidates are >= PDMS_HIGH; pick the lowest to keep the mean down.
            best = min(
                recs,
                key=lambda r: (r["pdms"], abs(r["pdms"] - TARGET_PDMS)),
            )
        selected[token] = best
    return selected


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


def compute_mean(best: Dict[str, dict]) -> float:
    if not best:
        return 0.0
    return sum(item["pdms"] for item in best.values()) / len(best)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select coarse poses in desired pdms band")
    parser.add_argument(
        "--recog-path",
        type=Path,
        default=Path("/mnt/data/ccy/EasyR1/debug/analysis/recog/recog_diverse_policy_stats.jsonl"),
        help="Input JSONL from recog stats",
    )
    parser.add_argument(
        "--human-path",
        type=Path,
        default=Path("/mnt/data/ccy/EasyR1/debug/analysis/human_gt/navtrain_human_gt_policy_stats.jsonl"),
        help="Auxiliary human GT JSONL",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/mnt/data/ccy/EasyR1/debug/analysis/recog/coarse_poses.jsonl"),
        help="Output JSONL for selected poses",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [args.recog_path, args.human_path]

    candidates = gather_candidates(paths)
    selected = select_per_token(candidates)

    write_jsonl(args.output_path, selected)

    mean_pdms = compute_mean(selected)
    print(f"Tokens selected: {len(selected)}")
    print(f"Mean pdms: {mean_pdms:.4f}")
    print(f"Output written to: {args.output_path}")


if __name__ == "__main__":
    main()
