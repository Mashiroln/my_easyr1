"""Navsim PDMS reward (text) via SimScale UDS RPC.

This is a drop-in alternative to `navsim_reward_text.py` but uses the
Ray scorer UDS RPC service instead of HTTP.

Current implementation mirrors the *actually used* path you described:
- reward_type=batch
- use `compute_score_fast`
- ignore/omit compute_score_group (known-buggy in the HTTP version)

This implementation batches by token (same-token rollouts) using RPC
`score_group`, which significantly reduces per-request overhead.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

from verl.utils.reward_score.navsim.helper import get_trajectory_parser, denormalize
from verl.utils.reward_score.navsim.ray_scorer_rpc_client import RpcClientSync

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REWARD_NAME = "navsim_span_grpo"
REWARD_TYPE = "batch"

# ------------------------------
# Tunables (edit in this file)
# ------------------------------
# UDS socket path of the RPC gateway (run via SimScale's ray_scorer_rpc_server.sh).
RPC_UDS_PATH = os.getenv("NAVSIM_RPC_UDS", "/tmp/simscale_scorer_rpc.sock")
RPC_TIMEOUT_S = 120.0

# Batching / concurrency
TOKEN_WORKERS = 64  # concurrency across tokens (one task per token)
MAX_GROUP_SIZE = 16  # 0 means no cap; otherwise chunk per token to this size

# Denorm toggle (kept off by default to match your current training script)
ENABLE_DENORM = True

# Logging (keeps same behavior as navsim_reward_text.py)
_time_str = datetime.now().strftime("%m%d%H%M")
_exp_name = os.environ.get("EXP_NAME", "default_exp")
_log_file_path = f"/mnt/data/ccy/EasyR1/debug/analysis/generations_{_exp_name}_{_time_str}.jsonl"
_log_lock = threading.Lock()


def log_to_jsonl(data: dict, file_path: str) -> None:
    with _log_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


_tls = threading.local()


def _get_rpc_client(uds_path: str, timeout_s: float) -> RpcClientSync:
    c = getattr(_tls, "rpc_client", None)
    if c is None:
        c = RpcClientSync(uds_path=str(uds_path), timeout_s=float(timeout_s))
        c.connect()
        _tls.rpc_client = c
    return c


def _unwrap_score_group_result(result: Any) -> List[Dict[str, Any]]:
    """Normalize RpcClientSync.score_group() result to list[dict]."""

    if isinstance(result, dict) and isinstance(result.get("_result"), list):
        out = result.get("_result")
        return [x for x in out if isinstance(x, dict)]
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]
    if isinstance(result, dict):
        return [result]
    return []


def compute_score_fast(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """Compute reward scores for a batch.

    This is the function you said training actually uses.

    Notes:
    - Uses token batching (one RPC score_group per token, with optional chunking).
    - Denorm is disabled by default (matches your current training file).
    """

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for pdms reward function.")

    parse_fn = get_trajectory_parser()

    # Parse first, then decide per-sample vs per-token batching.
    parsed_items: List[Tuple[int, str, List[List[float]]]] = []
    scores: List[Dict[str, float]] = [None] * len(reward_inputs)  # type: ignore[list-item]

    def _mk_score(pdms: float, pdms_scaled: float) -> Dict[str, float]:
        # Keep same simplified behavior as your current training file.
        format_score = 1.0 if pdms > 0.8 else 0.0
        overall_score = 0.9 * float(pdms) + 0.1 * format_score
        return {
            "overall": float(overall_score),
            "format": float(format_score),
            "accuracy": float(pdms),
            "pdms": float(pdms),
        }

    for i, reward_input in enumerate(reward_inputs):
        try:
            response = reward_input["response"]
            token = reward_input["ground_truth"]["token"]
            poses: List[List[float]] = parse_fn(response)
            if poses is None or np.asarray(poses, dtype=float).shape != (8, 3):
                raise ValueError("Parser returned None or wrong shape")
        except Exception:
            scores[i] = _mk_score(0.0, 0.0)
            continue

        
        if ENABLE_DENORM and poses:
            poses = denormalize(poses, token=str(token))

        if len(poses) != 8:
            # Preserve old behavior: invalid length -> score 0.
            pdms, pdms_scaled = 0.0, 0.0
            scores[i] = _mk_score(pdms, pdms_scaled)
            log_to_jsonl(
                {
                    "poses": poses,
                    "token": token,
                    "pdms": float(pdms),
                    "pdms_scaled": float(pdms_scaled),
                    "format_score": 1.0,
                    "overall_score": float(pdms_scaled),
                },
                _log_file_path,
            )
            continue

        parsed_items.append((i, str(token), poses))

    # Batch by token: one RPC call per token (or per chunk if group is huge).
    token_groups: Dict[str, List[Tuple[int, List[List[float]]]]] = defaultdict(list)
    for idx, token, poses in parsed_items:
        token_groups[token].append((idx, poses))

    uds_path = RPC_UDS_PATH
    timeout_s = float(RPC_TIMEOUT_S)
    max_group_size = int(MAX_GROUP_SIZE)
    token_workers = int(TOKEN_WORKERS)

    def _chunks(items: List[Tuple[int, List[List[float]]]]) -> Iterable[List[Tuple[int, List[List[float]]]]]:
        if max_group_size <= 0 or len(items) <= max_group_size:
            yield items
            return
        for i in range(0, len(items), max_group_size):
            yield items[i : i + max_group_size]

    def _process_token(token: str, items: List[Tuple[int, List[List[float]]]]) -> List[Tuple[int, Dict[str, float]]]:
        out: List[Tuple[int, Dict[str, float]]] = []
        client = _get_rpc_client(uds_path=uds_path, timeout_s=timeout_s)

        for chunk in _chunks(items):
            indices = [x[0] for x in chunk]
            trajs = [x[1] for x in chunk]

            try:
                raw = client.score_group(token=token, trajectories_se2=trajs, verbose=False)
                results = _unwrap_score_group_result(raw)
            except Exception as e:
                logger.warning("[rpc] score_group failed token=%s n=%s err=%s", token, len(chunk), e)
                results = []

            # Align lengths; missing -> 0.
            if len(results) < len(chunk):
                results.extend({} for _ in range(len(chunk) - len(results)))

            for idx, poses, r in zip(indices, trajs, results):
                pdms = float(r.get("pdms", 0.0)) if isinstance(r, dict) else 0.0
                pdms_scaled = float(r.get("pdms_scaled", 0.0)) if isinstance(r, dict) else 0.0
                log_to_jsonl(
                    {
                        "poses": poses,
                        "token": token,
                        "pdms": float(pdms),
                        "pdms_scaled": float(pdms_scaled),
                        "format_score": 1.0,
                        "overall_score": float(pdms_scaled),
                    },
                    _log_file_path,
                )
                out.append((idx, _mk_score(pdms, pdms_scaled)))

        return out

    with ThreadPoolExecutor(max_workers=min(token_workers, max(1, len(token_groups)))) as executor:
        futures = {
            executor.submit(_process_token, token, items): token for token, items in token_groups.items() if items
        }
        for fut in as_completed(futures):
            token = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                logger.warning("[rpc] token batch failed token=%s err=%s", token, e)
                results = []
            for idx, s in results:
                scores[idx] = s

    # Fill any gaps.
    return [s if s is not None else _mk_score(0.0, 0.0) for s in scores]
