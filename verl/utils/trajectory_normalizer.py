"""Shared trajectory normalizer for reward computation and prefill injection.

Handles dual-stats dispatch: regular tokens use NAVSIM_STAT_PATH,
synthetic tokens (matching -00\\d$) use NAVSIM_STAT_PATH_SYN.

NAVSIM_STAT_PATH_SYN supports two modes:
  1. Single JSON file  → all synthetic tokens (-00x) share this stats file.
  2. Index TXT file    → each line is an absolute path to a stats JSON.
     Token suffix -00x uses line x (0-indexed) from the TXT.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

_DEFAULT_STAT_PATH = "verl/utils/reward_score/navsim/trajectory_stats_train.json"
_SYN_TOKEN_RE = re.compile(r"-00(\d)$")


@dataclass(frozen=True)
class TrajStats:
    mean: np.ndarray  # (3,) or (8,3)
    std: np.ndarray


class TrajectoryNormalizer:
    """Normalize/denormalize 8x3 trajectories with dual-stats support.

    Usage:
        normalizer = TrajectoryNormalizer()  # reads from env
        poses_norm = normalizer.normalize(poses_denorm, token="abc123")
        poses_denorm = normalizer.denormalize(poses_norm, token="abc123")
        prefill_text = normalizer.format_prefill_text(poses_denorm, token="abc123")
    """

    def __init__(
        self,
        stat_path: Optional[str] = None,
        stat_path_syn: Optional[str] = None,
    ):
        stat_path = stat_path or os.environ.get("NAVSIM_STAT_PATH", _DEFAULT_STAT_PATH)
        stat_path_syn_raw = stat_path_syn or os.environ.get("NAVSIM_STAT_PATH_SYN", "").strip()

        self.stats = self._load(stat_path)

        # syn stats: single JSON, index TXT, or fallback to main stats
        self._syn_single: Optional[TrajStats] = None
        self._syn_indexed: Optional[List[str]] = None  # list of paths from TXT
        self._syn_cache: Dict[int, TrajStats] = {}

        if stat_path_syn_raw:
            if stat_path_syn_raw.endswith(".txt"):
                # Index mode: each line is a stats JSON path
                with open(stat_path_syn_raw, "r") as f:
                    self._syn_indexed = [line.strip() for line in f if line.strip()]
            else:
                # Single JSON mode
                self._syn_single = self._load(stat_path_syn_raw)

    @staticmethod
    def _load(path: str) -> TrajStats:
        with open(path, "r") as f:
            data = json.load(f)
        return TrajStats(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
        )

    def _pick_stats(self, token: Optional[str] = None) -> TrajStats:
        if not token:
            return self.stats
        m = _SYN_TOKEN_RE.search(str(token))
        if not m:
            return self.stats

        syn_idx = int(m.group(1))

        if self._syn_indexed is not None:
            # Index TXT mode: load (with cache) the stats for this digit
            if syn_idx not in self._syn_cache:
                self._syn_cache[syn_idx] = self._load(self._syn_indexed[syn_idx])
            return self._syn_cache[syn_idx]
        elif self._syn_single is not None:
            return self._syn_single
        else:
            return self.stats

    def normalize(self, poses_denorm: List[List[float]], token: Optional[str] = None) -> np.ndarray:
        """(8,3) denorm → (8,3) norm"""
        s = self._pick_stats(token)
        arr = np.asarray(poses_denorm, dtype=np.float64)
        return (arr - s.mean) / s.std

    def denormalize(self, poses_norm: List[List[float]], token: Optional[str] = None) -> np.ndarray:
        """(8,3) norm → (8,3) denorm"""
        s = self._pick_stats(token)
        arr = np.asarray(poses_norm, dtype=np.float64)
        return arr * s.std + s.mean

    def format_prefill_text(
        self,
        center_denorm: List[List[float]],
        token: Optional[str] = None,
        decimals: int = 2,
    ) -> str:
        """denorm center → normalize → format → prefill text string.

        Output format matches main_prefill.py:_build_prefill_text_from_center()
        """
        center_norm = self.normalize(center_denorm, token=token)
        traj_str = self._format_trajectory(center_norm, decimals=decimals)
        return '{\n  "coarse_trajectory": "<answer>' + traj_str + '</answer>",'

    @staticmethod
    def _format_trajectory(poses_norm: np.ndarray, prefix: str = "PT", decimals: int = 2) -> str:
        """Matches main_prefill.py:_format_trajectory_for_prefill() exactly."""
        fmt = f"{{:+.{decimals}f}}"

        def snap0(v):
            return 0.0 if abs(round(float(v), decimals)) <= 1e-2 else float(v)

        parts = [
            f"({fmt.format(snap0(p[0]))}, {fmt.format(snap0(p[1]))}, {fmt.format(snap0(p[2]))})"
            for p in poses_norm
        ]
        return f"[{prefix}, " + ", ".join(parts) + "]"
