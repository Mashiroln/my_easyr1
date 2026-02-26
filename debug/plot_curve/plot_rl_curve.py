#!/usr/bin/env python3

"""Plot RL reward curves for multiple outer-loop runs + PDMS test scatter.

- Left y-axis: reward (critic.rewards.mean + val.overall_reward)
- Right y-axis: PDMS (test-set results), scatter

Expected logs (default):
  - train_logs/experiment_log_1x_v2.jsonl
  - train_logs/experiment_log_2x.jsonl
  - train_logs/experiment_log_3x.jsonl

PDMS test results are provided manually via PDMS_TEST_RESULTS below.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# -------------------------
# Manual PDMS test results
# -------------------------
# Mocked example: steps 0..140, irregular spacing (10 or 20).
# Replace with your real test-set PDMS list[dict] later.
PDMS_TEST_RESULTS: List[Dict[str, float]] = [
    {"step": 0, "pdms": 88.30},
    {"step": 10, "pdms": 88.55},
    {"step": 30, "pdms": 88.90},
    {"step": 40, "pdms": 89.05},
    {"step": 60, "pdms": 89.20},
    {"step": 70, "pdms": 89.10},
    {"step": 90, "pdms": 89.45},
    {"step": 100, "pdms": 89.60},
    {"step": 120, "pdms": 89.75},
    {"step": 140, "pdms": 89.85},
]


# -------------------------
# Truncation (outer-loop stitching)
# -------------------------
# Show only 1x in [0, 80], only 2x in [80, 110], only 3x in [110, 150].
# Boundaries 80 and 110 are allowed to appear twice.
SEGMENTS: Dict[str, Tuple[int, int]] = {
    "1x": (0, 80),
    "2x": (80, 110),
    "3x": (110, 150),
}

# Additional guide lines / highlighting
EXTRA_TRAIN_START = 130
EXTRA_TRAIN_END = 150
EXTRA_TRAIN_NOTE = ""


def safe_get(obj: Any, path: List[str]) -> Optional[Any]:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def load_reward_series(jsonl_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load (critic_df, val_df).

    critic_df columns: Step, reward
    val_df columns:    Step, reward

    Notes:
    - "val" may be missing.
    - Duplicated steps can exist; keep last seen value per series.
    """

    critic_by_step: Dict[int, float] = {}
    val_by_step: Dict[int, float] = {}

    for item in iter_jsonl(jsonl_path):
        step = item.get("step")
        if not isinstance(step, int):
            continue

        critic_mean = safe_get(item, ["critic", "rewards", "mean"])
        if _is_finite_number(critic_mean):
            critic_by_step[step] = float(critic_mean)

        val_overall = safe_get(item, ["val", "overall_reward"])
        if _is_finite_number(val_overall):
            val_by_step[step] = float(val_overall)

    critic_df = pd.DataFrame(
        sorted(((s, v) for s, v in critic_by_step.items()), key=lambda x: x[0]),
        columns=["Step", "reward"],
    )
    val_df = pd.DataFrame(
        sorted(((s, v) for s, v in val_by_step.items()), key=lambda x: x[0]),
        columns=["Step", "reward"],
    )
    return critic_df, val_df


def apply_rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window is None or window <= 1 or df.empty:
        return df
    out = df.copy()
    out["reward"] = (
        out["reward"].rolling(window=window, min_periods=max(1, window // 2), center=True).mean()
    )
    return out


def apply_segment_soft_with_locked_endpoints(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Smooth within a segment while keeping segment endpoints exactly at raw values.

    This makes markers (plotted from this df) lie on the curve, while avoiding
    confusing cross-boundary smoothing.
    """

    if df.empty:
        return df
    if window is None or window <= 1:
        return df

    raw = df.copy()
    sm = apply_rolling_mean(df, window)
    if len(sm) >= 1:
        sm.loc[sm.index[0], "reward"] = raw.loc[raw.index[0], "reward"]
        sm.loc[sm.index[-1], "reward"] = raw.loc[raw.index[-1], "reward"]
    return sm


def draw_segment_guides(ax: plt.Axes, segments: Dict[str, Tuple[int, int]]) -> None:
    """Draw vertical separators and top arrow brackets labeling segments."""

    # Background highlight for "continued training" region
    ax.axvspan(
        EXTRA_TRAIN_START,
        EXTRA_TRAIN_END,
        color="#9ca3af",
        alpha=0.12,
        zorder=0,
    )

    # Vertical separators: only step130 bold; 80/110/150 normal
    ax.axvline(x=80, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(x=110, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(x=EXTRA_TRAIN_START, color="#333333", linewidth=2.6, alpha=0.75, zorder=3)
    ax.axvline(x=EXTRA_TRAIN_END, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)

    # Top arrow brackets in axes coords
    y_arrow = 1.07
    y_text = 1.095
    for name, (s0, s1) in segments.items():
        ax.annotate(
            "",
            xy=(s1, y_arrow),
            xytext=(s0, y_arrow),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            arrowprops=dict(arrowstyle="<->", lw=1.8, color="#333333"),
            annotation_clip=False,
        )
        ax.text(
            0.5 * (s0 + s1),
            y_text,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
            clip_on=False,
        )


def draw_segment_guides_with_colors(
    ax: plt.Axes,
    segments: Dict[str, Tuple[int, int]],
    label_colors: Dict[str, str],
) -> None:
    """Same as draw_segment_guides(), but segment labels are colored by run."""

    # Background highlight for "continued training" region
    ax.axvspan(
        EXTRA_TRAIN_START,
        EXTRA_TRAIN_END,
        color="#9ca3af",
        alpha=0.12,
        zorder=0,
    )

    # Short note for the shaded region
    ax.text(
        0.5 * (EXTRA_TRAIN_START + EXTRA_TRAIN_END),
        0.96,
        EXTRA_TRAIN_NOTE,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="#6b7280",
        clip_on=False,
    )

    # Vertical separators: only step130 bold; 80/110/150 normal
    ax.axvline(x=80, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(x=110, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(x=EXTRA_TRAIN_START, color="#333333", linewidth=2.6, alpha=0.75, zorder=3)
    ax.axvline(x=EXTRA_TRAIN_END, color="#333333", linewidth=1.5, alpha=0.6, zorder=3)

    y_arrow = 1.07
    y_text = 1.095
    for name, (s0, s1) in segments.items():
        seg_color = label_colors.get(name, "#333333")
        ax.annotate(
            "",
            xy=(s1, y_arrow),
            xytext=(s0, y_arrow),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            arrowprops=dict(
                arrowstyle="<->",
                lw=1.8,
                color=seg_color,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=12,
            ),
            annotation_clip=False,
        )
        ax.text(
            0.5 * (s0 + s1),
            y_text,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            color=seg_color,
            clip_on=False,
        )


def split_by_step_boundary(df: pd.DataFrame, boundary: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (pre<=boundary, post>=boundary). Keeps boundary in both for continuity."""
    if df.empty:
        return df, df
    pre = df.loc[df["Step"] <= boundary].reset_index(drop=True)
    post = df.loc[df["Step"] >= boundary].reset_index(drop=True)
    return pre, post


def truncate_steps(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["Step"] >= start) & (df["Step"] <= end)
    return df.loc[m].reset_index(drop=True)


def pdms_df_from_list(pdms_list: List[Dict[str, float]]) -> pd.DataFrame:
    rows: List[Tuple[int, float]] = []
    for d in pdms_list:
        step = d.get("step")
        value = d.get("pdms")
        if isinstance(step, int) and _is_finite_number(value):
            rows.append((step, float(value)))
    return pd.DataFrame(sorted(rows, key=lambda x: x[0]), columns=["Step", "PDMS"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RL curves for 1x/2x/3x outer loops + PDMS scatter.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).parent / "train_logs"),
        help="Directory containing jsonl logs (default: debug/plot_curve/train_logs).",
    )
    parser.add_argument(
        "--logs",
        type=str,
        nargs="*",
        default=["experiment_log_1x_v2.jsonl", "experiment_log_2x.jsonl", "experiment_log_3x.jsonl"],
        help="List of jsonl files (relative to log_dir) to plot as 3 outer-loop runs.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=["1x", "2x", "3x"],
        help="Labels corresponding to --logs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.cwd()),
        help="Output directory (default: current working directory).",
    )
    parser.add_argument("--prefix", type=str, default="rl_curve", help="Output filename prefix.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs (png).")
    parser.add_argument(
        "--critic_smooth_window",
        type=int,
        default=7,
        help="Rolling smoothing window for critic reward curves (<=1 disables).",
    )
    parser.add_argument(
        "--val_smooth_window",
        type=int,
        default=1,
        help="Rolling smoothing window for val reward curves (<=1 disables). Default 1 (no smoothing).",
    )
    # Back-compat: older versions used a single --smooth_window for both.
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=None,
        help="(Deprecated) Alias for --critic_smooth_window.",
    )
    parser.add_argument(
        "--pdms_ylim",
        type=float,
        nargs=2,
        default=[88.0, 91.0],
        help="Right axis PDMS y-limits.",
    )
    args = parser.parse_args()

    if args.smooth_window is not None:
        args.critic_smooth_window = args.smooth_window

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.labels) != len(args.logs):
        raise SystemExit("--labels length must match --logs length")

    # Style (match existing scripts)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    # 3 outer-loop colors (clear and not the earlier green/purple pair)
    outer_colors = ["#4e79a7", "#e15759", "#f28e2b"]
    color_map = {label: outer_colors[i % len(outer_colors)] for i, label in enumerate(args.labels)}

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Segment separators + labels
    draw_segment_guides_with_colors(ax1, SEGMENTS, color_map)

    any_reward = False
    for label, rel in zip(args.labels, args.logs):
        path = log_dir / rel
        if not path.exists():
            print(f"[warn] missing log: {path}")
            continue

        critic_df, val_df = load_reward_series(path)

        # Truncation/stitching: each outer-loop only contributes within its segment.
        if label in SEGMENTS:
            s0, s1 = SEGMENTS[label]
            critic_df = truncate_steps(critic_df, s0, s1)
            val_df = truncate_steps(val_df, s0, s1)
        else:
            print(f"[warn] label {label!r} not in SEGMENTS; no truncation applied")

        # Smooth AFTER truncation to avoid mixing across outer-loop boundaries.
        critic_df_sm = apply_rolling_mean(critic_df, args.critic_smooth_window)
        val_df_plot = apply_segment_soft_with_locked_endpoints(val_df, args.val_smooth_window)

        # Phase styles: 0-130 solid, 130-150 dashed
        critic_pre, critic_post = split_by_step_boundary(critic_df_sm, EXTRA_TRAIN_START)
        val_pre, val_post = split_by_step_boundary(val_df_plot, EXTRA_TRAIN_START)

        # Critic: no markers
        if not critic_pre.empty:
            ax1.plot(
                critic_pre["Step"],
                critic_pre["reward"],
                color=color_map[label],
                linewidth=2,
                linestyle="-",
                alpha=0.95,
            )
            any_reward = True
        if not critic_post.empty:
            ax1.plot(
                critic_post["Step"],
                critic_post["reward"],
                color=color_map[label],
                linewidth=2,
                linestyle="--",
                alpha=0.95,
            )
            any_reward = True

        # Val: markers kept; linestyle depends on phase
        if not val_pre.empty:
            ax1.plot(
                val_pre["Step"],
                val_pre["reward"],
                color=color_map[label],
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=5,
                alpha=0.95,
            )
            any_reward = True
        if not val_post.empty:
            ax1.plot(
                val_post["Step"],
                val_post["reward"],
                color=color_map[label],
                linewidth=2,
                linestyle="--",
                marker="o",
                markersize=5,
                alpha=0.95,
            )
            any_reward = True

    ax1.set_xlabel("Steps", fontsize=16)
    ax1.set_ylabel("reward", fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)

    # Right axis: PDMS scatter
    ax2 = ax1.twinx()
    pdms_df = pdms_df_from_list(PDMS_TEST_RESULTS)
    if not pdms_df.empty:
        ax2.scatter(
            pdms_df["Step"],
            pdms_df["PDMS"],
            s=35,
            color="#222222",
            alpha=0.9,
            label="PDMS (test)",
            zorder=5,
        )

    ax2.set_ylabel("PDMS", fontsize=16)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(args.pdms_ylim[0], args.pdms_ylim[1])
    ax2.grid(False)

    # Legend: metric styles + PDMS.
    style_handles = [
        Line2D([0], [0], color="#000000", lw=2, linestyle="-", label="critic/reward"),
        Line2D([0], [0], color="#000000", lw=2, linestyle="-", marker="o", markersize=5, label="val/reward"),
        Line2D([0], [0], color="#222222", marker="o", linestyle="None", markersize=6, label="PDMS(test)"),
    ]
    ax1.legend(handles=style_handles, loc="lower right", fontsize=10, framealpha=0.9)

    if not any_reward and pdms_df.empty:
        print("[warn] nothing plotted")

    # Reserve a bit of space at the top for arrow brackets
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    out_path = out_dir / f"{args.prefix}.{args.format}"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
