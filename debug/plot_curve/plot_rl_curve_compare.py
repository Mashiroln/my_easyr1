#!/usr/bin/env python3

"""Compare ADAS stitched training (1x/2x/3x) vs hard_sdr baseline.

- Left y-axis: val reward (val.overall_reward)
- Right y-axis: PDMS (scatter)

ADAS:
- val reward is stitched by SEGMENTS (1x/2x/3x)
- PDMS is an EXTERNAL test-set metric: fill the PDMS_*_RESULTS lists below

hard_sdr:
- val reward uses the full log range with a single color
- PDMS is an EXTERNAL test-set metric: fill the PDMS_HARD_RESULTS list below

Design goals:
- Keep the visual style/logic from plot_rl_curve.py
- Remove all critic plotting
- PDMS is NOT read from any train log
- Keep things hard-coded: edit variables in this file
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# -------------------------
# Manual PDMS results
# -------------------------
# User expectation:
# - ADAS: step <-> PDMS is a bijection (one PDMS per 10/20 step, irregular spacing)
# - hard_sdr: PDMS also at 10/20 step spacing (NOT every training step)
#
# Fill these four lists directly.
# Default mock values (so PDMS is always visible):
# - ADAS PDMS: ~85 -> 91
# - hard PDMS: ~80 -> 50
PDMS_ADAS_1X_RESULTS: List[Dict[str, float]] = [
    {"step": 0, "pdms": 87.65},
    {"step": 20, "pdms": 88.36}, # to eval
    {"step": 40, "pdms": 88.91},
    {"step": 60, "pdms": 89.35},
    {"step": 80, "pdms": 89.61},
]
PDMS_ADAS_2X_RESULTS: List[Dict[str, float]] = [
    {"step": 80, "pdms": 89.61},
    {"step": 90, "pdms": 89.85}, # to eval
    {"step": 100, "pdms": 90.12}, # to eval
    {"step": 110, "pdms": 90.17},
]
PDMS_ADAS_3X_RESULTS: List[Dict[str, float]] = [
    {"step": 110, "pdms": 90.17},
    {"step": 120, "pdms": 90.28},
    {"step": 130, "pdms": 90.31},
    {"step": 140, "pdms": 90.37},
    {"step": 150, "pdms": 90.31},
]
PDMS_HARD_RESULTS: List[Dict[str, float]] = [
    {"step": 0, "pdms": 87.65},
    {"step": 20, "pdms": 85.53}, 
    {"step": 40, "pdms": 80.40}, 
    {"step": 60, "pdms": 77.75}, 
    {"step": 80, "pdms": 77.76}, 
    {"step": 100, "pdms": 77.68},
    {"step": 120, "pdms": 78.05},
    # {"step": 140, "pdms": 78.02},
]


# -------------------------
# Hard-coded paths & output
# -------------------------
PLOT_DIR = Path(__file__).resolve().parent
LOG_DIR = PLOT_DIR / "train_logs"

ADAS_LOGS = {
    "1x": "experiment_log_1x_v2.jsonl",
    "2x": "experiment_log_2x.jsonl",
    "3x": "experiment_log_3x.jsonl",
}
HARD_LOG = "experiment_log_hard_sdr.jsonl"

# Save to a FILE (not a directory)
OUT_PATH = PLOT_DIR / "rl_curve_compare.pdf"
OUT_DPI = 300

# Axis ranges (hard-coded to avoid reward/PDMS visually mixing)
REWARD_YLIM = (0.7, 1.0)
PDMS_YLIM = (76, 92)

# Marker sizing (edit these only)
# - `markersize` is in points (used by line markers)
# - `s` is marker area in points^2 (used by scatter)
VAL_MARKERSIZE = 4
HARD_VAL_MARKERSIZE = 3
PDMS_SCATTER_S = 55
HARD_PDMS_SCATTER_S = 40
PDMS_EDGE_WIDTH = 1.4


# -------------------------
# Truncation (outer-loop stitching)
# -------------------------
SEGMENTS: Dict[str, Tuple[int, int]] = {
    "1x": (0, 80),
    "2x": (80, 110),
    "3x": (110, 150),
}

# Additional guide lines / highlighting
EXTRA_TRAIN_START = 130
EXTRA_TRAIN_END = 150
EXTRA_TRAIN_NOTE = ""  # e.g. "cont. train" if desired


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


def load_val_reward_series(jsonl_path: Path) -> pd.DataFrame:
    """Load val.overall_reward into df columns: Step, reward.

    Duplicated steps can exist; keep last seen.
    """

    val_by_step: Dict[int, float] = {}
    for item in iter_jsonl(jsonl_path):
        step = item.get("step")
        if not isinstance(step, int):
            continue

        val_overall = safe_get(item, ["val", "overall_reward"])
        if _is_finite_number(val_overall):
            val_by_step[step] = float(val_overall)

    return pd.DataFrame(
        sorted(((s, v) for s, v in val_by_step.items()), key=lambda x: x[0]),
        columns=["Step", "reward"],
    )


def auto_pdms_ylim(dfs: List[pd.DataFrame], pad_ratio: float = 0.06) -> Optional[Tuple[float, float]]:
    vals: List[float] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        vals.extend([float(v) for v in df["PDMS"].tolist() if _is_finite_number(v)])
    if not vals:
        return None
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        pad = 1.0
    else:
        pad = (vmax - vmin) * float(pad_ratio)
    return (vmin - pad, vmax + pad)


def truncate_steps(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["Step"] >= start) & (df["Step"] <= end)
    return df.loc[m].reset_index(drop=True)


def split_by_step_boundary(df: pd.DataFrame, boundary: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (pre<=boundary, post>=boundary). Keeps boundary in both for continuity."""
    if df.empty:
        return df, df
    pre = df.loc[df["Step"] <= boundary].reset_index(drop=True)
    post = df.loc[df["Step"] >= boundary].reset_index(drop=True)
    return pre, post


def classify_segment(step: int) -> Optional[str]:
    # Boundary handling: assign boundary steps to the *next* stage.
    # e.g. step==80 -> 2x, step==110 -> 3x.
    items = list(SEGMENTS.items())
    if not items:
        return None
    for idx, (name, (s0, s1)) in enumerate(items):
        is_last = idx == len(items) - 1
        if is_last:
            if s0 <= step <= s1:
                return name
        else:
            if s0 <= step < s1:
                return name
    return None


def draw_segment_guides_with_colors(
    ax: plt.Axes,
    segments: Dict[str, Tuple[int, int]],
    label_colors: Dict[str, str],
) -> None:
    # Background highlight for "continued training" region
    ax.axvspan(
        EXTRA_TRAIN_START,
        EXTRA_TRAIN_END,
        color="#9ca3af",
        alpha=0.2,
        zorder=0.5,
    )

    # if EXTRA_TRAIN_NOTE:
    #     ax.text(
    #         0.5 * (EXTRA_TRAIN_START + EXTRA_TRAIN_END),
    #         0.96,
    #         EXTRA_TRAIN_NOTE,
    #         transform=ax.get_xaxis_transform(),
    #         ha="center",
    #         va="top",
    #         fontsize=9,
    #         color="#6b7280",
    #         clip_on=False,
    #     )
    note_text = "Extra\nTrain\nFor Rbt."
    ax.text(
        0.5 * (EXTRA_TRAIN_START + EXTRA_TRAIN_END),
        0.3,  # 这里的 0.5 表示 y 轴的正中间，你也可以根据需要调成 0.8 或更高
        note_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold", # 加粗更专业
        color="#4b5563",   # 使用深灰色文字
        style='italic',    # 使用斜体显得这是一个备注
        clip_on=False,
    )

    # Vertical separators: only step130 bold; 80/110/150 normal
    ax.axvline(x=80, color="#333333", linewidth=1.5, alpha=0.55, zorder=1.0)
    ax.axvline(x=110, color="#333333", linewidth=1.5, alpha=0.55, zorder=1.0)
    ax.axvline(x=EXTRA_TRAIN_START, color="#333333", linewidth=2.2, alpha=0.70, zorder=1.0)
    ax.axvline(x=EXTRA_TRAIN_END, color="#333333", linewidth=1.5, alpha=0.55, zorder=1.0)

    # Arrow brackets + colored segment labels
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


def pdms_df_from_list(pdms_list: List[Dict[str, float]]) -> pd.DataFrame:
    rows: List[Tuple[int, float]] = []
    for d in pdms_list:
        step = d.get("step")
        value = d.get("pdms")
        if isinstance(step, int) and _is_finite_number(value):
            rows.append((step, float(value)))
    return pd.DataFrame(sorted(rows, key=lambda x: x[0]), columns=["Step", "PDMS"])


def resolve_log_path(log_dir: Path, rel_or_path: str) -> Path:
    """Resolve a user-provided log path.

    Users may pass either:
    - a basename like "experiment_log_1x_v1.jsonl" (resolved under log_dir)
    - a relative path like "train_logs/experiment_log_1x_v1.jsonl" (resolved as-is)
    - an absolute path
    """

    p = Path(rel_or_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return log_dir / p


def main() -> int:
    log_dir = LOG_DIR
    out_path = OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    # Segment colors (match user expectation)
    stage_colors = {
        "1x": "#4e79a7",  # blue
        "2x": "#e15759",  # red
        "3x": "#f1ce63",  # yellow
    }
    adas_color_map = dict(stage_colors)

    # Baseline color: single color across all steps (distinct from ADAS palette)
    hard_color = "#6b7280"  # purple

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Segment separators + labels (only meaningful for ADAS)
    # Guides should always reflect 1x/2x/3x colors (not whatever labels the user chose)
    draw_segment_guides_with_colors(ax1, SEGMENTS, stage_colors)

    any_reward = False

    # ----- ADAS: val reward stitched (NO smoothing) -----
    adas_val: Dict[str, pd.DataFrame] = {}
    for label in ["1x", "2x", "3x"]:
        rel = ADAS_LOGS.get(label)
        if not rel:
            continue
        path = resolve_log_path(log_dir, rel)
        if not path.exists():
            print(f"[warn] missing ADAS log: {path}")
            continue
        df = load_val_reward_series(path)
        s0, s1 = SEGMENTS[label]
        df = truncate_steps(df, s0, s1)
        adas_val[label] = df

    for label in ["1x", "2x", "3x"]:
        df = adas_val.get(label, pd.DataFrame(columns=["Step", "reward"]))
        if df.empty:
            continue

        # Phase styles: <=130 solid, >=130 dashed
        df_pre, df_post = split_by_step_boundary(df, EXTRA_TRAIN_START)

        if not df_pre.empty:
            ax1.plot(
                df_pre["Step"],
                df_pre["reward"],
                color=adas_color_map[label],
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=VAL_MARKERSIZE,
                alpha=0.95,
                zorder=3.6,
            )
            any_reward = True

        if not df_post.empty:
            ax1.plot(
                df_post["Step"],
                df_post["reward"],
                color=adas_color_map[label],
                linewidth=2,
                linestyle="--",
                marker="o",
                markersize=VAL_MARKERSIZE,
                alpha=0.95,
                zorder=3.6,
            )
            any_reward = True

    # ----- hard_sdr: val reward across all steps -----
    hard_path = resolve_log_path(log_dir, HARD_LOG)
    if hard_path.exists():
        hard_val_df = load_val_reward_series(hard_path)
        ax1.plot(
            hard_val_df["Step"],
            hard_val_df["reward"],
            color=hard_color,
            linewidth=2.2,
            linestyle="-",
            marker="D",
            markersize=HARD_VAL_MARKERSIZE,
            markerfacecolor=hard_color,
            markeredgecolor=hard_color,
            alpha=0.9,
            zorder=2.8,
        )
        any_reward = True
    else:
        print(f"[warn] missing hard_sdr log: {hard_path}")

    ax1.set_xlabel("Steps", fontsize=16)
    ax1.set_ylabel("val reward", fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_ylim(REWARD_YLIM[0], REWARD_YLIM[1])

    # ----- PDMS: scatter on right axis -----
    ax2 = ax1.twinx()

    # ADAS PDMS (manual lists) – per-stage colors
    adas_pdms_1x = pdms_df_from_list(PDMS_ADAS_1X_RESULTS)
    adas_pdms_2x = pdms_df_from_list(PDMS_ADAS_2X_RESULTS)
    adas_pdms_3x = pdms_df_from_list(PDMS_ADAS_3X_RESULTS)

    if not adas_pdms_1x.empty:
        ax2.scatter(
            adas_pdms_1x["Step"],
            adas_pdms_1x["PDMS"],
            s=PDMS_SCATTER_S,
            facecolors="none",
            edgecolors=stage_colors["1x"],
            linewidths=PDMS_EDGE_WIDTH,
            alpha=0.95,
            marker="o",
            zorder=5.2,
        )
    if not adas_pdms_2x.empty:
        ax2.scatter(
            adas_pdms_2x["Step"],
            adas_pdms_2x["PDMS"],
            s=PDMS_SCATTER_S,
            facecolors="none",
            edgecolors=stage_colors["2x"],
            linewidths=PDMS_EDGE_WIDTH,
            alpha=0.95,
            marker="o",
            zorder=5.2,
        )
    if not adas_pdms_3x.empty:
        # ax2.scatter(
        #     adas_pdms_3x["Step"],
        #     adas_pdms_3x["PDMS"],
        #     s=PDMS_SCATTER_S,
        #     facecolors="none",
        #     edgecolors=stage_colors["3x"],
        #     linewidths=PDMS_EDGE_WIDTH,
        #     alpha=0.95,
        #     marker="o",
        #     zorder=5.2,
        # )
        # 1. 拆分数据
        mask_dashed = adas_pdms_3x["Step"] >= 140  # 140及以后的点
        df_solid = adas_pdms_3x[~mask_dashed]      # 140以前的点
        df_dashed = adas_pdms_3x[mask_dashed]      # 140及以后的点

        # 2. 绘制前半部分（实线边框，Step 110-130）
        if not df_solid.empty:
            ax2.scatter(
                df_solid["Step"],
                df_solid["PDMS"],
                s=PDMS_SCATTER_S,
                facecolors=stage_colors["3x"] + "44",  # 半透明填充
                edgecolors=stage_colors["3x"],         # 实色边框
                linewidths=PDMS_EDGE_WIDTH,
                linestyle="-",                         # <--- 实线
                marker="o",
                zorder=5.2,
            )

        # 3. 绘制后半部分（虚线边框，Step 140-150）
        if not df_dashed.empty:
            ax2.scatter(
                df_dashed["Step"],
                df_dashed["PDMS"],
                s=PDMS_SCATTER_S,
                facecolors=stage_colors["3x"] + "44",  # 半透明填充
                edgecolors=stage_colors["3x"],         # 实色边框
                linewidths=PDMS_EDGE_WIDTH,
                linestyle=(0, (1, 1)),                        # <--- 虚线！
                marker="o",
                zorder=5.2,
            )

    # hard_sdr PDMS: manual list only (external test metric)
    hard_pdms_df = pdms_df_from_list(PDMS_HARD_RESULTS)

    if not hard_pdms_df.empty:
        ax2.scatter(
            hard_pdms_df["Step"],
            hard_pdms_df["PDMS"],
            s=HARD_PDMS_SCATTER_S,
            facecolors="none",
            edgecolors=hard_color,
            linewidths=PDMS_EDGE_WIDTH,
            alpha=0.95,
            marker="D",
            zorder=4.2,
        )

    ax2.set_ylabel("PDMS", fontsize=16)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(PDMS_YLIM[0], PDMS_YLIM[1])
    ax2.grid(False)

    # Legend: keep it compact and emphasize ADAS vs hard_sdr difference
    style_handles = [
        Line2D([0], [0], color="#000000", lw=2, linestyle="-", marker="o", markersize=5, label="ADAS val/reward"),
        Line2D([0], [0], color="#000000", lw=2, linestyle="-", marker="D", markersize=5, label="Random val/reward"),
        Line2D(
            [0],
            [0],
            color="#000000",
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="#000000",
            label="ADAS PDMS",
        ),
        Line2D(
            [0],
            [0],
            color="#000000",
            marker="D",
            linestyle="None",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#000000",
            label="Random PDMS",
        ),
        # Patch(facecolor='#9ca3af', alpha=0.3, label="Extra Exp.\nFor Rbt."),
    ]
    ax1.legend(handles=style_handles, loc="lower left", fontsize=8, framealpha=0.9)

    if (
        (not any_reward)
        and adas_pdms_1x.empty
        and adas_pdms_2x.empty
        and adas_pdms_3x.empty
        and hard_pdms_df.empty
    ):
        print("[warn] nothing plotted")

    # Reserve space for arrow brackets
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(out_path, dpi=OUT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
