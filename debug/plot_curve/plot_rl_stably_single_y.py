#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


@dataclass(frozen=True)
class SeriesPoint:
    step: int
    value: float
    idx: int


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            yield idx, item


def load_experiment_series(jsonl_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (critic_df, val_df) for one experiment file.

    critic_df columns: step, critic_rewards_mean
    val_df columns: step, val_overall_reward

    Duplicated steps can occur (train/val mixed); keep last non-null per series.
    """

    critic_points: List[SeriesPoint] = []
    val_points: List[SeriesPoint] = []

    for idx, item in iter_jsonl(jsonl_path):
        step = item.get("step")
        if not isinstance(step, int):
            continue

        critic_mean = safe_get(item, ["critic", "rewards", "mean"])
        if _is_finite_number(critic_mean):
            critic_points.append(SeriesPoint(step=step, value=float(critic_mean), idx=idx))

        val_overall = safe_get(item, ["val", "overall_reward"])
        if _is_finite_number(val_overall):
            val_points.append(SeriesPoint(step=step, value=float(val_overall), idx=idx))

    def to_df(points: List[SeriesPoint], value_col: str) -> pd.DataFrame:
        if not points:
            return pd.DataFrame(columns=["step", value_col])
        df = pd.DataFrame(
            [{"step": p.step, value_col: p.value, "_idx": p.idx} for p in points]
        )
        df = df.sort_values(["step", "_idx"], ascending=[True, True])
        df = df.groupby("step", as_index=False).tail(1)
        df = df.sort_values("step")
        return df[["step", value_col]].reset_index(drop=True)

    return to_df(critic_points, "critic_rewards_mean"), to_df(val_points, "val_overall_reward")


def discover_jsonl_files(log_dir: Path, pattern: str) -> List[Path]:
    return [p for p in sorted(log_dir.glob(pattern)) if p.is_file()]


def load_runs_long_frames(jsonl_files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple runs into long-form dataframes.

    critic_long columns: Step, Run, critic_rewards_mean
    val_long columns:    Step, Run, val_overall_reward
    """

    critic_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    for path in jsonl_files:
        run_name = path.stem
        critic_df, val_df = load_experiment_series(path)

        for _, row in critic_df.iterrows():
            critic_rows.append(
                {
                    "Step": int(row["step"]),
                    "Run": run_name,
                    "critic_rewards_mean": float(row["critic_rewards_mean"]),
                }
            )

        for _, row in val_df.iterrows():
            val_rows.append(
                {
                    "Step": int(row["step"]),
                    "Run": run_name,
                    "val_overall_reward": float(row["val_overall_reward"]),
                }
            )

    critic_long = pd.DataFrame(critic_rows, columns=["Step", "Run", "critic_rewards_mean"])
    val_long = pd.DataFrame(val_rows, columns=["Step", "Run", "val_overall_reward"])
    return critic_long, val_long


def aggregate_mean_sd(long_df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Aggregate long-form rows into per-step mean/sd band.

    Returns columns: Step, mean, sd, lower, upper
    """

    if long_df.empty:
        return pd.DataFrame(columns=["Step", "mean", "sd", "lower", "upper"])

    g = long_df.groupby("Step", as_index=False)[y_col].agg(["mean", "std"]).reset_index()
    g = g.rename(columns={"std": "sd"})
    g["sd"] = g["sd"].fillna(0.0)
    g["lower"] = g["mean"] - g["sd"]
    g["upper"] = g["mean"] + g["sd"]
    g = g.sort_values("Step").reset_index(drop=True)
    return g[["Step", "mean", "sd", "lower", "upper"]]


def smooth_agg_band(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Soft smoothing for mean and its std band (rolling mean on mean/lower/upper)."""

    if window is None or window <= 1 or df.empty:
        return df
    out = df.copy()
    for col in ["mean", "lower", "upper"]:
        out[col] = (
            out[col]
            .rolling(window=window, min_periods=max(1, window // 2), center=True)
            .mean()
        )
    out["sd"] = (out["upper"] - out["lower"]).clip(lower=0.0) / 2.0
    return out


def plot_single_axis_aggregated(
    critic_long: pd.DataFrame,
    val_long: pd.DataFrame,
    save_path: Path,
    title: Optional[str],
    dpi: int,
    critic_smooth_window: int,
    val_smooth_window: int,
    color_critic: str,
    color_val: str,
    band_alpha: float,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(8, 5))

    if not critic_long.empty:
        critic_agg = smooth_agg_band(
            aggregate_mean_sd(critic_long.sort_values("Step"), "critic_rewards_mean"),
            critic_smooth_window,
        )
        ax.plot(
            critic_agg["Step"],
            critic_agg["mean"],
            color=color_critic,
            linewidth=2,
            # label="critic(mean±std)",
            label="critic/reward",
        )
        ax.fill_between(
            critic_agg["Step"],
            critic_agg["lower"],
            critic_agg["upper"],
            color=color_critic,
            alpha=band_alpha,
            linewidth=0,
            label="_nolegend_",
        )

    if not val_long.empty:
        val_agg = smooth_agg_band(
            aggregate_mean_sd(val_long.sort_values("Step"), "val_overall_reward"),
            val_smooth_window,
        )
        ax.plot(
            val_agg["Step"],
            val_agg["mean"],
            color=color_val,
            linewidth=2,
            marker="o",
            markersize=6,
            # label="val(mean±std)",
            label="val/reward",
        )
        ax.fill_between(
            val_agg["Step"],
            val_agg["lower"],
            val_agg["upper"],
            color=color_val,
            alpha=band_alpha,
            linewidth=0,
            label="_nolegend_",
        )


    # --- 增量修改：增加指示面板和箭头 ---
    # 1. 计算 X 轴 25% 处的 Step 值 (基于两组数据的总范围)
    all_series = []
    if not critic_agg.empty: all_series.append(critic_agg["Step"])
    if not val_agg.empty: all_series.append(val_agg["Step"])
    
    if all_series:
        combined_steps = pd.concat(all_series)
        s_min, s_max = combined_steps.min(), combined_steps.max()
        target_step_x = s_min + (s_max - s_min) * 0.4
    else:
        target_step_x = 0

    # 2. 定义 Panel 位置 (轴坐标 0-1)
    panel_x, panel_y = 0.18, 0.6
    
    # 绘制带外框的 Panel
    # ec='gray' 添加边框颜色, boxstyle='round,pad=0.4' 增加圆角和内边距
    ax.text(
        panel_x, panel_y, "mean ± std\n(over k=4 runs)",
        transform=ax.transAxes,
        fontsize=14, va='center', ha='center',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=1)
    )

    # 定义箭头通用样式：shrinkA/B 负责两端空隙 (points 单位)
    arrow_style = dict(arrowstyle="->", lw=1.5, shrinkA=40, shrinkB=5)

    # 3. 绘制指向 Val 的箭头 (Val 在上方 -> 箭头起点设在 Panel 右上侧)
    if not val_long.empty:
        # 找到 Step 最接近 target_step_x 的那一行
        row = val_agg.iloc[(val_agg['Step'] - target_step_x).abs().argsort()[:1]]
        target_xy = (row['Step'].values[0], row['mean'].values[0])
        
        ax.annotate(
            "", 
            xy=target_xy, xycoords='data',
            # 起点：Panel中心偏右(0.06) 且 偏上(0.025)
            xytext=(panel_x + 0.06, panel_y + 0.005), textcoords='axes fraction',
            arrowprops=dict(color=color_val, connectionstyle="arc3,rad=0.2", **arrow_style)
        )

    # 4. 绘制指向 Critic 的箭头 (Critic 在下方 -> 箭头起点设在 Panel 右下侧)
    if not critic_long.empty:
        # 找到 Step 最接近 target_step_x 的那一行
        row = critic_agg.iloc[(critic_agg['Step'] - target_step_x).abs().argsort()[:1]]
        target_xy = (row['Step'].values[0], row['mean'].values[0])
        
        ax.annotate(
            "", 
            xy=target_xy, xycoords='data',
            # 起点：Panel中心偏右(0.06) 且 偏下(0.025) -> 避免与上方箭头交叉
            xytext=(panel_x + 0.06, panel_y - 0.025), textcoords='axes fraction',
            arrowprops=dict(color=color_critic, connectionstyle="arc3,rad=-0.2", **arrow_style)
        )
    # --- 增量修改：增加指示面板和箭头 ---
    
    ax.set_xlabel("Steps", fontsize=16)
    ax.set_ylabel("reward", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    if title:
        ax.set_title(title, fontsize=16)

    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated critic/val reward on a single y-axis (mean±std band) from train_logs/*.jsonl."
        )
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).parent / "train_logs"),
        help="Directory containing jsonl logs (default: debug/plot_curve/train_logs).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="experiment_log_1x_v*.jsonl",
        help="Glob pattern for log files inside log_dir (default excludes hard).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/data/ccy/EasyR1/debug/plot_curve/out_stable",
        help="Output directory (default: current working directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rl_curves_single_y_nosoft_3",
        help="Output filename prefix.",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title.")
    parser.add_argument(
        "--critic_smooth_window",
        type=int,
        default=7,
        help="Smoothing window for critic curve/band (<=1 disables).",
    )
    parser.add_argument(
        "--val_smooth_window",
        type=int,
        default=1,
        help="Smoothing window for val curve/band (<=1 disables).",
    )
    parser.add_argument(
        "--color_critic",
        type=str,
        default="#4e79a7",
        help="Line/band color for critic curve (hex).",
    )
    parser.add_argument(
        "--color_val",
        type=str,
        default="#665191",
        help="Line/band color for val curve (hex).",
    )
    parser.add_argument(
        "--band_alpha",
        type=float,
        default=0.18,
        help="Alpha for the std band shading.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Output file format. Use pdf/svg for crisp vector graphics.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI (mainly for png).",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = discover_jsonl_files(log_dir, args.pattern)
    if not jsonl_files:
        raise SystemExit(f"No jsonl files found in {log_dir} with pattern {args.pattern}")

    critic_long, val_long = load_runs_long_frames(jsonl_files)

    out_path = out_dir / f"{args.prefix}_single_y.{args.format}"
    plot_single_axis_aggregated(
        critic_long,
        val_long,
        out_path,
        title=args.title,
        dpi=args.dpi,
        critic_smooth_window=args.critic_smooth_window,
        val_smooth_window=args.val_smooth_window,
        color_critic=args.color_critic,
        color_val=args.color_val,
        band_alpha=args.band_alpha,
    )

    print(f"Runs: {len(jsonl_files)}")
    print(f"critic_samples={len(critic_long)}, val_samples={len(val_long)}")
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
