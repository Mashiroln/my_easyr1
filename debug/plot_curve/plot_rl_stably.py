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
    idx: int  # line index in file (for stable de-dup)


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


def load_experiment_series(
    jsonl_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (critic_df, val_df) for one experiment file.

    critic_df columns: step, critic_rewards_mean
    val_df columns: step, val_overall_reward

    Notes:
    - `val` may be missing on many lines.
    - steps may repeat across different record types; we keep the last non-null per series.
    """

    critic_points: List[SeriesPoint] = []
    val_points: List[SeriesPoint] = []

    for idx, item in iter_jsonl(jsonl_path):
        step = item.get("step", None)
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
        df = pd.DataFrame([{ "step": p.step, value_col: p.value, "_idx": p.idx } for p in points])
        df = df.sort_values(["step", "_idx"], ascending=[True, True])
        # keep the last record for each step within this series
        df = df.groupby("step", as_index=False).tail(1)
        df = df.sort_values("step")
        return df[["step", value_col]].reset_index(drop=True)

    critic_df = to_df(critic_points, "critic_rewards_mean")
    val_df = to_df(val_points, "val_overall_reward")

    return critic_df, val_df


def discover_jsonl_files(log_dir: Path, pattern: str) -> List[Path]:
    files = sorted(log_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def apply_rolling_mean(df: pd.DataFrame, y_col: str, window: int) -> pd.DataFrame:
    if window is None or window <= 1 or df.empty:
        return df
    out = df.copy()
    out[y_col] = (
        out[y_col]
        .rolling(window=window, min_periods=max(1, window // 2), center=True)
        .mean()
    )
    return out


def load_runs_long_frames(
    jsonl_files: List[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple runs into long-form dataframes for seaborn errorbar.

    Returns:
      critic_long: columns [Step, Run, critic_rewards_mean]
      val_long:    columns [Step, Run, val_overall_reward]
    """

    critic_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    for run_id, path in enumerate(jsonl_files):
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
    """Aggregate long-form rows into per-step mean/sd.

    Returns columns: Step, mean, sd, n, lower, upper
    where lower/upper are mean +/- sd (sd=0 when undefined).
    """

    if long_df.empty:
        return pd.DataFrame(columns=["Step", "mean", "sd", "n", "lower", "upper"])

    g = long_df.groupby("Step", as_index=False)[y_col].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"std": "sd", "count": "n"})
    g["sd"] = g["sd"].fillna(0.0)
    g["lower"] = g["mean"] - g["sd"]
    g["upper"] = g["mean"] + g["sd"]
    g = g.sort_values("Step").reset_index(drop=True)
    return g[["Step", "mean", "sd", "n", "lower", "upper"]]


def smooth_agg_band(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Smooth aggregated mean and band for a softer-looking curve.

    We smooth mean/lower/upper separately (rolling mean). This keeps an error band
    while reducing jaggedness.
    """

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


def plot_critic(
    exp_to_critic: Dict[str, pd.DataFrame],
    color_map: Dict[str, Tuple[float, float, float]],
    save_path: Path,
    title: Optional[str],
    dpi: int,
    smooth_window: int,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(8, 5))

    any_plotted = False
    for exp_name, df in exp_to_critic.items():
        if df.empty:
            continue
        df = apply_rolling_mean(df, "critic_rewards_mean", smooth_window)
        sns.lineplot(
            data=df,
            x="step",
            y="critic_rewards_mean",
            ax=ax,
            color=color_map.get(exp_name, None),
            label=exp_name,
            linewidth=2,
            errorbar=None,
        )
        any_plotted = True

    ax.set_xlabel("Steps", fontsize=16)
    ax.set_ylabel("critic.reward.mean", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(title or "Critic Rewards Mean", fontsize=16)

    if any_plotted:
        ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_val(
    exp_to_val: Dict[str, pd.DataFrame],
    color_map: Dict[str, Tuple[float, float, float]],
    save_path: Path,
    title: Optional[str],
    dpi: int,
    smooth_window: int,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(8, 5))

    any_plotted = False
    for exp_name, df in exp_to_val.items():
        if df.empty:
            continue
        df = apply_rolling_mean(df, "val_overall_reward", smooth_window)
        sns.lineplot(
            data=df,
            x="step",
            y="val_overall_reward",
            ax=ax,
            color=color_map.get(exp_name, None),
            label=exp_name,
            linewidth=2,
            errorbar=None,
            marker="o",
            markersize=5,
        )
        any_plotted = True

    ax.set_xlabel("Steps", fontsize=16)
    ax.set_ylabel("val.reward", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(title or "Validation Overall Reward", fontsize=16)

    if any_plotted:
        ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dual_axis_aggregated(
    critic_long: pd.DataFrame,
    val_long: pd.DataFrame,
    save_path: Path,
    title: Optional[str],
    dpi: int,
    critic_smooth_window: int,
    val_smooth_window: int,
    ylim_mode: str,
    ylim_padding: float,
    color_critic: str,
    color_val: str,
) -> None:
    """Plot aggregated curves with sd band on a dual y-axis chart (demo-like)."""

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    # Colors are configurable via CLI; defaults are set to a high-contrast pair.

    fig, ax1 = plt.subplots(figsize=(8, 5))

    critic_agg: Optional[pd.DataFrame] = None
    val_agg: Optional[pd.DataFrame] = None

    if not critic_long.empty:
        critic_long = critic_long.sort_values("Step")
        critic_agg = aggregate_mean_sd(critic_long, "critic_rewards_mean")
        critic_agg = smooth_agg_band(critic_agg, critic_smooth_window)
        ax1.plot(
            critic_agg["Step"],
            critic_agg["mean"],
            color=color_critic,
            linewidth=2,
            label="critic.reward",
        )
        ax1.fill_between(
            critic_agg["Step"],
            critic_agg["lower"],
            critic_agg["upper"],
            color=color_critic,
            alpha=0.18,
            linewidth=0,
            label="critic sd",
        )

    ax1.set_xlabel("Steps", fontsize=16)
    ax1.set_ylabel("critic.reward", fontsize=16, color=color_critic)
    ax1.tick_params(axis="y", labelcolor=color_critic, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)

    ax2 = ax1.twinx()
    if not val_long.empty:
        val_long = val_long.sort_values("Step")
        val_agg = aggregate_mean_sd(val_long, "val_overall_reward")
        val_agg = smooth_agg_band(val_agg, val_smooth_window)
        ax2.plot(
            val_agg["Step"],
            val_agg["mean"],
            color=color_val,
            linewidth=2,
            marker="o",
            markersize=6,
            label="val.reward",
        )
        ax2.fill_between(
            val_agg["Step"],
            val_agg["lower"],
            val_agg["upper"],
            color=color_val,
            alpha=0.18,
            linewidth=0,
            label="val sd",
        )

    # Optional: make the two y-axes visually comparable by unifying limits or spans.
    if ylim_mode not in {"auto", "shared", "matched_span"}:
        raise ValueError(f"Unknown ylim_mode: {ylim_mode}")
    if ylim_padding < 0:
        raise ValueError("ylim_padding must be >= 0")

    if ylim_mode != "auto" and (critic_agg is not None) and (val_agg is not None):
        y1_min = float(critic_agg["lower"].min())
        y1_max = float(critic_agg["upper"].max())
        y2_min = float(val_agg["lower"].min())
        y2_max = float(val_agg["upper"].max())

        if ylim_mode == "shared":
            y_min = min(y1_min, y2_min)
            y_max = max(y1_max, y2_max)
            span = max(1e-12, y_max - y_min)
            pad = span * float(ylim_padding)
            ax1.set_ylim(y_min - pad, y_max + pad)
            ax2.set_ylim(y_min - pad, y_max + pad)
        elif ylim_mode == "matched_span":
            span1 = max(1e-12, y1_max - y1_min)
            span2 = max(1e-12, y2_max - y2_min)
            target_span = max(span1, span2)
            target_span = target_span * (1.0 + 2.0 * float(ylim_padding))

            c1 = 0.5 * (y1_min + y1_max)
            c2 = 0.5 * (y2_min + y2_max)
            ax1.set_ylim(c1 - 0.5 * target_span, c1 + 0.5 * target_span)
            ax2.set_ylim(c2 - 0.5 * target_span, c2 + 0.5 * target_span)

    ax2.set_ylabel("val.reward", fontsize=16, color=color_val)
    ax2.tick_params(axis="y", labelcolor=color_val, labelsize=12)
    ax2.grid(False)

    if title:
        ax1.set_title(title, fontsize=16)

    # Merge legends (demo style)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # drop sd band labels to keep legend clean
    filtered = [(l, lab) for l, lab in zip(lines_1 + lines_2, labels_1 + labels_2) if not lab.endswith(" sd")]
    if filtered:
        ax1.legend(
            [x[0] for x in filtered],
            [x[1] for x in filtered],
            loc="lower right",
            fontsize=11,
            framealpha=0.9,
        )
    if ax2.get_legend():
        ax2.get_legend().remove()

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot RL curves from train_logs/*.jsonl. Default aggregates multiple runs "
            "(e.g. v1-v4) into mean curve with sd band, in a demo-like dual-y-axis plot."
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
        help="Glob pattern for log files inside log_dir.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.cwd()),
        help="Output directory for images (default: current working directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rl_curves_nosoft",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dual",
        choices=["dual", "separate"],
        help="Plot mode: dual (demo-like twin y-axis) or separate (two figures per-run).",
    )
    parser.add_argument("--title", type=str, default=None, help="Title for dual plot.")
    parser.add_argument("--title_critic", type=str, default=None, help="Title for critic plot (separate mode).")
    parser.add_argument("--title_val", type=str, default=None, help="Title for val plot (separate mode).")
    parser.add_argument(
        "--critic_smooth_window",
        type=int,
        default=7,
        help="Rolling-mean window for critic curve smoothing (<=1 disables).",
    )
    parser.add_argument(
        "--val_smooth_window",
        type=int,
        default=1,
        help="Rolling-mean window for val curve smoothing (<=1 disables).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output file format. Use pdf/svg for crisp vector graphics.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI (mainly for png). For pdf/svg this usually has little effect.",
    )
    parser.add_argument(
        "--ylim_mode",
        type=str,
        default="shared",
        choices=["auto", "shared", "matched_span"],
        help=(
            "Dual mode only: auto=independent y-lims; shared=both axes share same numeric ylim; "
            "matched_span=both axes have same ylim span but different centers."
        ),
    )
    parser.add_argument(
        "--ylim_padding",
        type=float,
        default=0.05,
        help="Dual mode only: padding ratio applied to y-span (e.g., 0.05 adds 5%).",
    )
    parser.add_argument(
        "--color_critic",
        type=str,
        default="#4e79a7",
        help="Dual mode only: line color for critic curve (hex). Default: Okabe-Ito blue.",
    )
    parser.add_argument(
        "--color_val",
        type=str,
        default="#665191",
        help="Dual mode only: line color for val curve (hex). Default: Okabe-Ito vermillion.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = discover_jsonl_files(log_dir, args.pattern)
    if not jsonl_files:
        raise SystemExit(f"No jsonl files found in {log_dir} with pattern {args.pattern}")

    if args.mode == "dual":
        critic_long, val_long = load_runs_long_frames(jsonl_files)
        dual_path = out_dir / f"{args.prefix}_dual.{args.format}"
        plot_dual_axis_aggregated(
            critic_long,
            val_long,
            dual_path,
            title=args.title,
            dpi=args.dpi,
            critic_smooth_window=args.critic_smooth_window,
            val_smooth_window=args.val_smooth_window,
            ylim_mode=args.ylim_mode,
            ylim_padding=args.ylim_padding,
            color_critic=args.color_critic,
            color_val=args.color_val,
        )
        print(f"Runs: {len(jsonl_files)}")
        print(f"critic_samples={len(critic_long)}, val_samples={len(val_long)}")
        print(f"Saved: {dual_path}")
    else:
        # separate mode: keep per-run curves, consistent colors within the run set
        exp_names = [p.stem for p in jsonl_files]
        palette = sns.color_palette("tab10", n_colors=max(3, len(exp_names)))
        color_map = {name: palette[i % len(palette)] for i, name in enumerate(exp_names)}

        exp_to_critic: Dict[str, pd.DataFrame] = {}
        exp_to_val: Dict[str, pd.DataFrame] = {}
        for p in jsonl_files:
            exp = p.stem
            critic_df, val_df = load_experiment_series(p)
            exp_to_critic[exp] = critic_df
            exp_to_val[exp] = val_df

        critic_path = out_dir / f"{args.prefix}_critic.{args.format}"
        val_path = out_dir / f"{args.prefix}_val.{args.format}"
        plot_critic(
            exp_to_critic,
            color_map,
            critic_path,
            args.title_critic,
            args.dpi,
            smooth_window=args.critic_smooth_window,
        )
        plot_val(
            exp_to_val,
            color_map,
            val_path,
            args.title_val,
            args.dpi,
            smooth_window=args.val_smooth_window,
        )

        for exp in exp_names:
            print(f"{exp}: critic_points={len(exp_to_critic[exp])}, val_points={len(exp_to_val[exp])}")
        print(f"Saved: {critic_path}")
        print(f"Saved: {val_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
