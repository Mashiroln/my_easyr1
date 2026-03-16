#!/usr/bin/env python3
"""
ADAS数据增强自动化流程
整合以下步骤:
1. run_navsim_data 准备数据 (调用外部脚本)
2. extract_parquet_from_csv 合并CSV到parquet
3. stats_parquet 统计group级别分布
4. filter_dynamic_to_csv 筛选动态样本 (可调整超参控制样本量3-6k)
5. construct_filtered_dataset 构造新数据集
"""
from __future__ import annotations

import os
import subprocess
import argparse
from extract_parquet_from_csv import main as extract_parquet
from stats_parquet import main as stats_parquet
from filter_dynamic_to_csv import main as filter_dynamic
from construct_filtered_dataset import main as construct_dataset
from enrich_prefill_parquet import enrich_parquet as enrich_prefill_parquet
from enrich_prefill_parquet import _find_single_file as _find_single_file_for_enrich

def run_pipeline(
    exp_name: str | None,
    input_json_path: str,
    source_dataset_path: str,
    output_dataset_name: str,
    # 筛选超参
    diversity_threshold: float = 0.1,
    n_rollout: int = 8,
    group_size: int = 16,
    std_threshold: float = 0.01,
    confidence_threshold: float = 0.1,
    # 目标样本量范围
    target_min: int = 3000,
    target_max: int = 6000,
    # 路径配置
    infer_output_root: str = "/mnt/data/ccy/VLA_train/parallel_infer/output",
    infer_folder: str | None = None,
    data_output_root: str = "/mnt/data/ccy/EasyR1/data",
    csv_mode: str = "all",
    include_glob: str | None = None,
    exclude_glob: str | None = None,
    include_bak: bool = False,
    group_size_mode: str | None = None,
    enrich_prefill: bool | None = None,
    prefill_meta_jsonl: str | None = None,
    # 控制选项
    skip_navsim_data: bool = False,
    auto_tune: bool = True,
):
    """
    运行完整的ADAS数据增强流程

    Args:
        exp_name: 实验名称
        input_json_path: navsim数据准备的输入json路径
        source_dataset_path: 源数据集路径
        output_dataset_name: 输出数据集名称
        diversity_threshold: 多样性阈值 (越小要求混合程度越高)
        n_rollout: rollout数量
        group_size: group大小
        std_threshold: 标准差阈值
        confidence_threshold: 置信度阈值
        target_min/max: 目标样本量范围
        auto_tune: 是否自动调整超参以达到目标样本量
    """
    if infer_folder is not None:
        folder_path = infer_folder
        if exp_name is None or exp_name.strip() == "":
            exp_name = os.path.basename(os.path.normpath(infer_folder))
    else:
        if exp_name is None or exp_name.strip() == "":
            raise ValueError("exp_name 不能为空（除非提供 infer_folder）")
        folder_path = os.path.join(infer_output_root, exp_name)

    if group_size_mode is None:
        # prefill 输出的 rollout 数往往按 token 不一致，推荐用统计出来的 group_size
        group_size_mode = "from_csv" if csv_mode == "prefill" else "fixed"

    if enrich_prefill is None:
        enrich_prefill = (csv_mode == "prefill")

    print("=" * 60)
    print(f"ADAS数据增强流程 - {exp_name}")
    print("=" * 60)

    # Step 1: run_navsim_data (可选跳过)
    if not skip_navsim_data:
        print("\n[Step 1] 运行 navsim_action_full_easyr1.py 准备数据...")
        local_dir = os.path.join("./data", exp_name)
        cmd = [
            "python", "scripts/navsim_action_full_easyr1.py",
            "--input_json_path", input_json_path,
            "--local_dir", local_dir
        ]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("\n[Step 1] 跳过 navsim_data 准备步骤")

    # Step 2: extract_parquet_from_csv
    print("\n[Step 2] 合并CSV文件到Parquet...")
    parquet_path = extract_parquet(
        folder_path,
        output_parquet=None,
        csv_mode=csv_mode,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
        include_bak=include_bak,
    )

    # Step 2.5: enrich prefill parquet
    if csv_mode == "prefill" and enrich_prefill:
        print("\n[Step 2.5] 为 Prefill parquet 增量补齐 norm coarse traj / prefill_text...")
        meta_jsonl = prefill_meta_jsonl
        if meta_jsonl is None:
            meta_jsonl = _find_single_file_for_enrich(folder_path, ".prefill_meta.jsonl")
        enriched_parquet = os.path.splitext(parquet_path)[0] + "_enriched.parquet"
        parquet_path = enrich_prefill_parquet(
            input_parquet=parquet_path,
            meta_jsonl=meta_jsonl,
            output_parquet=enriched_parquet,
            max_rows=None,
        )

    # Step 3: stats_parquet
    print("\n[Step 3] 统计group级别分布...")
    stats_csv = stats_parquet(parquet_path)

    # Step 4: filter_dynamic_to_csv (带自动调参)
    print("\n[Step 4] 筛选动态样本...")

    if auto_tune:
        # 自动调整diversity_threshold以达到目标样本量
        thresholds_to_try = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        best_threshold = diversity_threshold
        best_count = 0

        for thresh in thresholds_to_try:
            _, _, count = filter_dynamic(
                stats_csv,
                None,
                thresh,
                n_rollout,
                group_size,
                group_size_mode,
                std_threshold,
                confidence_threshold,
            )
            print(f"  diversity_threshold={thresh}: {count} samples")

            if target_min <= count <= target_max:
                best_threshold = thresh
                best_count = count
                print(f"  ✓ 找到合适阈值: {thresh}, 样本量: {count}")
                break
            elif count < target_min and count > best_count:
                best_threshold = thresh
                best_count = count

        diversity_threshold = best_threshold
        print(f"\n使用 diversity_threshold={diversity_threshold}")

    output_csv, txt_path, final_count = filter_dynamic(
        stats_csv,
        None,
        diversity_threshold,
        n_rollout,
        group_size,
        group_size_mode,
        std_threshold,
        confidence_threshold,
    )

    print(f"\n筛选完成，最终样本量: {final_count}")

    if final_count < target_min:
        print(f"⚠️ 警告: 样本量 {final_count} 低于目标最小值 {target_min}")
    elif final_count > target_max:
        print(f"⚠️ 警告: 样本量 {final_count} 高于目标最大值 {target_max}")

    # Step 5: construct_filtered_dataset
    print("\n[Step 5] 构造新数据集...")
    output_path = construct_dataset(
        source_dataset_path,
        txt_path,
        output_dataset_name,
        data_output_root,
        enriched_parquet=parquet_path,
    )

    print("\n" + "=" * 60)
    print("流程完成!")
    print(f"  - Parquet: {parquet_path}")
    print(f"  - Stats CSV: {stats_csv}")
    print(f"  - Filtered CSV: {output_csv}")
    print(f"  - Token list: {txt_path}")
    print(f"  - Output dataset: {output_path}")
    print(f"  - Final sample count: {final_count}")
    print("=" * 60)

    return {
        "parquet_path": parquet_path,
        "stats_csv": stats_csv,
        "filtered_csv": output_csv,
        "txt_path": txt_path,
        "output_dataset": output_path,
        "sample_count": final_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS数据增强自动化流程")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称（infer_folder 未提供时必填）")
    parser.add_argument("--infer_folder", type=str, default=None, help="直接指定推理输出目录（会覆盖 infer_output_root/exp_name 拼接）")
    parser.add_argument("--input_json_path", type=str, default="", help="navsim输入json路径")
    parser.add_argument("--source_dataset_path", type=str, required=True, help="源数据集路径")
    parser.add_argument("--output_dataset_name", type=str, required=True, help="输出数据集名称")

    # 筛选超参
    parser.add_argument("--diversity_threshold", type=float, default=0.1)
    parser.add_argument("--n_rollout", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)

    # 目标样本量
    parser.add_argument("--target_min", type=int, default=3000)
    parser.add_argument("--target_max", type=int, default=6000)

    # 路径配置
    parser.add_argument("--infer_output_root", type=str,
                        default="/mnt/data/ccy/VLA_train/parallel_infer/output")
    parser.add_argument("--data_output_root", type=str,
                        default="/mnt/data/ccy/EasyR1/data")

    parser.add_argument(
        "--csv_mode",
        type=str,
        default="all",
        choices=["all", "prefill", "no_prefill"],
        help="合并哪些 scorer CSV（按文件名是否包含 'prefill' 判断）",
    )
    parser.add_argument("--include_glob", type=str, default=None, help="仅包含匹配该 glob 的 CSV basename")
    parser.add_argument("--exclude_glob", type=str, default=None, help="排除匹配该 glob 的 CSV basename")
    parser.add_argument("--include_bak", action="store_true", help="是否包含 *_bak.csv（默认忽略）")
    parser.add_argument(
        "--group_size_mode",
        type=str,
        default=None,
        choices=["fixed", "from_csv"],
        help="fixed: 使用 --group_size 常量；from_csv: 使用统计CSV的 group_size 列（prefill 推荐）",
    )

    parser.add_argument(
        "--enrich_prefill",
        action="store_true",
        default=None,
        help="在 csv_mode=prefill 时，把 norm 后 coarse traj 与 prefill_text 写入 *_enriched.parquet（默认：prefill 模式自动开启）",
    )
    parser.add_argument(
        "--no_enrich_prefill",
        action="store_true",
        help="禁用 enrich_prefill（即使 csv_mode=prefill）",
    )
    parser.add_argument(
        "--prefill_meta_jsonl",
        type=str,
        default=None,
        help="显式指定 <prefill_output>.prefill_meta.jsonl；默认自动在 infer_folder 下找最大的一份",
    )

    # 控制选项
    parser.add_argument("--skip_navsim_data", action="store_true", help="跳过navsim数据准备步骤")
    parser.add_argument("--no_auto_tune", action="store_true", help="禁用自动调参")

    args = parser.parse_args()

    if args.infer_folder is None and (args.exp_name is None or args.exp_name.strip() == ""):
        parser.error("必须提供 --exp_name 或 --infer_folder")

    run_pipeline(
        exp_name=args.exp_name,
        input_json_path=args.input_json_path,
        source_dataset_path=args.source_dataset_path,
        output_dataset_name=args.output_dataset_name,
        diversity_threshold=args.diversity_threshold,
        n_rollout=args.n_rollout,
        group_size=args.group_size,
        std_threshold=args.std_threshold,
        confidence_threshold=args.confidence_threshold,
        target_min=args.target_min,
        target_max=args.target_max,
        infer_output_root=args.infer_output_root,
        data_output_root=args.data_output_root,
        infer_folder=args.infer_folder,
        csv_mode=args.csv_mode,
        include_glob=args.include_glob,
        exclude_glob=args.exclude_glob,
        include_bak=args.include_bak,
        group_size_mode=args.group_size_mode,
        enrich_prefill=(False if args.no_enrich_prefill else args.enrich_prefill),
        prefill_meta_jsonl=args.prefill_meta_jsonl,
        skip_navsim_data=args.skip_navsim_data,
        auto_tune=not args.no_auto_tune,
    )
