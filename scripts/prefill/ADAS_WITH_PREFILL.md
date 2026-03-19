# ADAS with Prefill: Coarse-Center Guided Reinforcement Learning

> 原始 ADAS 的增量插件。所有设计决策以"与标准 ADAS 数据流完全解耦"为第一原则。

---

## Motivation

Coarse2Refine 数据格式要求模型先输出 coarse trajectory，再基于场景分析输出 refined trajectory：

```json
{
  "coarse_trajectory": "<answer>[PT, (x1,y1,h1), ...]</answer>",
  "explanation": "<answer>场景分析 + 修正理由</answer>",
  "future_trajectory": "<answer>[PT, (x2,y2,h2), ...]</answer>"
}
```

纯 on-policy RL（原始 ADAS）中，模型自己生成 coarse trajectory 再 refine。但实验表明：IL 后模型的 coarse 策略熵崩溃，大多数场景只产生一种 coarse，导致 refine 的探索空间极窄。

**核心想法**：在 prompt 末尾注入一个 off-policy 的 coarse trajectory（称为 prefill），强迫模型从不同的 coarse 起点做 refine。这些 coarse 来自同场景的聚类中心（medoid），代表该场景下的典型多样化起点。

---

## 两条数据流：独立但有上游依赖

```
标准 ADAS 数据流
  推理(标准, 32×103k) → 打分 → merge → stats → filter → .txt ──→ 训练
       │                                                           ↑
       │ 全量推理结果                                               │ token_filter_file=[standard.txt, prefill.jsonl]
       ↓                                                           │
  extract → analyze → ★ centers.jsonl                              │
                          │                                        │
                          ↓                                        │
Prefill 数据流            │                                        │
  推理(prefill, 从 centers 选稀有 center) → 打分 → merge → stats → filter → .jsonl ─┘
```

Prefill 数据流依赖标准 ADAS 的全量推理结果来做聚类分析（提取 coarse trajectory → 聚类 → 得到 centers.jsonl）。这是唯一的上游依赖。从 prefill 推理开始，两条流完全独立：
- **不同的推理脚本**（标准用标准推理，prefill 从 centers 选稀有 center 注入 prompt）
- **不同的 scorer CSV**（文件名含/不含 `prefill`）
- **不同的中间产物**（标准无后缀，prefill 带 `_prefill` 后缀）
- **不同的最终输出**（`.txt` vs `.jsonl`）

唯一的下游交汇点是训练时的 `token_filter_file` list，由用户手动组合。

---

## Prefill 数据流详解

### 阶段 1: 聚类分析（外部，一次性）

基于标准 ADAS 推理结果，提取 coarse trajectory 并聚类，找到每个场景的多样化 coarse 中心。

```
标准 ADAS 推理 JSONL
    ↓ extract_coarse_refine_traj.py
coarse_refine.traj.bin (性能优化的二进制中间格式，128 进程并行)
    ↓ analyze_coarse_intragroup.py
★ coarse_intragroup_centers.jsonl (每行: {token, k, cluster_sizes, coarse_centers})
```

`coarse_centers` 是 denorm 8x3 轨迹列表（medoid），代表该 token 下的 K 个典型 coarse 起点。

这些脚本在 `parallel_infer/` 中，不需要改动。`.bin` 是性能关键的内部格式，最终产物 `centers.jsonl` 是可读 JSONL。

### 阶段 2: Prefill 推理 + 打分（外部）

从 `centers.jsonl` 中选择稀有 center，注入 prompt 做推理，再打分。

```
★ centers.jsonl + 稀有度选择逻辑
    ↓ parallel_infer (prefill 模式)
prefill 推理 JSONL + ★ prefill_meta.jsonl
    ↓ aggregate_traj.py (透传 prefill_meta_id)
prefill_agg.pkl
    ↓ send_score_request.py (透传 prefill_meta_id)
prefill scorer CSV (含 prefill_meta_id, prefill_centers_row, prefill_center_rank 列)
```

`prefill_meta.jsonl` 记录每个 prefill 样本的溯源信息：
```json
{
  "prefill_meta_id": 257,
  "id": "016b36e1eff55300",
  "centers_jsonl": "/path/to/coarse_intragroup_centers.jsonl",
  "centers_row": 1,
  "center_rank": 1,
  "cluster_size": 14,
  "stats_path": "/path/to/trajectory_stats.json"
}
```

### 阶段 3: 筛选（EasyR1，`scripts/adas/`）

与标准 ADAS 使用完全相同的 pipeline，只是 `--csv_mode prefill`：

```bash
cd scripts/adas
python pipeline.py \
    --infer_folder /path/to/output_dir \
    --csv_mode prefill \
    -p 0.1 --conf 0.1 \
    --group_size_mode from_csv
```

内部三步（与标准 ADAS 完全相同的代码路径）：

| 步骤 | 输入 → 输出 | prefill 特殊行为 |
|------|-------------|-----------------|
| merge | scorer CSV → `generations_full_prefill.parquet` | `csv_mode=prefill` 只选文件名含 "prefill" 的 CSV |
| stats | parquet → `generations_full_prefill.csv` | 按 `(token, prefill_meta_id)` 分组（自动检测列） |
| filter | stats CSV → `.csv` + `.txt` + `.jsonl` | 从 `prefill_meta.jsonl` 加载 denorm center 写入 JSONL |

关键：`--group_size_mode from_csv`，因为 prefill 的每个 (token, prefill_meta_id) 组的 rollout 数量不固定（取决于 cluster_size_threshold）。

### 阶段 4: 训练（EasyR1）

```bash
# 纯 prefill 训练
data.token_filter_file=path/to/generations_full_prefill_filtered_0.1.jsonl

# 混合训练（推荐）
data.token_filter_file=[path/to/standard.txt,path/to/prefill.jsonl]
```

Dataloader 逐文件读取，自动检测每个 entry：
- 有 `coarse_center` 字段 → prefill sample → normalize + format → 注入 prompt 末尾
- 无 `coarse_center` 字段 → base sample → 不注入

---

## 文件命名解耦

同一 `infer_folder` 下，标准 ADAS 和 prefill 的所有中间产物通过文件名自动区分。标准 ADAS 无后缀，prefill 带 `_prefill` 后缀：

```
${INFER_FOLDER}/
├── *_result_localhost_rpc.csv                          ← 标准 scorer CSV
├── *_prefill_result_localhost_rpc.csv                  ← prefill scorer CSV
├── *.prefill_meta.jsonl                                ← prefill 元数据
│
├── generations_full.parquet                            ← merge (标准，默认)
├── generations_full.csv                                ← stats
├── generations_full_filtered_<p>.txt                   ← filter → 训练用
│
├── generations_full_prefill.parquet                    ← merge (prefill)
├── generations_full_prefill.csv                        ← stats
├── generations_full_prefill_filtered_<p>.csv           ← filter (stats)
├── generations_full_prefill_filtered_<p>.txt           ← filter (token list, legacy)
└── generations_full_prefill_filtered_<p>.jsonl         ← filter → 训练用 (含 coarse_center)
```

`merge_scorer_csv.py` 的 `_is_prefill_csv()` 通过文件名是否含 `prefill` 来分流。
标准模式输出无后缀，prefill 模式输出带 `_prefill` 后缀。两者绝不互相覆盖。

---

## Norm 透明原则

ADAS pipeline（merge → stats → filter）全程只传递 denorm 原始轨迹，不做任何 normalize/denormalize。

| 阶段 | 坐标形式 | 说明 |
|------|----------|------|
| centers.jsonl | denorm | medoid 在物理空间选取 |
| prefill_meta.jsonl | 索引 | 指向 centers.jsonl 的行号+rank |
| filter_spec.jsonl (`coarse_center`) | denorm | 从 centers.jsonl 直取，自包含 |
| Dataloader 内部 | denorm → norm → format | `TrajectoryNormalizer.format_prefill_text()` |
| vLLM prompt | norm (formatted) | `[PT, (+0.12, ...), ...]` |

所有 norm 逻辑封装在 `verl/utils/trajectory_normalizer.py` 的 `TrajectoryNormalizer` 中。
它支持双 stats dispatch：普通 token 用 `NAVSIM_STAT_PATH`，合成 token (`-00\d$` 后缀) 用 `NAVSIM_STAT_PATH_SYN`。

---

## Filter Spec JSONL 格式（自包含）

```jsonl
{"token": "abc123def456"}
{"token": "abc123def456", "coarse_center": [[0.52,1.21,0.01],[1.03,2.34,0.02],...], "prefill_meta_id": 12345}
{"token": "abc123def456", "coarse_center": [[-0.31,0.82,-0.02],[0.12,1.45,0.03],...], "prefill_meta_id": 12346}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `token` | 是 | 场景 token |
| `coarse_center` | 否 | 8x3 denorm 轨迹。有 = prefill sample |
| `prefill_meta_id` | 否 | debug 溯源用，训练不依赖 |

JSONL 完全自包含，不需要索引任何 meta 文件。`coarse_center` 是完整的 denorm 轨迹数据。

---

## Dataloader Prefill 注入机制

`verl/utils/dataset.py` 中的 `RLHFDataset`:

1. `_load_filter_spec(paths)` 读取 filter file（支持 `str | list[str]`，自动检测 JSONL/TXT）
2. 有 prefill entry 时，构建 `_sample_entries: list[(dataset_idx, coarse_center_or_None)]`
3. `__getitem__` 中，若 `coarse_center is not None`：
   - `TrajectoryNormalizer.format_prefill_text(center_denorm, token=token)` → denorm → norm → format
   - 输出形如 `{\n  "coarse_trajectory": "<answer>[PT, (+0.12, ...), ...]</answer>",`
   - tokenize 后 append 到 `raw_prompt_ids` 末尾
   - 标记 `example["is_prefill"] = True`

这样 vLLM 看到的 prompt 末尾已经有了 coarse trajectory 的开头，模型从这里续写 refine。

---

## 代码索引

### ADAS 筛选（本仓库，标准和 prefill 共用）

```
scripts/adas/
├── pipeline.py              # 一键 pipeline（merge→stats→filter + auto-tune）
├── merge_scorer_csv.py      # CSV → Parquet（csv_mode 控制 standard/prefill 分流）
├── compute_stats.py         # Parquet → 组级统计（自动检测 prefill_meta_id 列）
├── filter_dynamic.py        # 三阶段过滤 → .txt + .jsonl（prefill 时从 meta 加载 denorm center）
├── run_adas_filter.sh       # 标准 ADAS 筛选示例
└── run_adas_filter_prefill.sh  # prefill 筛选示例
```

### Prefill 专用（本仓库）

```
verl/utils/trajectory_normalizer.py   # 共享 normalizer（reward + prefill injection）
verl/utils/dataset.py                 # RLHFDataset: _load_filter_spec + prefill injection
```

### 训练框架集成

```
verl/trainer/config.py       # DataConfig.token_filter_file: str | list[str]
verl/trainer/ray_trainer.py  # token 提取, sample_id uid, metrics filter
verl/workers/reward/function.py  # prefill metadata 透传
```

### 外部（parallel_infer，不需要改动）

```
/mnt/data/ccy/VLA_train/parallel_infer/
├── scorer/aggregate_traj.py              # 透传 prefill_meta_id
├── scorer/send_score_request.py          # 透传 prefill_meta_id
└── output/<exp>/
    ├── coarse_refine.traj.bin/
    │   └── coarse_intragroup_centers.jsonl  # 聚类中心（denorm 8x3）
    └── *.prefill_meta.jsonl                 # prefill 溯源元数据
```
