# ADAS Integration Plan

## 整体架构

ADAS 流程分为三个独立阶段，各自一行命令：

```
[Step 1] main_adas.py        → 推理 + 打分 → adas_scores.csv
[Step 2] run_adas_filter.sh  → 合并 + 统计 + 筛选 → token_filter.txt
[Step 3] main.py             → 训练（使用 token_filter_file）
```

各阶段完全解耦，可独立运行、重试、在不同机器上执行。

---

## Step 1: ADAS 推理（`verl/trainer/main_adas.py`）

复用 verl 的 Ray + vLLM rollout + reward function，
只初始化推理组件（无 critic / ref policy / optimizer），显存全给 vLLM。

```bash
python -m verl.trainer.main_adas \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=null \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 \
    worker.rollout.n=32 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8
    # trainer.load_checkpoint_path=${load_path}  # 多轮 ADAS 时指定上一轮 checkpoint
```

> 完整示例见 `scripts/adas/run_adas_infer.sh`

**自动 force 的参数**（`main_adas.py` 内部强制覆盖）：
- `algorithm.adv_estimator = "grpo"`
- `algorithm.disable_kl = True`
- `data.token_filter_file = None`（推理时遍历全量数据）
- `data.shuffle = False`（确保顺序确定性）

**输出**：`checkpoints/adas/${EXP_NAME}/adas_scores.csv`

```csv
token,pdms,pdms_scaled
a99a8f7c...,0.8818,0.9712
a99a8f7c...,0.7523,0.8901
bd2a4d57...,0.6134,0.7245
...
```

- 每行 = 一条 rollout 的打分（一个 token 有 n 行）
- 列名 `token, pdms, pdms_scaled` 命中 `compute_stats.py` 的优先路径
- 与手工 scorer CSV 格式兼容，可混放在同一目录下

**同目录自动产生**（reward function 的 log）：
- `generations_<timestamp>.jsonl`

---

## Step 2: ADAS 筛选（`scripts/adas/run_adas_filter.sh`）

```bash
cd scripts/adas

INFER_FOLDER="../../checkpoints/adas/${EXP_NAME}"

python pipeline.py \
    --infer_folder "$INFER_FOLDER" \
    --csv_mode prefill \
    -p 0.2 \
    --conf 0.1
```

Pipeline 内部流程（3 步）：

```
scorer CSV(s)
    ↓ merge_scorer_csv    → generations_full_<mode>.parquet
    ↓ compute_stats       → generations_full_<mode>.csv
    ↓ filter_dynamic      → generations_full_<mode>_filtered_<p>.{csv,txt,jsonl}
```

- `--csv_mode standard`（默认）：排除文件名含 "prefill" 的 scorer CSV
- `--csv_mode prefill`：只选取文件名含 "prefill" 的 scorer CSV

输出文件通过 `_prefill` 后缀区分，标准模式无后缀。

**Normalization 透明**：Pipeline 全程只传递 denorm 原始轨迹（`coarse_center`），
不做任何 normalize/denormalize 变换。所有 norm 逻辑封装在 dataloader 的
`TrajectoryNormalizer.format_prefill_text()` 中，对 pipeline 完全透明。

**prefill 模式**：`filter_dynamic` 直接从 `prefill_meta.jsonl` + `centers_jsonl`
加载 denorm center（仅加载 filtered rows 需要的），写入 `.jsonl` filter spec。
不需要 enrich 中间步骤。

`-p` 和 `--conf` 参数可手动调整后重跑，无需重新推理。

---

## Step 3: 训练

```bash
# 单文件：Prefill 模式（使用 .jsonl，包含 coarse_center）
python -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${FULL_DATA}@train \
    data.val_files=${FULL_DATA}@test \
    data.format_prompt=null \
    data.max_response_length=3072 \
    data.token_filter_file=${INFER_FOLDER}/generations_full_prefill_filtered_0.2.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    worker.reward.reward_function=${REWARD_FN}:compute_score_fast \
    trainer.experiment_name=${EXP_NAME}_train \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=8

# 单文件：No-prefill 模式（使用 .txt）
#   data.token_filter_file=${INFER_FOLDER}/generations_full_filtered_0.2.txt

# 多文件混合：Prefill + No-prefill（dataloader 自动检测每个 entry 是否有 coarse_center）
#   data.token_filter_file=[${INFER_FOLDER}/generations_full_prefill_filtered_0.2.jsonl,${INFER_FOLDER}/generations_full_filtered_0.15.txt]
```

**多文件混合说明**：
- `token_filter_file` 接受 list（YAML list 语法：`[file1,file2,...]`）
- Dataloader 逐个读取，自动检测每个 entry 是否有 `coarse_center` 字段
- 有 `coarse_center` → prefill sample（normalize → format → 注入 prompt 末尾）
- 无 `coarse_center` → base sample（不注入）
- 支持任意组合：纯 prefill、纯 base、或混合
- Dataloader 的 shuffle 在混合后的完整 sample list 上执行

**文件格式**：
- `.jsonl`（prefill）：每行 `{"token": "...", "coarse_center": [[x,y,z], ...]}`
- `.jsonl`（no prefill）：每行 `{"token": "..."}`（无 `coarse_center` 字段）
- `.txt`（legacy）：每行一个 token（等价于 no prefill）

---

## 多轮 ADAS

```
Round 1: main_adas(base_model)                           → filter → train
Round 2: main_adas(load_checkpoint_path=round1_ckpt)     → filter → train
Round 3: ...
```

- `worker.actor.model.model_path` 始终指向 SFT base model（用于加载 tokenizer）
- `trainer.load_checkpoint_path` 指向上一轮训练产出的 checkpoint（如 `global_step_88`）
- 通过 `trainer.experiment_name` 区分不同轮次的输出目录

---

## 输出目录结构

```
${INFER_FOLDER}/
├── *_result_*.csv                                ← 原始 scorer CSV
├── *.prefill_meta.jsonl                          ← prefill 元数据（含 centers 路径）
├── generations_full.parquet                            ← Step 1 (merge, standard mode)
├── generations_full.csv                                ← Step 2 (group stats)
├── generations_full_filtered_<p>.csv                   ← Step 3 (filtered stats)
├── generations_full_filtered_<p>.txt                   ← Step 3 (token list)
├── generations_full_filtered_<p>.jsonl                 ← Step 3 (filter spec)
├── generations_full_prefill.parquet                    ← Step 1 (merge, prefill mode)
├── generations_full_prefill.csv                        ← Step 2 (group stats)
├── generations_full_prefill_filtered_<p>.csv           ← Step 3 (filtered stats)
├── generations_full_prefill_filtered_<p>.txt           ← Step 3 (token list, legacy)
└── generations_full_prefill_filtered_<p>.jsonl         ← Step 3 (filter spec with coarse_center)
```

---

## 兼容性

| 场景 | 兼容 |
|------|------|
| 单个 prefill jsonl | `token_filter_file=path/to/prefill.jsonl` |
| 单个 no-prefill txt/jsonl | `token_filter_file=path/to/base.txt` |
| 多文件混合 | `token_filter_file=[prefill.jsonl,base.jsonl]` |
| Dataloader 自动检测 | 每个 entry 根据 `coarse_center` 字段自动判断 prefill/base |
| Pipeline 不生成 mix | Pipeline 只生成 standard 和 prefill 各自的输出，用户自己组合 |

---

## 已完成的 Merge 记录

> 私有路径说明：dev 仓库中的硬编码路径（`/mnt/data/ccy/...`）无需清理。
> 开源版本有独立的目录隔离流程（`scripts/copy_to_opensource.sh`），
> 本仓库以开发为主，开源为此仓库让步。

### 从 opensource-adas 合并到 dev-local 的文件

**新增文件（from `a2cb7ab`）：**

| 文件 | 说明 |
|------|------|
| `verl/trainer/main_adas.py` | ADAS inference 入口 |
| `scripts/adas/run_adas_infer.sh` | ADAS 推理示例 |
| `scripts/adas/run_adas_filter.sh` | ADAS 筛选示例 |
| `scripts/adas/ADAS_INTEGRATION_PLAN.md` | 本文档 |
| `train_scripts/train_adas_new.sh` | ADAS 训练示例 |

**Pipeline 脚本（from `c7ad6ab`，初始版本，非 `a2cb7ab` 简化版）：**

| 文件 | 说明 |
|------|------|
| `scripts/adas/pipeline.py` | 编排：merge → stats → filter |
| `scripts/adas/compute_stats.py` | 按 token 分组统计 |
| `scripts/adas/merge_scorer_csv.py` | 合并 scorer CSV → Parquet |
| `scripts/adas/filter_dynamic.py` | 三阶段筛选 + 输出 .txt/.jsonl（直接从 prefill_meta 加载 denorm center，不依赖 enrich） |
| `scripts/adas/enrich_prefill.py` | 保留但 pipeline 不再调用（独立工具） |
| `scripts/adas/run_adas.sh` | 原始筛选示例 |

**整文件 checkout（from `a2cb7ab`）：**

| 文件 | 说明 |
|------|------|
| `verl/trainer/config.py` | DataConfig 新增 `token_filter_file` |
| `verl/utils/dataset.py` | RLHFDataset 支持 token_filter_file 过滤 |
| `verl/trainer/data_loader.py` | create_dataloader 传入 token_filter_file |

**手动编辑（在 dev-local 版本上 patch）：**

| 文件 | 改动 |
|------|------|
| `examples/config_vla.yaml` | 加 `token_filter_file: null` 一行 |
| `navsim_reward_text.py` | httpx 替换 requests + log dir → `checkpoints/adas/` |
| `debug/analysis/filter_dynamic_to_csv.py` | 追加 txt 输出（debug 版本也加上） |

**未合并（仅限 opensource-adas 分支）：**

| 文件 | 原因 |
|------|------|
| `helper.py` syn stats 移除 | 仅限开源版本 |
| `navsim_reward_text_rpc.py` denormalize 签名改动 | 仅限开源版本 |
| Pipeline 脚本的 `a2cb7ab` 简化 | 仅限开源版本 |
| `enrich_prefill.py` 删除 | 仅限开源版本 |
| `run_adas.sh` 删除 | 仅限开源版本 |
| `scripts/copy_to_opensource.sh` | 舍弃 |

---

## Git Commit 记录

合并操作在 `dev-local` 分支上产生了以下 commit：

| Commit | 说明 |
|--------|------|
| `8099c59` | [feat] ADAS: inference integration + token filter dataloader — 主体合并：`main_adas.py`、`config.py`、`dataset.py`、`data_loader.py`、`config_vla.yaml`、`navsim_reward_text.py`（httpx + log dir）、示例脚本、本文档 |
| `5bdc76b` | [fix] add missing scripts/adas/ pipeline scripts from c7ad6ab — 补充 `run_adas_filter.sh` 依赖的 pipeline 脚本（`pipeline.py`、`compute_stats.py`、`merge_scorer_csv.py`、`filter_dynamic.py`、`enrich_prefill.py`、`run_adas.sh`） |
| `70abe50` | [docs] update ADAS plan: fix merge record + private path policy — 修正合并记录表格，添加私有路径说明 |
