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
    -p 0.2 \
    --conf 0.1
```

Pipeline 内部流程：

```
adas_scores.csv
    ↓ merge_scorer_csv    → generations_full.parquet
    ↓ compute_stats       → group_stats.csv
    ↓ filter_dynamic      → group_stats_filtered_<p>.txt
```

**输出**：`checkpoints/adas/${EXP_NAME}/group_stats_filtered_<p>.txt`

`-p` 和 `--conf` 参数可手动调整后重跑，无需重新推理。

---

## Step 3: 训练

```bash
python -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${FULL_DATA}@train \
    data.val_files=${FULL_DATA}@test \
    data.format_prompt=null \
    data.max_response_length=3072 \
    data.token_filter_file=checkpoints/adas/${EXP_NAME}/group_stats_filtered_0.2.txt \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    worker.reward.reward_function=${REWARD_FN}:compute_score_fast \
    trainer.experiment_name=${EXP_NAME}_train \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=8
```

> 完整示例见 `train_scripts/train_adas_new.sh`

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
checkpoints/adas/${EXP_NAME}/
├── generations_<timestamp>.jsonl     ← reward function 自动 log
├── adas_scores.csv                   ← main_adas.py 输出
├── generations_full.parquet          ← pipeline Step 1 (merge)
├── generations_full.csv              ← pipeline Step 2 (group stats)
└── group_stats_filtered_<p>.txt      ← pipeline Step 3 (token filter)
```

---

## 兼容性

| 场景 | 兼容 |
|------|------|
| main_adas CSV → run_adas_filter.sh | 直接兼容，`merge_scorer_csv` 自动发现 |
| 手工 scorer CSV → run_adas_filter.sh | 原有流程不变 |
| 两者混放同一目录 | `_align_columns` 自动对齐缺失列 |
| CSV → compute_stats.py | `token, pdms, pdms_scaled` 命中优先路径 |

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
| `scripts/adas/filter_dynamic.py` | 三阶段筛选 + 输出 .txt token list |
| `scripts/adas/enrich_prefill.py` | Prefill 元信息注入 |
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
