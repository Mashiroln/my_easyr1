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

## Merge 到 dev-local 的改动清单

### 文件分类

**A 类 — 新文件（直接 checkout）**

| 文件 | 说明 |
|------|------|
| `verl/trainer/main_adas.py` | ADAS inference 入口 |
| `scripts/adas/run_adas_infer.sh` | ADAS 推理示例 |
| `scripts/adas/run_adas_filter.sh` | ADAS 筛选示例 |
| `scripts/adas/ADAS_INTEGRATION_PLAN.md` | 本文档 |
| `train_scripts/train_adas_new.sh` | ADAS 训练示例 |

**B 类 — 整文件 checkout（dev-local 侧无冲突）**

| 文件 | 说明 |
|------|------|
| `verl/trainer/config.py` | DataConfig 新增 `token_filter_file` |
| `verl/utils/dataset.py` | RLHFDataset 支持 token_filter_file 过滤 |
| `verl/trainer/data_loader.py` | create_dataloader 传入 token_filter_file |

**C 类 — 手动加一行（两分支都改了同一文件）**

| 文件 | 说明 |
|------|------|
| `examples/config_vla.yaml` | dev-local 的 `6c3168e` (segment sapo) 也改了此文件。只需手动加 `token_filter_file: null` |

**D 类 — 部分合并（只 cherry-pick 特定改动，不 checkout 整文件）**

| 文件 | 要合并的改动 | 不合并的改动 |
|------|-------------|-------------|
| `navsim_reward_text.py` | httpx 替换 requests、httpx INFO log 抑制、log dir → `checkpoints/adas/` | 其余开源清理 |
| `scripts/adas/filter_dynamic.py` | 新增输出 `.txt` 功能 | 其余参数简化 |

**E 类 — 不合并**

| 文件 | 原因 |
|------|------|
| `helper.py` | syn stats 移除仅限开源 |
| `navsim_reward_text_rpc.py` | denormalize 签名改动仅限开源 |
| `pipeline.py` / `compute_stats.py` / `merge_scorer_csv.py` | 重构仅限开源 |
| `enrich_prefill.py` 删除 | 仅开源版本移除 |
| `run_adas.sh` 删除 | 仅开源版本移除 |
| `scripts/copy_to_opensource.sh` | 舍弃 |

---

## Git 操作步骤

当前分支状态：

```
de90f8a (共同祖先)
  ├── 6c3168e (dev-local HEAD) — segment sapo
  │     └── stash@{0}: dev-local 未提交的工作
  └── c7ad6ab (opensource-adas HEAD) — token mask + pipeline
        └── 未提交: main_adas.py, navsim httpx, 脚本, 开源清理 ...
```

### Step 0: 在 opensource-adas 上提交快照

```bash
git add -A
git commit -m "[feat] ADAS: inference runner + open-source cleanup"
```

保留开源版本的完整快照，后续不再需要此分支的未提交状态。

### Step 1: 切到 dev-local

```bash
git checkout dev-local
```

暂不 pop stash，先合并 ADAS 改动。

### Step 2: Checkout A 类 + B 类文件

```bash
# A 类：新文件
git checkout opensource-adas -- verl/trainer/main_adas.py
git checkout opensource-adas -- scripts/adas/run_adas_infer.sh
git checkout opensource-adas -- scripts/adas/run_adas_filter.sh
git checkout opensource-adas -- scripts/adas/ADAS_INTEGRATION_PLAN.md
git checkout opensource-adas -- train_scripts/train_adas_new.sh

# B 类：整文件（dev-local 侧这些文件自 de90f8a 以来无改动）
git checkout opensource-adas -- verl/trainer/config.py
git checkout opensource-adas -- verl/utils/dataset.py
git checkout opensource-adas -- verl/trainer/data_loader.py
```

### Step 3: 手动处理 C 类

`examples/config_vla.yaml`：在 `filter_overlong_prompts: true` 下一行加：

```yaml
  token_filter_file: null
```

### Step 4: 手动处理 D 类

`navsim_reward_text.py`：在 dev-local 版本上手动应用 3 处改动：
1. `import requests` → `import httpx` + `logging.getLogger("httpx").setLevel(logging.WARNING)`
2. `requests.post(...)` → `httpx.Client(trust_env=False, timeout=timeout)` 模式
3. log 路径从 `debug/` → `checkpoints/adas/${EXP_NAME}/`

`filter_dynamic.py`：在 dev-local 版本上追加输出 `.txt` 的逻辑。

### Step 5: 提交

```bash
git add verl/trainer/main_adas.py \
       verl/trainer/config.py \
       verl/utils/dataset.py \
       verl/trainer/data_loader.py \
       examples/config_vla.yaml \
       verl/utils/reward_score/navsim/navsim_reward_text.py \
       scripts/adas/run_adas_infer.sh \
       scripts/adas/run_adas_filter.sh \
       scripts/adas/ADAS_INTEGRATION_PLAN.md \
       scripts/adas/filter_dynamic.py \
       train_scripts/train_adas_new.sh

git commit -m "[feat] ADAS: inference integration + token filter dataloader"
```

### Step 6: Pop stash

```bash
git stash pop
```

如有冲突，手动解决后继续开发。
