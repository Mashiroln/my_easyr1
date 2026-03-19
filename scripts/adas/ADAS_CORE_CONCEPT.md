# ADAS: Adaptive Diversity-Aware Sampling

> 论文: Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models (CVPR 2026)

---

## 一句话

IL 后模型策略熵崩溃，大多数场景的 rollout 几乎一致，RL 梯度消失。ADAS 在每轮训练前筛选"高多样性"场景作为 active training set，只在这些场景上做 GRPO。

---

## 筛选条件

对每个场景做 G 次 rollout，估计 `p̂ = pdms_mean / pdms_range`：

| 条件 | 公式 | 含义 |
|------|------|------|
| std gate | `σ > ε_std` | 去掉零方差场景 |
| diversity | `p^G + (1-p)^G < ε_div` | Bernoulli 多样性，p 越接近 0.5 越好 |
| confidence | `\|σ_obs - σ_theory\| / σ_obs < ε_conf` | 观测 std 与 Bernoulli 预测 std 一致 |

默认超参：`-p 0.1 --conf 0.1 --std_threshold 0.01 --n_rollout 8`

`pipeline.py` 有 auto-tune：遍历多个 `-p` 值，选使样本数落在 `[target_min, target_max]` 的阈值。

---

## 端到端流程

```
checkpoint → [推理+打分] → scorer CSV → [pipeline.py] → token list → [训练]
                                          merge→stats→filter
```

### Step 1: 推理 + 打分（parallel_infer，生产级）

推理和打分本质上是两个解耦的阶段，目前的生产代码将其封装到同一个project中，通过多个脚本协作完成。

推理和打分由 `/mnt/data/ccy/VLA_train/parallel_infer/` 完成，这是经过深度优化的独立 infra，比 `Easry1/verl/trainer/main_adas.py` 快约 20 倍。

#### 架构

```
vLLM serve (8 GPU, 每 GPU 一个实例, 端口 8192-8199)
    ↑
parallel_infer/main.py (48 worker 进程, 每进程 16 inflight)
    ↓
推理输出 JSONL (每行: id, predict, repeat_index, ...)
    ↓
scorer/aggregate_traj.py (按 token 聚合, denormalize)
    ↓
scorer/send_score_request.py (UDS RPC score_group, 192 并发)
    ↓
scorer CSV (每行: token, pdms, pdms_scaled, valid, ...)
```

#### 关键设计（为什么快 20×）

| 优化 | 原理 |
|------|------|
| 固定绑定端口 | `worker_idx % num_servers`，提升 continuous batching 稳定性和 prefix cache 命中 |
| steady-state pipeline | 每个 worker 保持 `inflight` 个请求常驻，完成一个补一个，GPU 不断粮 |
| choices(n=K) | 一次请求返回 K 个 rollout，prefill 一次 decode K 次 |
| 按 token 聚合打分 | `score_group(token, trajectories_se2=[...])` 一次 RPC 打一组，减少 UDS 开销 |
| 断点续传 | 按 `(id, repeat_index)` 追踪，中断后只补缺失的 |

#### 使用

```bash
cd /mnt/data/ccy/VLA_train/parallel_infer

# 1. 启动 vLLM serve，第一个tmux session
bash scripts/vllm_serve_local.sh 8

# 2. 推理（repeat=8 示例）, 第二个tmux session
bash scripts/run_infer_adas_repeat8.sh

# 3. 聚合
python scorer/aggregate_traj.py \
    --exp_name ${EXP_NAME} \
    --input_dir output \
    --all_hosts --format pkl

# 4. 打分
python scorer/send_score_request.py \
    --exp_name ${EXP_NAME} \
    --input_dir output \
    --version 2 --concurrent_workers 192
```

详见：
- 推理：`/mnt/data/ccy/VLA_train/parallel_infer/README.md`
- 打分：`/mnt/data/ccy/VLA_train/parallel_infer/scorer/README.md`

#### main_adas.py（demo/备用）

`verl/trainer/main_adas.py` 是一个自包含的 Ray + vLLM rollout + reward 推理脚本，功能等价但速度远慢于 parallel_infer。仅用于：
- 开源版本提供开箱即用（但缓慢的）入口
- 快速验证 pipeline 是否跑通
- 没有 parallel_infer 环境时的 fallback

---

### Step 2: 过滤（pipeline.py）[ADAS核心算法]

```bash
cd scripts/adas
python pipeline.py \
    --infer_folder /path/to/scorer_csv_dir \
    -p 0.1 --conf 0.1
```

内部三步：

| 步骤 | 脚本 | 输入 → 输出 |
|------|------|-------------|
| merge | `merge_scorer_csv.py` | scorer CSV → `generations_full.parquet` |
| stats | `compute_stats.py` | parquet → `generations_full.csv`（每 token 一行：mean/std/range） |
| filter | `filter_dynamic.py` | stats CSV → `*_filtered_{p}.txt`（token list） |

auto-tune 默认开启，目标 3000-6000 样本。

---

### Step 3: 训练（GRPO）

```bash
# === Model & Data ===
MODEL_PATH=/path/to/sft_or_previous_round_checkpoint
data_path=/path/to/EasyR1/data/your_dataset
exp_name=your_adas_train_exp
active_data_path=/path/to/generations_full_filtered_0.1.txt

# === Environment ===
export EXP_NAME=${exp_name}
export NAVSIM_STAT_PATH="/path/to/stats/trajectory_stats.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag

reward_function_path=/path/to/EasyR1/verl/utils/reward_score/navsim/navsim_reward_text_rpc.py

python3 -m verl.trainer.main \
    config=examples/config_vla.yaml \
    data.train_files=${data_path}@train \
    data.val_files=${data_path}@test \
    data.format_prompt=None \
    data.max_response_length=3072 \
    data.rollout_batch_size=448 \
    data.val_batch_size=896 \
    data.token_filter_file=${active_data_path} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=224 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_function_path}:compute_score_fast \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=12 \
    trainer.save_freq=20
```

关键环境变量：
- `EXP_NAME`：实验名，reward 日志目录依赖此变量
- `NAVSIM_STAT_PATH`：轨迹归一化统计文件（denormalize 用）
- `NAVSIM_TRAJ_PARSER_FUNC`：轨迹解析函数路径

Dataloader 读取 `token_filter_file`，只加载匹配的 token 进行训练。

`token_filter_file` 支持：
- 单个文件：`path/to/file.txt`
- 多文件 list：`[path/to/a.jsonl, path/to/b.txt]`（Dataloader 自动检测每个 entry 是否 prefill）

---

## Multi-round Outer Loop

```
Round 0: SFT ckpt    → 推理+打分 → filter → 训练 → ckpt_0
Round 1: ckpt_0      → 推理+打分 → filter → 训练 → ckpt_1
Round 2: ckpt_1      → 推理+打分 → filter → 训练 → ckpt_2
```

每轮重新采样 active set，因为模型能力变化后高多样性场景集合也变。论文用 3 轮。

---

## 代码索引

### ADAS 过滤（本仓库）

```
scripts/adas/
├── pipeline.py              # 一键 pipeline（merge→stats→filter + auto-tune）
├── merge_scorer_csv.py      # CSV → Parquet
├── compute_stats.py         # Parquet → 组级统计
├── filter_dynamic.py        # 三阶段过滤 → token list
├── run_adas_filter.sh       # 过滤脚本 (生产级)
└── run_adas_infer.sh        # main_adas.py 推理示例（demo，不要使用）
```

### 推理 + 打分（parallel_infer，生产级）

```
/mnt/data/ccy/VLA_train/parallel_infer/
├── main.py                  # 并行推理入口（多进程 + steady-state + choices(n=K)）
├── api_client.py            # vLLM OpenAI client（固定端口绑定）
├── scripts/
│   ├── vllm_serve_local.sh  # 启动 vLLM 多实例
│   └── run_infer_*.sh       # 各类推理脚本
└── scorer/
    ├── aggregate_traj.py    # 按 token 聚合 + denormalize
    ├── send_score_request.py # UDS RPC 批量打分
    └── ray_scorer_rpc_client.py  # RPC client
```

### 训练框架集成

```
verl/trainer/main_adas.py    # ADAS 推理入口（demo/备用）
verl/trainer/config.py       # DataConfig.token_filter_file: str | list[str]
verl/utils/dataset.py        # RLHFDataset token filter + prefill 自动检测
```
