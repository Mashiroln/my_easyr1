# GRPO with Prefill — 重构 & 迁移计划 (v3)

**来源分支**: `save/20260303-prefillmix-nanfix` (commit `f1c4888`)
**目标分支**: `dev-local` (HEAD `bb53954`)
**排除**: 所有 SBRPO 代码 + 所有硬件 debug 中间代码（NaN/CUDA fixes）

---

## 〇、实施进度

| Commit | 状态 | 涉及文件 | 备注 |
|--------|------|----------|------|
| 1. TrajectoryNormalizer + helper.py 重构 | ✅ 完成 | `verl/utils/trajectory_normalizer.py` (新增), `verl/utils/reward_score/navsim/helper.py` | 已验证 roundtrip 一致性 |
| 2. filter_dynamic JSONL 输出 | ✅ 完成 | `scripts/adas/filter_dynamic.py`, `scripts/adas/pipeline.py` | 新增 `enriched_parquet` 参数 + JSONL 输出 |
| 3. Dataloader prefill injection | ✅ 完成 | `verl/utils/dataset.py` | `_load_filter_spec` + `_sample_entries` + prefill injection + robust parquet loading |
| 4. Trainer + reward 适配 | ✅ 完成 | `verl/trainer/ray_trainer.py`, `verl/workers/reward/function.py` | sample_id uid, token 提取, metrics string filter, reward 透传 |
| 5. Prefillmix 训练脚本 | ✅ 完成 | `train_scripts/qwen3vl/qwen3_vl_4b_dynamic_grpo_prefillmix.sh` (新增) | |

**额外改进** (实施过程中追加):
- `TrajectoryNormalizer` 支持 `NAVSIM_STAT_PATH_SYN` 的两种模式：
  - **单文件模式**: 指向一个 stats JSON → 所有 `-00x` token 共用
  - **索引文件模式**: 指向 `.txt`，每行是一个 stats JSON 路径 → `-00x` 的 `x` 对应第 `x` 行

**待完成**:
- [ ] 尚未 git commit（所有改动在工作区）
- [ ] 端到端验证（§六 验证策略）
- [ ] 实际训练运行测试

---

## 一、完整数据流 & meta info 索引图

### 1.1 每个阶段的输入/输出和对 meta info 的依赖

```
阶段                               输入                    输出                        需要索引 meta?
─────────────────────────────────────────────────────────────────────────────────────────────────────
[外部] 1. No-prefill 推理          数据集 + 模型           JSONL (token, predict)      ✗
[外部] 2. extract_coarse_refine    JSONL + stats.json      bin shards (denorm traj)    ✗
[外部] 3. analyze_coarse_intra     bin shards              ★ centers.jsonl             ✗  ← 产生 meta
[外部] 4. Prefill 推理             ★ centers.jsonl         JSONL + prefill_meta.jsonl  ✓  ← 读取 meta (选择稀有 center)
[外部] 5. aggregate_traj           JSONL                   agg.pkl                     ✗  (透传 meta_id)
[外部] 6. send_score_request       agg.pkl                 scorer CSV                  ✗  (透传 meta_id)
─────────────────────────────────────────────────────────────────────────────────────────────────────
[EasyR1] 7. merge_scorer_csv       scorer CSV              parquet                     ✗  (透传 meta_id)
[EasyR1] 8. enrich_prefill         parquet + meta.jsonl    enriched parquet            ✓  ← 读取 meta (解析 center_denorm)
                                   + ★ centers.jsonl
[EasyR1] 9. compute_stats          enriched parquet        stats CSV                   ✗
[EasyR1] 10. filter_dynamic        stats CSV + enriched pq filter_spec.jsonl           ✗  (从 enriched pq 读 denorm)
─────────────────────────────────────────────────────────────────────────────────────────────────────
[EasyR1] 11. Dataloader            原始数据集 + jsonl      training batch              ✗  ← 完全自包含
[EasyR1] 12. vLLM rollout          prefilled prompt        生成结果                    ✗
```

**结论**：meta info (`centers.jsonl` + `prefill_meta.jsonl`) 只在两个阶段被索引：
- **阶段 4** (外部 prefill 推理)：从 `centers.jsonl` 选择稀有 center 做推理
- **阶段 8** (EasyR1 `enrich_prefill.py`)：通过 `prefill_meta_id` → `(centers_row, center_rank)` → 回查 `centers.jsonl` 取出 denorm center

**从阶段 10 (filter_spec.jsonl) 开始，数据完全自包含**：
- JSONL 中直接嵌入了 8x3 denorm center，不再需要任何 meta 文件
- `prefill_meta_id` 仅作为 debug 溯源字段保留

### 1.2 为什么 extract/analyze 不需要改动

| 脚本 | 是否需要改动 | 原因 |
|------|-------------|------|
| `extract_coarse_refine_traj.py` | **不改** | `.bin` 是性能关键的内部格式，用于 128 进程并行分析。人类不可读但速度极快 (~30s 处理数百万条)。最终产物 `centers.jsonl` 是可读 JSONL |
| `analyze_coarse_intragroup.py` | **不改** | 同上。输出 `coarse_intragroup_centers.jsonl` 已是清晰的 JSONL，每行含 `{token, k, cluster_sizes, coarse_centers}`，coarse_centers 是 denorm 8x3 |
| `.bin` 文件 | **不改** | 这是中间产物，只在 extract→analyze 之间存在。不暴露给下游。性能优先的正确工程选择 |

**唯一需要改动的是 EasyR1 仓库内的代码**：
- `scripts/adas/filter_dynamic.py` — 输出 JSONL
- `verl/utils/dataset.py` — prefill injection
- `verl/utils/trajectory_normalizer.py` — **新增**共享 normalizer
- `verl/trainer/ray_trainer.py` — trainer 适配

### 1.3 始终传递 denorm 原则

轨迹坐标在整个链路中的形式：

| 阶段 | 坐标形式 | 备注 |
|------|----------|------|
| 模型输出 predict text | norm | 模型 I/O 空间 |
| extract → bin | **denorm** | 聚类需要物理尺度 |
| centers.jsonl | **denorm** | medoid 在 denorm 空间选取 |
| enriched parquet (prefill_center_denorm) | **denorm** | 从 centers.jsonl 直取 |
| filter_spec.jsonl (coarse_center) | **denorm** | 自包含，不依赖任何外部文件 |
| Dataloader 内部 | denorm → **normalize → format** | 训练时刻才做，使用 TrajectoryNormalizer |
| vLLM prompt | norm (formatted) | `[PT, (+0.12, ...), ...]` |

---

## 二、重构设计

### 2.1 `TrajectoryNormalizer` — 共享 normalizer 类

**动机**：当前 `helper.py` 有双 stats 机制 (`NAVSIM_STAT_PATH` + `NAVSIM_STAT_PATH_SYN`)，通过 token 后缀 (`-00\d$`) 判断使用哪套 stats。reward function 和 prefill dataloader 都需要这个逻辑。

**新文件**: `verl/utils/trajectory_normalizer.py`

**`NAVSIM_STAT_PATH_SYN` 双模式支持**：

| 模式 | `NAVSIM_STAT_PATH_SYN` 值 | 行为 |
|------|--------------------------|------|
| 单文件 | `/path/to/stats.json` | 所有 `-00x` 后缀 token 共用此 stats |
| 索引文件 | `/path/to/syn_stats.txt` | TXT 每行一个 JSON 路径，`-00x` 的 `x` 对应第 `x` 行 (0-indexed) |
| 未设置 | 空 | 所有 token 使用 `NAVSIM_STAT_PATH` |

核心 dispatch 逻辑：
```python
_SYN_TOKEN_RE = re.compile(r"-00(\d)$")  # 捕获组提取 x

def _pick_stats(self, token):
    m = _SYN_TOKEN_RE.search(str(token))
    if not m:
        return self.stats  # 普通 token → main stats
    syn_idx = int(m.group(1))
    if self._syn_indexed is not None:
        # 索引 TXT 模式: 按行号加载 (带缓存)
        return self._load_cached(self._syn_indexed[syn_idx])
    elif self._syn_single is not None:
        # 单文件模式
        return self._syn_single
    else:
        return self.stats  # 未配置 → fallback
```

提供 `normalize()` / `denormalize()` / `format_prefill_text()` 三个公开方法。

**`helper.py` 改造**：让现有 reward function 的 `denormalize()` 委托给 `TrajectoryNormalizer`：

```python
# helper.py 改为:
from verl.utils.trajectory_normalizer import TrajectoryNormalizer

_normalizer = TrajectoryNormalizer()

def denormalize(poses, token=None):
    return _normalizer.denormalize(poses, token=token).tolist()
```

这样 reward function 行为完全不变，但 normalizer 逻辑被统一。

### 2.2 Filter Spec JSONL 格式

```jsonl
{"token": "abc123def456"}
{"token": "abc123def456", "coarse_center": [[0.52,1.21,0.01],[1.03,2.34,0.02],...], "prefill_meta_id": 12345}
{"token": "abc123def456", "coarse_center": [[-0.31,0.82,-0.02],[0.12,1.45,0.03],...], "prefill_meta_id": 12346}
```

| 字段 | 必需? | 说明 |
|------|-------|------|
| `token` | 是 | 场景 token |
| `coarse_center` | 否 | 8x3 denorm 轨迹。有 = prefill 样本 |
| `prefill_meta_id` | 否 | debug 溯源用，训练不依赖 |

**自包含保证**：JSONL 文件可以独立工作，不需要索引任何 meta 文件。`coarse_center` 是完整的 denorm 轨迹数据。

### 2.3 Pipeline 改造 (`scripts/adas/`)

#### `filter_dynamic.py` — 输出 JSONL + 从 enriched parquet 读 denorm center

新增参数 `enriched_parquet`。Prefill 模式下，从 enriched parquet 的 `prefill_center_denorm` 列读取 denorm center，写入 JSONL。

```python
def _load_center_denorm_lookup(enriched_parquet: str) -> dict:
    """Build (token, prefill_meta_id) → center_denorm lookup."""
    import pyarrow.dataset as ds
    lookup = {}
    dataset = ds.dataset(enriched_parquet, format="parquet")
    for batch in dataset.to_batches(columns=["token", "prefill_meta_id", "prefill_center_denorm"]):
        for row in batch.to_pylist():
            key = (str(row["token"]), int(row["prefill_meta_id"]))
            center = row.get("prefill_center_denorm")
            if center is not None:
                lookup[key] = center
    return lookup
```

#### `pipeline.py` — 透传 enriched_parquet

`enrich_prefill.py` 已经输出 `prefill_center_denorm` 列（当前实现中已有）。Pipeline 将 enriched parquet 路径传给 `filter_dynamic()`。

### 2.4 Dataloader 改造 (`verl/utils/dataset.py`)

#### `_load_filter_spec()` — 替换 `_load_token_filter_set()`

```python
@dataclass
class FilterEntry:
    token: str
    coarse_center: Optional[List[List[float]]] = None  # 8x3 denorm, None = base sample

def _load_filter_spec(path: str) -> list[FilterEntry]:
    """支持 JSONL (新) 和 TXT (旧) 两种格式，自动检测"""
    entries = []
    with open(path, "r") as f:
        first = f.readline().strip()
        f.seek(0)
        if first.startswith("{"):
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                entries.append(FilterEntry(
                    token=obj["token"],
                    coarse_center=obj.get("coarse_center"),
                ))
        else:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(FilterEntry(token=line.split("\t")[0]))
    return entries
```

#### `__init__` — 构建有效样本索引

```python
# RLHFDataset.__init__ 中:
if token_filter_file is not None:
    filter_entries = _load_filter_spec(token_filter_file)
    token_set = {e.token for e in filter_entries}

    # 按 token 筛选原始数据集
    self.dataset = self.dataset.filter(lambda ex: ex[answer_key]["token"] in token_set, ...)

    has_prefill = any(e.coarse_center is not None for e in filter_entries)
    if has_prefill:
        # 构建 token → dataset_index
        token_to_idx = {}
        for i in range(len(self.dataset)):
            token_to_idx[self.dataset[i][answer_key]["token"]] = i

        # 有效样本索引: (dataset_idx, coarse_center_or_None)
        self._sample_entries = [
            (token_to_idx[e.token], e.coarse_center)
            for e in filter_entries if e.token in token_to_idx
        ]
        # TrajectoryNormalizer (从环境变量读取 stats)
        from verl.utils.trajectory_normalizer import TrajectoryNormalizer
        self._traj_normalizer = TrajectoryNormalizer()
    else:
        self._sample_entries = None
        self._traj_normalizer = None
```

#### `__len__` + `__getitem__`

```python
def __len__(self):
    if self._sample_entries is not None:
        return len(self._sample_entries)
    return len(self.dataset)

def __getitem__(self, index):
    if self._sample_entries is not None:
        dataset_idx, coarse_center = self._sample_entries[index]
        example = dict(self.dataset[dataset_idx])
    else:
        coarse_center = None
        example = dict(self.dataset[index])

    # ... 现有 tokenization 逻辑不变 ...

    raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

    # Prefill injection: denorm → normalize → format → tokenize → append
    if coarse_center is not None:
        token_str = example.get(self.answer_key, {}).get("token")
        prefill_text = self._traj_normalizer.format_prefill_text(
            coarse_center, token=token_str
        )
        prefill_ids = self.tokenizer.encode(prefill_text, add_special_tokens=False)
        raw_prompt_ids = raw_prompt_ids + prefill_ids
        example["is_prefill"] = True

    # ... truncation 等后续逻辑不变 ...
```

注意 `format_prefill_text` 接收 `token` 参数，内部自动判断使用哪套 stats（regular vs synthetic）。

### 2.5 Robust local parquet loading

同时迁入 prefillmix 的本地 parquet 加载改进（替换 `dataset.py` line 144）。

### 2.6 Trainer 适配 (`verl/trainer/ray_trainer.py`)

从 prefillmix 分支迁入（只取非 SBR、非 NaN-debug 部分）：

1. **从 `ground_truth` 提取 `token`** → `non_tensor_batch["token"]`
2. **`sample_id` 作为 `uid`**（存在时）
3. **`reduce_metrics` 前过滤 string keys**

### 2.7 Reward 透传 (`verl/workers/reward/function.py`)

透传 `prefill_trajectory`, `prefill_score`, `pool_type`（存在时）。

---

## 三、`helper.py` 重构说明

当前 `helper.py` 的 stats 管理是模块级全局变量：

```python
# 当前 helper.py
means, stds = _load_stats(stat_path)
means_syn, stds_syn = _load_stats(stat_path_syn) if stat_path_syn else (means, stds)

def denormalize(poses, token=None):
    m, s = (means_syn, stds_syn) if _use_syn_stats(token) else (means, stds)
    return (np.array(poses) * s + m).tolist()
```

重构后：

```python
# helper.py
from verl.utils.trajectory_normalizer import TrajectoryNormalizer
_normalizer = TrajectoryNormalizer()

def denormalize(poses, token=None):
    return _normalizer.denormalize(poses, token=token).tolist()
```

`TrajectoryNormalizer` 封装了：
- 双 stats 加载 (`NAVSIM_STAT_PATH` + `NAVSIM_STAT_PATH_SYN`)
- Token 后缀 dispatch (`-00(\d)$` → syn stats)
- `NAVSIM_STAT_PATH_SYN` 单文件 / 索引 TXT 双模式
- `normalize()` / `denormalize()` / `format_prefill_text()`

这样 reward function 和 dataloader prefill injection 共用完全相同的 normalize/denormalize 逻辑。

---

## 四、不迁移的内容

| 内容 | 原因 |
|------|------|
| P1 全部 (torch_functional, fsdp_workers, fsdp_vllm, dp_actor NaN debug) | 硬件 debug 中间代码, 硬件已修复 |
| `scripts/sbr/`, `verl/utils/sbr/`, SBR 相关全部 | SBRPO 专用 |
| `scripts/concat_dynamic_prefill_parquet.py` | 被 filter_spec JSONL 取代 |
| `debug/aug_adas/` 全部 | 功能被 `scripts/adas/` 覆盖 + 新 JSONL 方案取代 |
| `verl/trainer/core_algos.py` segment_sapo 禁用 | dev-local 有可用实现 |
| `extract_coarse_refine_traj.py` / `analyze_coarse_intragroup.py` | 不需要改动。`.bin` 是性能关键内部格式，最终产物 `centers.jsonl` 已是可读 JSONL |

---

## 五、实施顺序

### Commit 1: `[feat] TrajectoryNormalizer: shared normalizer for reward + prefill`
- 新增 `verl/utils/trajectory_normalizer.py`
- 重构 `verl/utils/reward_score/navsim/helper.py` 委托给 TrajectoryNormalizer
- 行为完全不变，纯重构

### Commit 2: `[feat] filter_dynamic: JSONL output with denorm coarse center`
- `scripts/adas/filter_dynamic.py`: 新增 JSONL 输出 + `enriched_parquet` 参数
- `scripts/adas/pipeline.py`: 透传 enriched parquet，返回 `.jsonl` 路径
- 保留 `.txt` + `.csv` 输出（向后兼容）

### Commit 3: `[feat] dataloader: prefill injection from filter_spec.jsonl`
- `verl/utils/dataset.py`:
  - `FilterEntry` + `_load_filter_spec()` (JSONL/TXT 双格式)
  - `_sample_entries` 间接寻址, `__len__` / `__getitem__` 适配
  - 使用 `TrajectoryNormalizer.format_prefill_text()` 做 normalize+format
  - Robust local parquet loading

### Commit 4: `[feat] trainer: token extraction, sample_id uid, metrics filter`
- `verl/trainer/ray_trainer.py`: 3 处改动
- `verl/workers/reward/function.py`: prefill metadata 透传

### Commit 5: `[feat] prefillmix training script`
- `train_scripts/qwen3vl/qwen3_vl_4b_dynamic_grpo_prefillmix.sh`

---

## 六、验证策略

1. **TrajectoryNormalizer 单测**: 对比 `normalizer.denormalize()` 与旧 `helper.denormalize()` 输出一致
2. **Syn token dispatch**: 验证 `-001` 后缀 token 使用 syn stats
3. **No-prefill 回归**: 现有 `.txt` filter → 训练行为不变
4. **Pipeline JSONL**: `pipeline.py --csv_mode=prefill` → 检查 JSONL 中 `coarse_center` 是 denorm 8x3
5. **格式一致性**: 对比 `normalizer.format_prefill_text(center, token)` 与 `main_prefill.py:_build_prefill_text_from_center()` 输出完全一致
6. **Prefill injection**: JSONL 加载 → `raw_prompt_ids` 尾部包含正确 prefill token
7. **Mixed 训练**: `cat base.jsonl prefill.jsonl` → 训练正常
