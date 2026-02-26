# dev-local 分支同步指南

## 分支结构

```
origin   -> git@github.com-ccy:Mashiroln/my_easyr1.git (你的 fork)
upstream -> https://github.com/hiyouga/EasyR1.git     (上游仓库)
```

- `main`: 与 upstream/main 保持同步
- `dev-local`: 你的本地开发分支，基于 main 进行开发

## 历史做法（Merge）

从 git 历史可以看到，之前使用的是 **merge** 方式：

```
4ee0aad Merge branch 'main' into dev-local
```

## 推荐同步流程

### 1. 先同步 main 分支

```bash
# 切换到 main
git checkout main

# 拉取 upstream 最新
git fetch upstream

# 快进合并（main 应该只跟踪 upstream，不做本地修改）
git merge upstream/main --ff-only

# 推送到你的 origin
git push origin main
```

### 2. 将 main 合并到 dev-local

```bash
# 切换到 dev-local
git checkout dev-local

# 合并 main 到 dev-local
git merge main

# 解决冲突（如果有）
# git add .
# git commit

# 推送
git push origin dev-local
```

## Merge vs Rebase 选择

| 方式 | 优点 | 缺点 |
|------|------|------|
| **Merge** (推荐) | 保留完整历史，冲突只解决一次 | 历史有 merge commit |
| **Rebase** | 历史线性清晰 | 每次 rebase 可能重复解决冲突，需要 force push |

**你之前用的是 Merge，建议继续使用 Merge**，原因：
1. dev-local 是长期维护的分支，merge 更安全
2. 不需要 force push，不会丢失历史
3. 冲突只需解决一次

## 快速命令（一键同步）

```bash
# 完整同步流程
git checkout main && \
git fetch upstream && \
git merge upstream/main --ff-only && \
git push origin main && \
git checkout dev-local && \
git merge main && \
git push origin dev-local
```

## Remote 配置

```
origin   -> git@github.com-ccy:Mashiroln/my_easyr1.git (SSH)
upstream -> https://github.com/hiyouga/EasyR1.git      (HTTPS)
```

## 当前状态

- `dev-local` 已包含 `main` 的所有内容 ✓
- `dev-local` 有 8 个独有提交（不含 merge commit）
- 无需立即同步
