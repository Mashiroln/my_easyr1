import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 生成 Mock 数据 (分离不同频率的数据) ---
np.random.seed(42)
n_runs = 5  # 模拟5次实验以产生阴影带

# 数据集 A: Validation Accuracy (保持原样，高频，每50步一个点)
steps_acc = np.arange(0, 2600, 50) 
data_acc = []

for _ in range(n_runs):
    noise = np.random.normal(0, 0.5, len(steps_acc))
    # 模拟准确率: 对数上升
    accuracy = 22 + 16 * (1 - np.exp(-steps_acc / 400)) + noise
    
    for i, step in enumerate(steps_acc):
        data_acc.append({"Step": step, "Accuracy": accuracy[i]})

df_acc = pd.DataFrame(data_acc)

# 数据集 B: Training Entropy (修改需求，低频，每300步一个点)
steps_ent = np.arange(0, 2600, 300) 
data_ent = []

for _ in range(n_runs):
    noise = np.random.normal(0, 0.01, len(steps_ent)) # 噪声稍小一点以免低频波动太大
    # 模拟熵: 指数下降
    entropy = 0.05 + 0.5 * np.exp(-steps_ent / 400) + noise
    
    for i, step in enumerate(steps_ent):
        data_ent.append({"Step": step, "Entropy": entropy[i]})

df_ent = pd.DataFrame(data_ent)


# --- 2. 设置绘图风格 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

color_acc = "#2c9087"  # Teal
color_ent = "#3b3b8f"  # Purple

fig, ax1 = plt.subplots(figsize=(8, 5))

# --- 3. 绘制左侧 Y 轴 (Accuracy - 高频) ---
# 使用 df_acc 数据集
sns.lineplot(
    data=df_acc, x="Step", y="Accuracy", 
    ax=ax1, color=color_acc, label="Test Accuracy",
    linewidth=2, errorbar='sd'
)

ax1.set_xlabel("Steps", fontsize=16)
ax1.set_ylabel("Validation Accuracy (%)", fontsize=16, color=color_acc)
ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
ax1.set_ylim(20, 40)

# --- 4. 绘制右侧 Y 轴 (Entropy - 低频 300 step) ---
ax2 = ax1.twinx()

# 使用 df_ent 数据集
# 为了让"300 step一个点"更明显，我添加了 marker='o'，如果不需要圆点可去掉
sns.lineplot(
    data=df_ent, x="Step", y="Entropy", 
    ax=ax2, color=color_ent, label="Training Entropy",
    linewidth=2, errorbar='sd', marker='o', markersize=6
)

ax2.set_ylabel("Entropy", fontsize=16, color=color_ent)
ax2.tick_params(axis='y', labelcolor=color_ent, labelsize=12)
ax2.set_ylim(0, 0.55)
ax2.grid(False)

# --- 5. 合并图例 ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=11)
if ax2.get_legend():
    ax2.get_legend().remove()

# --- 6. 装饰线 (可选) ---
ax1.hlines(y=37.5, xmin=0, xmax=2500, colors=color_acc, linestyles='--', alpha=0.7)
ax2.hlines(y=0.05, xmin=0, xmax=2500, colors=color_ent, linestyles='--', alpha=0.7)

plt.tight_layout()

# --- 7. 保存图片 ---
save_name = 'dual_y_axis_chart.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"图片已保存至当前目录: {save_name}")

plt.show()