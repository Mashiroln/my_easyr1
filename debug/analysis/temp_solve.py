# 读取源文件并按行拆分
with open("generations_dynamic_normalized.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 计算拆分点（向下取整，前半部分可能比后半部分少1行）
mid = len(lines) // 2

# 写入第一部分
with open("generations_dynamic_normalized_traj.jsonl", "w", encoding="utf-8") as f1:
    f1.writelines(lines[:mid])

# 写入第二部分
with open("generations_dynamic_normalized_cot.jsonl", "w", encoding="utf-8") as f2:
    f2.writelines(lines[mid:])