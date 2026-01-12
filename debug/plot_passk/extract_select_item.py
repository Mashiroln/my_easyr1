import orjson

jsonl_path = "/mnt/data/ccy/EasyR1/debug/analysis/navie_cot_text/generations_policy_stats.jsonl"   # 你的 JSONL 文件路径
txt_path = "selected_tokens.txt"     # 你的 TXT 文件路径
output_path = "archived_data/qwen_vla_selected_all.jsonl"  # 输出筛选后的 JSONL

log_path = "bad_lines.log"

with open(txt_path, "r", encoding="utf-8") as f:
    tokens_set = set(line.strip() for line in f if line.strip())

bad_count = 0

with open(jsonl_path, "rb") as infile, \
     open(output_path, "wb") as outfile, \
     open(log_path, "w", encoding="utf-8") as logf:

    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
        except orjson.JSONDecodeError as e:
            bad_count += 1
            logf.write(f"Line {i}: {e}\n")
            continue
        token = obj.get("token")
        if token in tokens_set:
            outfile.write(orjson.dumps(obj) + b"\n")

print(f"Done. {bad_count} bad lines skipped.")