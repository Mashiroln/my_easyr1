import json

input_file = "recog_diverse_policy_stats.jsonl"
output_file = "_recog_diverse_policy_stats.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        if "PDMS" in obj:
            obj["pdms"] = obj.pop("PDMS")
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
