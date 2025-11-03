import os
from safetensors.torch import load_file
import torch

def load_all_safetensors(model_dir):
    """
    从一个 HF 模型目录加载所有 .safetensors 文件并合并为一个 dict。
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path) as f:
            index = json.load(f)
        tensors = {}
        for weight_file in sorted(set(index["weight_map"].values())):
            full_path = os.path.join(model_dir, weight_file)
            print(f"Loading {full_path}")
            current_tensors = load_file(full_path)
            print("current key count", len(set(current_tensors.keys())))
            tensors.update(load_file(full_path))
    else:
        # 单文件模型
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors"):
                print(f"Loading {fname}")
                tensors = load_file(os.path.join(model_dir, fname))
                break
    return tensors

def summarize_differences(sd1, sd2):
    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)
    both = sorted(keys1 & keys2)

    shape_mismatch = []
    dtype_mismatch = []
    for k in both:
        t1, t2 = sd1[k], sd2[k]
        if t1.shape != t2.shape:
            shape_mismatch.append((k, t1.shape, t2.shape))
        elif t1.dtype != t2.dtype:
            dtype_mismatch.append((k, t1.dtype, t2.dtype))

    print("\n=== Summary ===")
    print(f"Total keys in model1: {len(keys1)}")
    print(f"Total keys in model2: {len(keys2)}")
    print(f"Only in model1: {len(only1)}")
    print(f"Only in model2: {len(only2)}")
    print(f"Shape mismatch: {len(shape_mismatch)}")
    print(f"Dtype mismatch: {len(dtype_mismatch)}")

    # print(f"model1 keys: {keys1}")

    if only1:
        print("\n--- Keys only in model1 ---")
        print("\n".join(only1[:20]))
        if len(only1) > 20:
            print(f"... ({len(only1)-20} more)")

    if only2:
        print("\n--- Keys only in model2 ---")
        print("\n".join(only2[:20]))
        if len(only2) > 20:
            print(f"... ({len(only2)-20} more)")

    if shape_mismatch:
        print("\n--- Shape mismatches ---")
        for k, s1, s2 in shape_mismatch[:10]:
            print(f"{k}: {s1} vs {s2}")
        if len(shape_mismatch) > 10:
            print(f"... ({len(shape_mismatch)-10} more)")

if __name__ == "__main__":
    model1 = "/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_grpo_dynamic_analysis_6k/global_step_50/actor/huggingface"
    model2 = "/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_normtrajtext_cot_filter_dynamic_6k/global_step_88/actor/huggingface"

    print(f"Loading model1: {model1}")
    sd1 = load_all_safetensors(model1)
    print(f"Loaded {len(sd1)} tensors from model1")

    print(f"\nLoading model2: {model2}")
    sd2 = load_all_safetensors(model2)
    print(f"Loaded {len(sd2)} tensors from model2")

    summarize_differences(sd1, sd2)
