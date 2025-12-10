from datasets import load_dataset
dataset = load_dataset("/mnt/data/ccy/EasyR1/data/navsim_normtrajtext_cot_filter_mix_88step_3k", split="train")
# dataset = load_dataset("hiyouga/math12k", split="train")
print(dataset[0])
tokens = []
# for item in dataset:
#     tokens.append(item['answer']['token'])
# print(tokens)
print(dataset.__len__())

from transformers import Qwen2_5_VLForConditionalGeneration