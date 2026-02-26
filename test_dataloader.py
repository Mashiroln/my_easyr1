from datasets import load_dataset
dataset = load_dataset("/mnt/data/ccy/EasyR1/data/CPR103k_v4_filter_dynamic_5k/data", split="train")
# dataset = load_dataset("hiyouga/math12k", split="train")
print(dataset[0])
tokens = []
# for item in dataset:
#     # tokens.append(item['answer']['token'])
#     token = item['answer']['token']
#     print(token)
# print(tokens)
print(dataset.__len__())

# from transformers import Qwen2_5_VLForConditionalGeneration

