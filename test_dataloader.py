from datasets import load_dataset
dataset = load_dataset("/mnt/data/ccy/EasyR1/data/navsim_cot_action_k12_full", split="train")
# dataset = load_dataset("hiyouga/math12k", split="train")
print(dataset[0])
tokens = []
# for item in dataset:
#     tokens.append(item['answer']['token'])
# print(tokens)
print(dataset.__len__())
