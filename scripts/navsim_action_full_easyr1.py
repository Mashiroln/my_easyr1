# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocesses a ShareGPT4V-style JSON dataset to a structured Parquet format.

This script reads a large JSON file, restructures each item according to a
pre-defined mapping function (similar to verl/scripts/preprocess_geo3k.py),
and saves the result as a Parquet file.

It mirrors the transformation logic by:
1. Integrating the 'system' message into a new 'prompt' field.
2. Reformatting the 'image' field into a dictionary under an 'images' key.
3. Adopting a simplified I/O model for local and HDFS file handling.

Example usage:
----------------
python process_sharegpt4v.py \
    --input_json_path /path/to/your/QA_navsim_train.json \
    --local_dir ./data/navsim_parquet \
    --hdfs_dir hdfs://user/data/processed/navsim
----------------
"""

DEFAULT_TOEKN_LIST = ['8f4fbd68196b5c61', '411c6771f9985893', 'f99999645bd851ea', '520bcb47bdaa5685', 'e65b15b2baf05b05', 
'21c588bde4c7576a', '06e72237924559ba', 'dc9880f13fb85307', '46798825222d5a96', '482b5439cb6c5350', '7623dd6cbc29535c', 
'a9a5c33facc65562', '8fe9ff32681d576a', '3ffea98fd4db5f8d', '1d1e7480ff6e53a5', 'b5f27b8d489a5063', '0c79562f13b65929', 
'b30aa0fdf9fb57a5', 'b7c7c8f23b795794', 'f3c5429aa16852b0', 'f0ab7103b506598c', '61c7721242d35121', '8036c47e9c9f5818',
 'c68b15c055765b73', 'da1232cae7ff5812', '6151643563d9521c', '7b8a821e20b65dc4', 'c72a27927e065ce1', '761bc8feb786586b', 
 'c21ebac51f0a547a', 'cbbf3f5578a05f21', '26d603a303695c76', '0a861391e5915512', '288f0194b6d45858', '801cd7280c355e18', 
 '45e0a389984950c8', '668c88037cc25c02', '6cf2433326d45bf9', 'c6769fe924b451d4', 'b99c96fc9c635092', '650a3add83f15808', 
 '1d7f9f198e0c57bc', 'b0bc661f5b3a53d3', 'bb004da2772555d3', 'e0f88542017e5924', '82d7018f5e1c5ec1', '2797a61b55f050d1', 
 'e7f46a882ac2504b', 'b6194744063b5df4', '43518c87791656b0', '5fa95cf055cf5113', 'fc186ea3f2825a9f', 'e51e9a4250075dfa', 
 'e47bc367393d546f', 'a60e534fa2375098', '31bd4a42981c5a1a', '74dc0108320553fc', '7760e889babb5568', 'e184eaa8a75f528c', 
 'a9a53744b08659b3', '10e628dc19da575b', 'bfab40e7d86552c2', '17dfa7ec678255a2', '45aba0f487445607', 'b1e5692751db5c66', 
 'aa6c4599cfc8545a', '8379f8a7dcff5459', 'dac22faaab5c54fa', '7ae1fa9094f355c5', '0076db3c84715464', 'a5f1aad3dd9555fb', 
 'f9f9a7d197a7562f', '3c8fa9885cbc55f2', '5d57954e734958cd', '21c4020486cb5a19', 'ba6b75a8853a55b5', '957b64e370ee51ab', 
 '72bf913f2d7f523a', '8427e856770c5a1a', '298dc64710b85e41', '93fd9b5bfef55864', '96c9afd31086542f', '3426203045cb5778', 
 '54d64bae86805fb3', 'f4b8870335a85a7a', 'b2ecf2ad84035ea1', '68c171e0c35a52cd', 'd854b5a7a6de5298', '18516d35c2df56a7', 
 'cf8e39c28de65c10', '6c927ca63e7a5977', '2682b658c66b5f7f', '130e725a1594571f', 'fc46de11a408576d', '6358d67c04ca54ce', 
 '0e7e77fbbaa150e0', '25ee32067ee65e75', '379cbef2d89e5149', 'af8d975bb1825617', '5d0eb074397f591b']


import argparse
import json
import os
import subprocess
import re
import random

import datasets
from verl.utils.hdfs_io import copy, makedirs

def make_map_fn(data_source_name: str, split: str='train'):
    def match_answer(text: str, mode: str = 'action1'):
        if mode == 'action1':
            pattern = r"<action_(\d+)>"
            matches = re.findall(pattern, text)
            return [int(match) for match in matches] # sequence of action id

        elif mode == 'text1':
            pattern = r'\(\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\s*\)'
            matches = re.findall(pattern, text)
            return [(float(x), float(y), float(z)) for x, y, z in matches] # sequence of waypoint (x, y, heading)

        else:
            print(f"Invalid mode: '{mode}'. Valid modes are 'action1' or 'text1'.")
            return ""

    def process_fn(example, idx):
        """The actual transformation function applied to each row."""
        # Pop original fields to ensure a clean final structure
        image_path = example.pop("image")
        system_prompt = example.pop("system", "")  # Use default if 'system' is missing
        conversations = example.pop("conversations")
        item_id = example.pop("id")

        # 1. Combine system prompt and conversations into a new 'prompt' list
        prompt_list = []
        answer = ""
        problem = ""
        if system_prompt and system_prompt.strip():
            prompt_list.append({"role": "system", "content": system_prompt.strip()})

        for turn in conversations:
            if turn["from"] == "gpt":
                content = turn["value"]
                answer = match_answer(content, mode='action1')
            elif turn["from"] == "human":
                role = "user"
                content = turn["value"]
                prompt_list.append({"role": role, "content": content})
                problem = content

        if isinstance(image_path, list):
            images_list = image_path
        else:
            images_list = [image_path]

        data = {
            "images": images_list,
            "problem": problem,
            "answer": {'token': item_id, 'gt': answer}
        }

        # Side effect: print the first processed item for verification
        if idx == 0:
            print("--- Example of first processed item ---")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("---------------------------------------")

        return data

    return process_fn


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a ShareGPT4V-style JSON to a structured Parquet format."
    )
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=True,
        help="Local path to the source ShareGPT4V-style JSON file.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default=None,
        help="Local path to the source ShareGPT4V-style JSON file.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/chencanyu-sh2/envs/verl/data/navsim_action",
        help="Local directory to save the output Parquet file(s).",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="[Optional] HDFS parent directory to upload the processed local_dir.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes to use for dataset mapping.",
    )

    args = parser.parse_args()


    # --- 1. Load the source JSON data ---
    
    if not os.path.exists(args.input_json_path):
        raise FileNotFoundError(f"Train file not found at {args.input_json_path}")

    print(f"Loading JSON data from: {args.input_json_path}")  
    with open(args.input_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    if args.test_json_path is None or not os.path.exists(args.test_json_path):
        # test_data = random.sample(train_data, 100)
        test_data = [item for item in train_data if item.get("id") in DEFAULT_TOEKN_LIST]
        # 确保测试集数量与DEFAULT_TOEKN_LIST匹配（可选）
        print(f"Filtered {len(test_data)} test items from DEFAULT_TOEKN_LIST")
    else:
        print(f"Loading JSON data from: {args.input_json_path}")
        with open(args.test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)


    print(f"Loaded TrainSet {len(train_data)} item.")
    print(f"Loaded TestSet {len(test_data)} items.")

    # --- 2. Convert to Hugging Face Dataset and apply transformation ---
    print("Converting list to a Hugging Face Dataset...")
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    train_data_source = os.path.basename(args.input_json_path)
    test_data_source = os.path.basename(args.test_json_path) if args.test_json_path is not None else train_data_source

    print(f"Processing dataset with data_source='{train_data_source}'...")
    
    train_dataset = train_dataset.map(
        function=make_map_fn(train_data_source, 'train'),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=train_dataset.column_names,
    )

    test_dataset = test_dataset.map(
        function=make_map_fn(test_data_source, 'test'),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=test_dataset.column_names, # Remove old columns after processing
    )
    
    print("Dataset successfully processed. New info:")
    print(train_dataset)

    # --- 3. Save to Parquet locally ---
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'data', "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'data', "test.parquet"))
 
    print("Successfully saved to Parquet format.")

    # --- 4. (Optional) Upload to HDFS ---
    if args.hdfs_dir:
        print("\nHDFS upload requested.")
        # Ensure the parent destination directory exists on HDFS
        makedirs(args.hdfs_dir)
        # Copy the entire local output directory to HDFS
        copy(src=args.local_dir, dst=args.hdfs_dir)





if __name__ == "__main__":
    # python navsim_action.py --input_json_path /chencanyu-sh2/VLA_train/annotation/QA_navsim_train_smart_sim_search_k12_0.99_actiononly.json --local_dir ./data/navsim_action_k12
    main()


