# 文件名: run_simple_filter.py
# 确保这个文件与 'jsonl_processor.py' 在同一目录

from jsonl import JsonlProcessor

if __name__ == "__main__":

    # --- 1. 定义你的操作函数 ---
    
    # 这是一个 "过滤" 函数。
    # 它接收一行数据 (row)。
    # 如果 row.get("intent") 的值 *不是* "unknown", 它返回 True (保留该行)。
    # 如果 row.get("intent") 的值 *是* "unknown", 它返回 False (过滤掉该行)。
    # def filter_unknown_intent(row: dict) -> bool:
    #     return row.get("intent") != "unknown"

    # --- 2. 配置和运行处理器 ---
    def filter_high_pdms(row: dict) -> bool:
        pdms = row.get("pdms", 0.0)
        return pdms >= 0.99  # 只保留 PDMS >= 0.99 的行
    
    
    INPUT_JSONL = "/mnt/data/ccy/EasyR1/debug/explorary_data/88step_augment_filter/augmented_norm_cot_text_88step_npu_policy_stats.jsonl"
    OUTPUT_JSONL = "/mnt/data/ccy/EasyR1/debug/explorary_data/88step_augment_filter/fileter_0.99.jsonl"
    
    # 实例化处理器
    processor = JsonlProcessor(INPUT_JSONL, OUTPUT_JSONL)
    
    # 注册你的过滤函数，然后运行
    processor.filter(filter_high_pdms).run()