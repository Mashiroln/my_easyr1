import orjson
import sys
from tqdm import tqdm
from typing import Callable, List, Tuple, Any, Dict

class JsonlProcessor:
    """
    一个用于处理 JSONL 文件的链式操作类。

    允许你注册一系列的过滤、键重命名和值修改操作，
    然后一次性运行它们。
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        初始化处理器。

        Args:
            input_path: 输入的 .jsonl 文件路径。
            output_path: 输出的 .jsonl 文件路径。
        """
        self.input_path = input_path
        self.output_path = output_path
        # 操作流水线，存储 (类型, 函数/参数)
        self.operations: List[Tuple[str, Any]] = []

    def filter(self, filter_func: Callable[[Dict], bool]) -> 'JsonlProcessor':
        """
        注册一个“过滤”函数。
        
        函数应接收一个 dict (行数据)，如果应保留该行，则返回 True，
        如果应丢弃该行，则返回 False。

        Args:
            filter_func: 过滤函数 (e.g., my_filter(row) -> True)
        
        Returns:
            self, 以便链式调用。
        """
        if not callable(filter_func):
            raise TypeError("filter_func 必须是一个可调用函数。")
        self.operations.append(('filter', filter_func))
        return self

    def replace_key(self, old_key: str, new_key: str) -> 'JsonlProcessor':
        """
        注册一个“替换键名”操作。
        如果 old_key 存在，它将被重命名为 new_key。

        Args:
            old_key: 原始键名。
            new_key: 新的键名。

        Returns:
            self, 以便链式调用。
        """
        self.operations.append(('replace_key', (old_key, new_key)))
        return self

    def modify_value(self, key: str, modifier_func: Callable[[Any], Any]) -> 'JsonlProcessor':
        """
        注册一个“修改特定键值”的操作。
        
        函数将接收 key 对应的 *旧值*，并应返回 *新值*。
        如果 key 不存在，此操作将被跳过。

        Args:
            key: 要修改的键。
            modifier_func: 修改函数 (e.g., lambda x: x * 100)

        Returns:
            self, 以便链式调用。
        """
        if not callable(modifier_func):
            raise TypeError("modifier_func 必须是一个可调用函数。")
        
        # 内部将其转换为一个 modify_row 操作
        def _row_modifier(row: Dict) -> Dict:
            if key in row:
                row[key] = modifier_func(row[key])
            return row
            
        self.operations.append(('modify_row', _row_modifier))
        return self
        
    def modify_row(self, modifier_func: Callable[[Dict], Dict]) -> 'JsonlProcessor':
        """
        注册一个“修改整行”的操作（最灵活）。
        
        函数将接收 *整行 dict*，并应返回 *修改后的 dict*。
        可用于添加/删除键，或基于多个键的值进行复杂修改。

        Args:
            modifier_func: 行修改函数 (e.g., my_row_func(row) -> modified_row)

        Returns:
            self, 以便链式调用。
        """
        if not callable(modifier_func):
            raise TypeError("modifier_func 必须是一个可调用函数。")
        self.operations.append(('modify_row', modifier_func))
        return self

    def run(self):
        """
        执行所有已注册的操作，处理文件。
        """
        total_lines = 0
        kept_lines = 0
        error_lines = 0
        
        print(f"--- JsonlProcessor 启动 ---")
        print(f"输入: {self.input_path}")
        print(f"输出: {self.output_path}")
        print(f"注册了 {len(self.operations)} 个操作。")
        
        try:
            with open(self.input_path, 'rb') as fin, open(self.output_path, 'wb') as fout:
                
                pbar = tqdm(fin, desc=f"Processing {self.input_path}", unit=" lines", unit_scale=True)

                for line in pbar:
                    total_lines += 1
                    try:
                        # 1. 读取
                        data = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        print(f"Skipping line {total_lines} due to JSON decode error.", file=sys.stderr)
                        error_lines += 1
                        continue
                    
                    # 2. 按顺序执行操作流水线
                    keep_this_line = True
                    
                    for op_type, op_payload in self.operations:
                        try:
                            if op_type == 'filter':
                                if not op_payload(data):
                                    keep_this_line = False
                                    break  # 停止处理此行
                            
                            elif op_type == 'replace_key':
                                old_key, new_key = op_payload
                                if old_key in data:
                                    data[new_key] = data.pop(old_key)
                                    
                            elif op_type == 'modify_row':
                                data = op_payload(data)
                                if not isinstance(data, dict):
                                    # 保护措施，防止用户函数返回非 dict
                                    print(f"Skipping line {total_lines}: modify_row 函数未返回 dict。", file=sys.stderr)
                                    keep_this_line = False
                                    break
                                    
                        except Exception as e:
                            print(f"Skipping line {total_lines} due to operation error: {e}", file=sys.stderr)
                            keep_this_line = False
                            break # 停止处理此行

                    # 3. 写入
                    if keep_this_line:
                        fout.write(orjson.dumps(data) + b"\n")
                        kept_lines += 1
                        
        except FileNotFoundError:
            print(f"错误: 输入文件 '{self.input_path}' 未找到。", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"发生意外的文件错误: {e}", file=sys.stderr)
            sys.exit(1)

        print("\n--- 筛选完成 ---")
        print(f"总共处理行数: {total_lines}")
        print(f"保留行数 (Kept): {kept_lines}")
        print(f"过滤行数 (Filtered): {total_lines - kept_lines - error_lines}")
        print(f"格式错误行数 (Errors): {error_lines}")
        print(f"成功写入到: {self.output_path}")