import json
import threading
import time
from collections import deque
from typing import Dict, Optional, List


class BatchJsonlLogger:
    _instances: Dict[str, 'BatchJsonlLogger'] = {}  # 单例池，按文件路径区分
    _lock = threading.Lock()  # 单例创建锁

    def __new__(cls, file_path: str, max_workers: int = 1, batch_size: int = 100, flush_interval: int = 5):
        """
        单例模式创建日志器，同一文件路径共享一个实例
        :param file_path: 日志文件路径
        :param max_workers: 写入线程数量（通常1个足够）
        :param batch_size: 批量写入阈值
        :param flush_interval: 定时刷新间隔(秒)
        """
        with cls._lock:
            if file_path not in cls._instances:
                cls._instances[file_path] = super().__new__(cls)
                # 初始化实例属性
                cls._instances[file_path]._init(
                    file_path=file_path,
                    max_workers=max_workers,
                    batch_size=batch_size,
                    flush_interval=flush_interval
                )
        return cls._instances[file_path]

    def _init(self, file_path: str, max_workers: int, batch_size: int, flush_interval: int):
        self.file_path = file_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = deque()  # 线程安全的日志队列
        self.queue_lock = threading.Lock()  # 队列操作锁
        self.running = True
        self.workers = []
        
        # 启动写入线程
        for _ in range(max_workers):
            worker = threading.Thread(target=self._write_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _write_worker(self):
        """后台写入线程逻辑"""
        while self.running:
            try:
                # 批量提取日志（最多batch_size条）
                batch = self._get_batch()
                if batch:
                    self._write_batch(batch)
                else:
                    # 队列空时休眠等待
                    time.sleep(self.flush_interval)
            except Exception as e:
                print(f"日志写入线程错误: {e}")
                time.sleep(1)  # 出错后休眠避免死循环

    def _get_batch(self) -> List[Dict]:
        """从队列提取一批日志"""
        batch = []
        with self.queue_lock:
            while len(batch) < self.batch_size and self.queue:
                batch.append(self.queue.popleft())
        return batch

    def _write_batch(self, batch: List[Dict]):
        """写入一批日志到文件"""
        with open(self.file_path, "a", encoding="utf-8") as f:
            for data in batch:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def write(self, data: Dict):
        """写入单条日志（线程安全）"""
        with self.queue_lock:
            self.queue.append(data)

    def flush(self):
        """强制刷新队列中所有日志到磁盘"""
        while True:
            batch = self._get_batch()
            if not batch:
                break
            self._write_batch(batch)

    def close(self):
        """关闭日志器（程序退出时调用）"""
        self.running = True  # 先停止线程循环
        self.flush()  # 强制刷新剩余日志
        # 等待所有工作线程结束
        for worker in self.workers:
            worker.join(timeout=5)
        # 从单例池移除
        with self._lock:
            if self.file_path in BatchJsonlLogger._instances:
                del BatchJsonlLogger._instances[self.file_path]

    def __del__(self):
        """对象销毁时自动关闭"""
        self.close()


# ----------------------使用示例----------------------
if __name__ == "__main__":
    # 全局唯一实例（相同路径会复用）
    logger = BatchJsonlLogger(
        file_path="/tmp/test_log.jsonl",
        batch_size=50,  # 每积累50条写入一次
        flush_interval=3  # 3秒内没满50条也会写入
    )
    
    # 在多线程中调用（示例）
    import concurrent.futures
    def test_write(i):
        logger.write({"id": i, "message": f"test log {i}"})
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(test_write, range(1000))
    
    # 程序退出前建议手动关闭（确保所有日志写入）
    logger.close()