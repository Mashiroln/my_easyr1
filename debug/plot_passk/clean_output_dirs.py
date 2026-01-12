
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# ==============================================================================
# ======================== 用户配置区 (USER CONFIGURATION) ========================
# ==============================================================================

# 1. 指定包含要保留的token的列表文件
#    这个文件里的token对应的文件夹将被保留。
TOKEN_LIST_TO_KEEP = "/mnt/data/ccy/EasyR1/debug/plot_passk/selected_tokens.txt"

# 2. 指定包含所有token子文件夹的总目录
#    脚本将在这个目录下进行清理。
BASE_VISUALIZATION_DIR = "/mnt/data/ccy/EasyR1/debug/plot_passk/"

# ==============================================================================
# =============================== 脚本主体 (SCRIPT BODY) ===============================
# ==============================================================================

def cleanup_unselected_directories(base_dir_path: str, keep_list_path: str):
    """
    清理基础目录，只保留在'keep_list'中指定的子目录。
    """
    
    base_dir = Path(base_dir_path)
    keep_list_file = Path(keep_list_path)

    # --- 1. 检查路径是否存在 ---
    if not base_dir.is_dir():
        print(f"[ERROR] Base directory not found: {base_dir}")
        return
    if not keep_list_file.is_file():
        print(f"[ERROR] Token list file to keep not found: {keep_list_file}")
        return
        
    # --- 2. 加载要保留的token列表 ---
    try:
        with open(keep_list_file, 'r') as f:
            # 使用set以便进行快速查找 (O(1)复杂度)
            tokens_to_keep = {line.strip() for line in f if line.strip()}
        print(f"Successfully loaded {len(tokens_to_keep)} tokens to keep from {keep_list_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to read token list file. Error: {e}")
        return

    # --- 3. 识别所有存在的目录和待删除的目录 ---
    dirs_to_delete = []
    
    # 遍历基础目录下的所有项目
    for item in base_dir.iterdir():
        # 确保我们只处理文件夹，并且文件夹的名字不在保留列表中
        if item.is_dir() and item.name not in tokens_to_keep:
            # 排除预处理数据文件夹等特殊文件夹
            if item.name not in ["preprocessed_data", "visualization_outputs"]:
                dirs_to_delete.append(item)

    if not dirs_to_delete:
        print("\nAll existing directories are in the keep list. Nothing to delete. Cleanup is complete.")
        return

    # --- 4. 打印总结并请求用户最终确认 ---
    print("\n" + "="*80)
    print(" " * 30 + "!!! WARNING: DESTRUCTIVE ACTION !!!")
    print("="*80)
    print(f"This script will permanently delete {len(dirs_to_delete)} subdirectories from:")
    print(f"  {base_dir}")
    print("\nThis action CANNOT be undone.")
    
    # 显示一些将要删除的目录作为示例
    print("\nExamples of directories to be deleted:")
    for dir_to_delete in dirs_to_delete[:5]:
        print(f"  - {dir_to_delete.name}")
    if len(dirs_to_delete) > 5:
        print("  - ... and more")

    print("\n" + "="*80)
    
    try:
        confirm = input("Are you absolutely sure you want to proceed? Type 'yes' to continue: ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

    # --- 5. 执行删除 ---
    if confirm.lower() == 'yes':
        print("\nConfirmation received. Starting deletion process...")
        for dir_path in tqdm(dirs_to_delete, desc="Deleting directories"):
            try:
                shutil.rmtree(dir_path)
            except OSError as e:
                print(f"\n[ERROR] Failed to delete {dir_path}: {e}")
        print(f"\nCleanup complete. Successfully deleted {len(dirs_to_delete)} directories.")
    else:
        print("\nOperation cancelled. No directories were deleted.")

if __name__ == "__main__":
    cleanup_unselected_directories(
        base_dir_path=BASE_VISUALIZATION_DIR,
        keep_list_path=TOKEN_LIST_TO_KEEP
    )