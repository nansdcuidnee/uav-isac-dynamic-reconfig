import os
import gc
import torch


def clean_memory():
    """
    清理系统内存和GPU内存
    """
    print("开始清理内存...")
    
    # 1. 清理Python垃圾
    print("\n1. 清理Python垃圾回收...")
    gc.collect()
    print("   垃圾回收完成，未使用对象已清理")
    
    # 2. 清理GPU内存（如果可用）
    if torch.cuda.is_available():
        print("\n2. 清理GPU内存...")
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        print("   GPU内存已清空")
        print(f"   当前GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("\n2. GPU不可用，跳过GPU内存清理")
    
    # 3. 尝试释放系统内存
    print("\n3. 尝试释放系统内存...")
    # 对于Windows系统，可以使用ctypes调用系统API
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetProcessWorkingSetSize(kernel32.GetCurrentProcess(), -1, -1)
        print("   系统内存已尝试释放")
    
    print("\n内存清理完成！")


if __name__ == "__main__":
    clean_memory()
