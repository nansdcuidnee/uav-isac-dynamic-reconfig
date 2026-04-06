import os
import yaml
import numpy as np

def create_directory(path):
    """
    创建目录
    :param path: 目录路径
    """
    os.makedirs(path, exist_ok=True)
    print(f"目录已创建: {path}")

def load_config(config_path):
    """
    加载配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results, save_path):
    """
    保存结果
    :param results: 结果字典
    :param save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 根据文件扩展名选择保存方式
    if save_path.endswith('.npy'):
        np.save(save_path, results)
    elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True)
    elif save_path.endswith('.txt'):
        with open(save_path, 'w', encoding='utf-8') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    else:
        print(f"不支持的文件格式: {save_path}")
    
    print(f"结果已保存到: {save_path}")
