import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mean_std(data_list, labels, xlabel='Episode', ylabel='Value', title='Mean ± Std', save_path=None):
    """
    绘制均值±标准差曲线
    :param data_list: 数据列表
    :param labels: 标签列表
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param title: 标题
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    for i, data in enumerate(data_list):
        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(mean, label=labels[i])
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    else:
        plt.show()
    plt.close()
