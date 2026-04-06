import os
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.utils.video_utils import create_directory

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BaseVisualizer:
    """
    基础可视化器
    """
    def __init__(self):
        """
        初始化可视化器
        """
        pass
    
    def set_plot_style(self):
        """
        设置绘图样式
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
    
    def save_figure(self, fig, save_path):
        """
        保存图表
        :param fig: 图表对象
        :param save_path: 保存路径
        """
        create_directory(os.path.dirname(save_path) if os.path.dirname(save_path) else '.')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    def create_figure(self, figsize=(12, 8)):
        """
        创建图表
        :param figsize: 图表大小
        :return: 图表对象
        """
        self.set_plot_style()
        return plt.figure(figsize=figsize)
    
    def close_figure(self):
        """
        关闭图表
        """
        plt.close()
        plt.clf()
        plt.close('all')