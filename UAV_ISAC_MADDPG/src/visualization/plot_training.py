import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(score_history, coverage_history, loss_history, save_path=None):
    """
    绘制训练曲线
    :param score_history: 分数历史
    :param coverage_history: 覆盖率历史
    :param loss_history: 损失历史
    :param save_path: 保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制分数曲线
    plt.subplot(1, 3, 1)
    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # 绘制覆盖率曲线
    plt.subplot(1, 3, 2)
    plt.plot(coverage_history)
    plt.title('Coverage History')
    plt.xlabel('Episode')
    plt.ylabel('Coverage Rate')
    
    # 绘制损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练曲线已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()
