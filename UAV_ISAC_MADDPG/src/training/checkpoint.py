import os
import torch

class Checkpoint:
    """
    检查点管理器
    - 保存和加载模型
    - 管理模型版本
    """

    def __init__(self, save_dir):
        """
        初始化检查点管理器
        :param save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

    def save(self, model, name):
        """
        保存模型
        :param model: 模型
        :param name: 模型名称
        """
        model_path = os.path.join(self.save_dir, 'models', f'{name}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")

    def load(self, model, name):
        """
        加载模型
        :param model: 模型
        :param name: 模型名称
        """
        model_path = os.path.join(self.save_dir, 'models', f'{name}.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"模型已加载从 {model_path}")
        else:
            print(f"模型文件不存在: {model_path}")

    def get_latest_checkpoint(self):
        """
        获取最新的检查点
        :return: 最新检查点的路径
        """
        models_dir = os.path.join(self.save_dir, 'models')
        if not os.path.exists(models_dir):
            return None

        files = os.listdir(models_dir)
        if not files:
            return None

        # 按修改时间排序
        files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        return os.path.join(models_dir, files[0])