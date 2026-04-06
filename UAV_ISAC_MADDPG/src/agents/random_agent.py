import numpy as np

class RandomAgent:
    """
    随机智能体 - 作为对比实验的基线
    """
    
    def __init__(self, num_drones=3, action_dim=8):
        """
        初始化随机智能体
        :param num_drones: 无人机数量
        :param action_dim: 动作空间维度
        """
        self.num_drones = num_drones
        self.action_dim = action_dim
    
    def select_action(self, obs):
        """
        随机选择动作
        :param obs: 观察状态
        :return: 随机动作
        """
        # 对于多无人机，返回多个随机动作
        if self.num_drones > 1:
            return np.random.uniform(-1, 1, self.num_drones)
        # 对于单无人机，返回单个随机动作
        return np.random.uniform(-1, 1)
    
    def update(self, *args, **kwargs):
        """
        随机智能体不需要更新
        """
        pass
    
    def save(self, *args, **kwargs):
        """
        随机智能体不需要保存
        """
        pass
    
    def load(self, *args, **kwargs):
        """
        随机智能体不需要加载
        """
        pass
