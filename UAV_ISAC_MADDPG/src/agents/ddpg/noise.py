import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck噪声
    - 用于DDPG算法的探索
    - 产生具有时间相关性的噪声
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        """
        初始化OU噪声
        :param action_dim: 动作维度
        :param mu: 均值
        :param theta: 均值回归系数
        :param sigma: 噪声标准差
        """
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        重置噪声
        """
        self.state = np.copy(self.mu)

    def sample(self):
        """
        采样噪声
        :return: 噪声
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
