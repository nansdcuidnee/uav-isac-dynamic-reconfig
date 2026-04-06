import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor网络
    - 输入：观察
    - 输出：动作
    """

    def __init__(self, obs_dim, action_dim):
        """
        初始化Actor网络
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        """
        super(Actor, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)

    def forward(self, obs):
        """
        前向传播
        :param obs: 观察
        :return: 动作
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 输出动作，使用tanh激活函数将动作限制在[-1, 1]范围
        action = torch.tanh(self.fc4(x))
        return action
