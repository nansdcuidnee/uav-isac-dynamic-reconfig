import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Critic网络
    - 输入：观察和动作
    - 输出：Q值
    """

    def __init__(self, obs_dim, action_dim):
        """
        初始化Critic网络
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        """
        super(Critic, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, obs, action):
        """
        前向传播
        :param obs: 观察
        :param action: 动作
        :return: Q值
        """
        # 拼接观察和动作
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 输出Q值
        q_value = self.fc4(x)
        return q_value
