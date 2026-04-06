import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    ISAC Critic网络
    - 输入：观察和动作（包含感知信息）
    - 输出：Q值
    """

    def __init__(self, obs_dim, action_dim):
        """
        初始化Critic网络
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        """
        super(Critic, self).__init__()

        # 网络结构 - 增强版，添加感知辅助通信层
        self.fc1 = nn.Linear(obs_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # 感知辅助层
        self.sensing_fc = nn.Linear(128, 64)
        # Q值输出层
        self.q_fc = nn.Linear(64, 1)

    def forward(self, obs, action):
        """
        前向传播
        :param obs: 观察
        :param action: 动作
        :return: Q值
        """
        # 拼接观察和动作
        x = torch.cat([obs, action], dim=1)
        # 特征提取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 感知辅助处理
        sensing_feature = F.relu(self.sensing_fc(x))
        # 输出Q值
        q_value = self.q_fc(sensing_feature)
        return q_value