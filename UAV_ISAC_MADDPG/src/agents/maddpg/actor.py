import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    ISAC Actor网络
    - 输入：观察（包含感知信息）
    - 输出：动作概率分布
    """

    def __init__(self, obs_dim, action_dim):
        """
        初始化Actor网络
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        """
        super(Actor, self).__init__()

        # 网络结构 - 增强版，添加感知辅助通信层
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # 感知辅助层
        self.sensing_fc = nn.Linear(128, 64)
        # 通信决策层
        self.comm_fc = nn.Linear(64, action_dim)

    def forward(self, obs):
        """
        前向传播
        :param obs: 观察
        :return: 动作概率分布
        """
        # 特征提取
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # 感知辅助处理
        sensing_feature = F.relu(self.sensing_fc(x))
        # 输出动作概率分布，使用softmax激活函数
        action_probs = F.softmax(self.comm_fc(sensing_feature), dim=-1)
        return action_probs