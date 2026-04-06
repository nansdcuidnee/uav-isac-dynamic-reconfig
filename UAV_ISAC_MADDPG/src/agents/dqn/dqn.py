import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQN:
    """
    DQN算法
    - 基于值函数的强化学习算法
    - 适用于离散动作空间
    """

    def __init__(self, num_agents, obs_dim, action_dim, lr=0.0003, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, batch_size=128, buffer_size=1000000):
        """
        初始化DQN算法
        :param num_agents: 智能体数量
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr: 学习率
        :param gamma: 折扣因子
        :param epsilon_start: 初始探索率
        :param epsilon_end: 最终探索率
        :param epsilon_decay: 探索率衰减系数
        :param batch_size: 批次大小
        :param buffer_size: 经验回放缓冲区大小
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # 创建智能体
        self.agents = []
        for i in range(num_agents):
            agent = DQNAgent(obs_dim, action_dim, lr, gamma)
            self.agents.append(agent)

        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # 目标网络更新频率
        self.target_update_freq = 100  # 每 100 步更新一次
        self.learn_step = 0

    def choose_action(self, obs, noise_std=0.1):
        """
        选择动作
        :param obs: 观察
        :param noise_std: 噪声标准差（用于与其他算法保持接口一致）
        :return: 动作
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i], self.epsilon)
            actions.append(action)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return actions

    def learn(self):
        """
        学习
        :return: 损失
        """
        if len(self.memory) < self.batch_size:
            return 0

        total_loss = 0
        # 每次 learn 更新 10 次
        for _ in range(10):
            # 从经验回放缓冲区中采样
            experiences = self.memory.sample()
            obs, actions, rewards, next_obs, dones = experiences

            # 计算目标Q值
            target_q = []
            for i, agent in enumerate(self.agents):
                max_next_q = agent.target_q_network(next_obs[:, i]).max(1)[0].unsqueeze(1)
                target = rewards[:, i].unsqueeze(1) + self.gamma * max_next_q * (1 - dones[:, i].unsqueeze(1))
                target_q.append(target)
            target_q = torch.cat(target_q, dim=1)

            # 计算预期Q值
            expected_q = []
            for i, agent in enumerate(self.agents):
                q_values = agent.q_network(obs[:, i])
                action_indices = actions[:, i].long().unsqueeze(1)
                expected = q_values.gather(1, action_indices)
                expected_q.append(expected)
            expected_q = torch.cat(expected_q, dim=1)

            # 计算损失
            loss = 0
            for i in range(self.num_agents):
                loss += F.mse_loss(expected_q[:, i].unsqueeze(1), target_q[:, i].unsqueeze(1))

            # 优化网络
            for agent in self.agents:
                agent.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            for agent in self.agents:
                torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
                agent.optimizer.step()

            total_loss += loss.item()

        # 硬更新目标网络
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            for agent in self.agents:
                agent.target_q_network.load_state_dict(agent.q_network.state_dict())

        return total_loss / 10

    def add_experience(self, obs, obs_full, actions, rewards, obs_, obs_full_, dones):
        """
        添加经验到回放缓冲区
        :param obs: 观察
        :param obs_full: 完整观察
        :param actions: 动作
        :param rewards: 奖励
        :param obs_: 下一观察
        :param obs_full_: 下一完整观察
        :param dones: 完成标志
        """
        self.memory.add(obs, obs_full, actions, rewards, obs_, obs_full_, dones)

    def save_checkpoint(self, path):
        """
        保存模型
        :param path: 保存路径
        """
        for i, agent in enumerate(self.agents):
            agent.save_checkpoint(f"{path}_agent{i}")

    def load_checkpoint(self, path):
        """
        加载模型
        :param path: 加载路径
        """
        for i, agent in enumerate(self.agents):
            agent.load_checkpoint(f"{path}_agent{i}")


class DQNAgent:
    """
    DQN智能体
    - 单个无人机的智能体
    - 包含Q网络和目标Q网络
    """

    def __init__(self, obs_dim, action_dim, lr=0.001, gamma=0.99):
        """
        初始化DQN智能体
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr: 学习率
        :param gamma: 折扣因子
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Q网络
        self.q_network = QNetwork(obs_dim, action_dim)
        self.target_q_network = QNetwork(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # 初始化目标网络参数
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, obs, epsilon=0.1):
        """
        选择动作
        :param obs: 观察
        :param epsilon: 探索率
        :return: 动作
        """
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(0, self.action_dim)
        else:
            # 贪婪选择
            obs = torch.as_tensor(obs, dtype=torch.float32)
            was_training = self.q_network.training
            self.q_network.eval()
            try:
                with torch.no_grad():
                    q_values = self.q_network(obs)
            finally:
                self.q_network.train(was_training)
            return q_values.argmax().item()



    def save_checkpoint(self, path):
        """
        保存模型
        :param path: 保存路径
        """
        torch.save(self.q_network.state_dict(), f"{path}_q_network.pth")

    def load_checkpoint(self, path):
        """
        加载模型
        :param path: 加载路径
        """
        self.q_network.load_state_dict(torch.load(f"{path}_q_network.pth"))
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class QNetwork(nn.Module):
    """
    Q网络
    - 输入：观察
    - 输出：每个动作的Q值
    """

    def __init__(self, obs_dim, action_dim):
        """
        初始化Q网络
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        """
        super(QNetwork, self).__init__()

        # 增强网络结构
        self.fc1 = nn.Linear(obs_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, obs):
        """
        前向传播
        :param obs: 观察
        :return: Q值
        """
        # 确保输入是 2D 张量
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = F.relu(self.bn1(self.fc1(obs)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        # 如果输入是单样本，返回 1D 张量
        if obs.size(0) == 1:
            q_values = q_values.squeeze(0)
        return q_values


class ReplayBuffer:
    """
    经验回放缓冲区
    - 用于存储智能体的经验
    - 支持随机采样
    """

    def __init__(self, buffer_size, batch_size):
        """
        初始化经验回放缓冲区
        :param buffer_size: 缓冲区大小
        :param batch_size: 批次大小
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def add(self, obs, obs_full, actions, rewards, obs_, obs_full_, dones):
        """
        添加经验到缓冲区
        :param obs: 观察
        :param obs_full: 完整观察
        :param actions: 动作
        :param rewards: 奖励
        :param obs_: 下一观察
        :param obs_full_: 下一完整观察
        :param dones: 完成标志
        """
        # 只存储必要的数据，减少内存占用
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        # 存储为元组，避免额外的对象开销
        self.buffer[self.position] = (obs, actions, rewards, obs_, dones)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """
        从缓冲区中采样经验
        :return: 采样的经验
        """
        # 确保缓冲区有足够的经验
        if len(self.buffer) < self.batch_size:
            # 如果经验不足，重复采样现有经验
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        
        experiences = [self.buffer[i] for i in indices]

        # 转换为张量
        obs = torch.tensor([e[0] for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.float32)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32)
        next_obs = torch.tensor([e[3] for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        """
        返回缓冲区的长度
        :return: 缓冲区长度
        """
        return len(self.buffer)
