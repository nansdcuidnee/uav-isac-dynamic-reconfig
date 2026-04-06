import torch
import numpy as np
from collections import deque
import random

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
        self.buffer = deque(maxlen=buffer_size)

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
        # 对于单智能体情况，确保obs和obs_是单个元素而不是列表
        if isinstance(obs, list) and len(obs) == 1:
            obs = obs[0]
            obs_ = obs_[0]
            rewards = rewards[0] if isinstance(rewards, list) else rewards
            dones = dones[0] if isinstance(dones, list) else dones
        
        experience = (obs, actions, rewards, obs_, dones)
        self.buffer.append(experience)

    def sample(self):
        """
        从缓冲区中采样经验
        :return: 采样的经验
        """
        experiences = random.sample(self.buffer, k=self.batch_size)

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
