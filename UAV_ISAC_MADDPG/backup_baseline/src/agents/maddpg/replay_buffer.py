import numpy as np
import torch

class MultiAgentReplayBuffer:
    """
    多智能体经验回放缓冲区
    - 存储和采样经验
    - 支持多智能体经验
    """

    def __init__(self, memory_size, obs_dim, action_dim, num_agents, batch_size):
        """
        初始化多智能体经验回放缓冲区
        :param memory_size: 缓冲区大小
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param num_agents: 智能体数量
        :param batch_size: 批量大小
        """
        self.memory_size = memory_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.batch_size = batch_size

        # 初始化缓冲区
        self.obs = np.zeros((memory_size, num_agents, obs_dim), dtype=np.float32)
        self.obs_full = np.zeros((memory_size, num_agents * obs_dim), dtype=np.float32)
        self.actions = np.zeros((memory_size, num_agents * action_dim), dtype=np.float32)
        self.rewards = np.zeros((memory_size, num_agents), dtype=np.float32)
        self.obs_ = np.zeros((memory_size, num_agents, obs_dim), dtype=np.float32)
        self.obs_full_ = np.zeros((memory_size, num_agents * obs_dim), dtype=np.float32)
        self.dones = np.zeros((memory_size, num_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, obs_full, actions, rewards, obs_, obs_full_, dones):
        """
        添加经验到缓冲区
        :param obs: 观察
        :param obs_full: 完整观察
        :param actions: 动作
        :param rewards: 奖励
        :param obs_: 下一观察
        :param obs_full_: 下一完整观察
        :param dones: 结束标志
        """
        self.obs[self.ptr] = obs
        self.obs_full[self.ptr] = obs_full
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.obs_[self.ptr] = obs_
        self.obs_full_[self.ptr] = obs_full_
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self):
        """
        采样经验
        :return: 采样的经验
        """
        indices = np.random.randint(0, self.size, size=self.batch_size)

        obs = torch.tensor(self.obs[indices], dtype=torch.float32)
        obs_full = torch.tensor(self.obs_full[indices], dtype=torch.float32)
        actions = torch.tensor(self.actions[indices], dtype=torch.float32)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32)
        obs_ = torch.tensor(self.obs_[indices], dtype=torch.float32)
        obs_full_ = torch.tensor(self.obs_full_[indices], dtype=torch.float32)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32)

        return obs, obs_full, actions, rewards, obs_, obs_full_, dones

    def __len__(self):
        """
        获取缓冲区大小
        :return: 缓冲区大小
        """
        return self.size