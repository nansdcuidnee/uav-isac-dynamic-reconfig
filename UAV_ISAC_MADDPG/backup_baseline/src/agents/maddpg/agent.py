import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.agents.maddpg.actor import Actor
from src.agents.maddpg.critic import Critic
from src.agents.maddpg.noise import OUNoise

# 处理PyTorch 2.9.1兼容性问题
try:
    import torch._dynamo
except ImportError:
    # 如果没有dynamo模块，跳过相关初始化
    pass

class DDPGAgent:
    """
    DDPG智能体
    - 单个无人机的智能体
    - 包含Actor和Critic网络
    """

    def __init__(self, obs_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.001, agent_idx=0, num_agents=3):
        """
        初始化DDPG智能体
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr_actor: Actor网络学习率
        :param lr_critic: Critic网络学习率
        :param gamma: 折扣因子
        :param tau: 目标网络软更新系数
        :param agent_idx: 智能体索引
        :param num_agents: 智能体数量
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.agent_idx = agent_idx
        self.num_agents = num_agents

        # Actor网络
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic网络
        self.critic = Critic(obs_dim * num_agents, action_dim * num_agents)
        self.target_critic = Critic(obs_dim * num_agents, action_dim * num_agents)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 创建噪声
        self.noise = OUNoise(action_dim)

        # 初始化目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def choose_action(self, obs, noise_std=0.1):
        """
        选择动作
        :param obs: 观察
        :param noise_std: 噪声标准差
        :return: 动作
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        action_probs = self.actor(obs).detach().numpy()
        
        # 添加噪声到概率分布
        noise = np.random.normal(0, noise_std, size=action_probs.shape)
        action_probs += noise
        action_probs = np.clip(action_probs, 0, 1)
        action_probs /= np.sum(action_probs)
        
        # 从概率分布中采样动作
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def soft_update(self):
        """
        软更新目标网络
        """
        # 软更新Actor网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 软更新Critic网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, path):
        """
        保存模型
        :param path: 保存路径
        """
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_checkpoint(self, path):
        """
        加载模型
        :param path: 加载路径
        """
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())