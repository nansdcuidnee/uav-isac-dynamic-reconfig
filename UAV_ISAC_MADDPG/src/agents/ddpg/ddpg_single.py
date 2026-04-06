import numpy as np
import torch
import torch.nn.functional as F
from src.agents.ddpg.actor import Actor
from src.agents.ddpg.critic import Critic
from src.agents.ddpg.noise import OUNoise
from src.agents.ddpg.replay_buffer import ReplayBuffer

class DDPG:
    """
    单智能体DDPG算法
    - 适用于连续动作空间
    """

    def __init__(self, obs_dim, action_dim, lr_actor=0.0001, lr_critic=0.0005, gamma=0.99, tau=0.001, batch_size=128, buffer_size=2000000):
        """
        初始化DDPG算法
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr_actor: Actor网络学习率
        :param lr_critic: Critic网络学习率
        :param gamma: 折扣因子
        :param tau: 目标网络软更新系数
        :param batch_size: 批次大小
        :param buffer_size: 经验回放缓冲区大小
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor网络
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic网络
        self.critic = Critic(obs_dim, action_dim)
        self.target_critic = Critic(obs_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # 噪声
        self.noise = OUNoise(action_dim)

        # 目标网络更新频率
        self.target_update_freq = 2
        self.learn_step = 0

        # 初始化目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def choose_action(self, obs, noise_std=0.1):
        """
        选择动作
        :param obs: 观察
        :param noise_std: 噪声标准差
        :return: 动作
        """
        # 处理不同类型的观察值
        if isinstance(obs, list):
            # 如果是列表，尝试展平或获取第一个元素
            if len(obs) > 0 and isinstance(obs[0], (list, np.ndarray)):
                # 如果是嵌套列表，取第一个元素
                obs = obs[0]
            obs = np.array(obs, dtype=np.float32)
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # 确保obs是一维数组
        obs = obs.flatten()
        
        obs = torch.tensor(obs, dtype=torch.float32)
        action = self.actor(obs).detach().numpy()
        # 添加噪声
        action += self.noise.sample() * noise_std
        # 裁剪动作到[-1, 1]范围
        action = np.clip(action, -1, 1)
        return action

    def learn(self):
        """
        学习
        :return: critic_loss, actor_loss, total_loss
        """
        if len(self.memory) < self.batch_size:
            return 0, 0, 0

        # 从经验回放缓冲区中采样
        obs, actions, rewards, next_obs, dones = self.memory.sample()

        # 计算目标Q值
        target_actions = self.target_actor(next_obs)
        target_q = self.target_critic(next_obs, target_actions)
        y = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        # 计算预期Q值
        expected_q = self.critic(obs, actions)

        # 计算Critic损失
        critic_loss = torch.nn.functional.mse_loss(expected_q, y)

        # 优化Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 计算Actor损失
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        # 优化Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 增加学习步数
        self.learn_step += 1
        # 延迟更新目标网络
        if self.learn_step % self.target_update_freq == 0:
            self.soft_update()

        total_loss = (critic_loss + actor_loss).item()
        return critic_loss.item(), actor_loss.item(), total_loss

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
        # 对于单智能体情况，确保obs和obs_是单个元素而不是列表
        if isinstance(obs, list) and len(obs) == 1:
            obs = obs[0]
            obs_ = obs_[0]
            rewards = rewards[0] if isinstance(rewards, list) else rewards
            dones = dones[0] if isinstance(dones, list) else dones
        
        self.memory.add(obs, obs_full, actions, rewards, obs_, obs_full_, dones)

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
