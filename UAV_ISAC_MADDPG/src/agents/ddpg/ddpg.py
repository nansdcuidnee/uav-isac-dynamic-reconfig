import numpy as np
import torch
import torch.nn.functional as F
from src.agents.ddpg.actor import Actor
from src.agents.ddpg.critic import Critic
from src.agents.ddpg.noise import OUNoise
from src.agents.ddpg.replay_buffer import ReplayBuffer

class DDPG:
    """
    DDPG算法
    - 单智能体强化学习算法
    - 适用于连续动作空间
    """

    def __init__(self, num_agents, obs_dim, action_dim, lr_actor=0.0001, lr_critic=0.0005, gamma=0.99, tau=0.001, batch_size=128, buffer_size=2000000):
        """
        初始化DDPG算法
        :param num_agents: 智能体数量
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr_actor: Actor网络学习率
        :param lr_critic: Critic网络学习率
        :param gamma: 折扣因子
        :param tau: 目标网络软更新系数
        :param batch_size: 批次大小
        :param buffer_size: 经验回放缓冲区大小
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # 创建智能体
        self.agents = []
        for i in range(num_agents):
            agent = DDPGAgent(obs_dim, action_dim, lr_actor, lr_critic, gamma, tau, i, num_agents)
            self.agents.append(agent)

        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def choose_action(self, obs, noise_std=0.1):
        """
        选择动作
        :param obs: 观察
        :param noise_std: 噪声标准差
        :return: 动作
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i], noise_std)
            actions.append(action)
        return actions

    def learn(self):
        """
        学习
        :return: 损失
        """
        if len(self.memory) < self.batch_size:
            return 0

        # 从经验回放缓冲区中采样
        experiences = self.memory.sample()
        obs, actions, rewards, next_obs, dones = experiences

        # 处理单智能体情况
        if self.num_agents == 1:
            # 单智能体情况
            agent = self.agents[0]
            
            # 计算目标Q值
            target_action = agent.target_actor(next_obs)
            target_q = agent.target_critic(next_obs, target_action)
            y = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
            
            # 计算预期Q值
            expected_q = agent.critic(obs, actions)
            
            # 计算Critic损失
            critic_loss = torch.nn.functional.mse_loss(expected_q, y)
            
            # 优化Critic网络
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # 计算Actor损失
            actor_loss = -agent.critic(obs, agent.actor(obs)).mean()
            
            # 优化Actor网络
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # 软更新目标网络
            agent.soft_update()
            
            return (critic_loss + actor_loss).item()
        else:
            # 多智能体情况
            # 计算目标Q值
            target_actions = []
            for i, agent in enumerate(self.agents):
                target_action = agent.target_actor(next_obs[:, i])
                target_actions.append(target_action)
            target_actions = torch.cat(target_actions, dim=1)

            # 计算目标Q值
            target_q = []
            for i, agent in enumerate(self.agents):
                target_q_value = agent.target_critic(
                    next_obs.view(self.batch_size, -1),
                    target_actions
                )
                target_q.append(target_q_value)
            target_q = torch.cat(target_q, dim=1)

            # 计算预期Q值
            expected_q = []
            for i, agent in enumerate(self.agents):
                expected_q_value = agent.critic(
                    obs.view(self.batch_size, -1),
                    actions.view(self.batch_size, -1)
                )
                expected_q.append(expected_q_value)
            expected_q = torch.cat(expected_q, dim=1)

            # 计算Critic损失
            critic_loss = 0
            for i, agent in enumerate(self.agents):
                y = rewards[:, i].unsqueeze(1) + self.gamma * target_q[:, i].unsqueeze(1) * (1 - dones[:, i].unsqueeze(1))
                critic_loss += torch.nn.functional.mse_loss(expected_q[:, i].unsqueeze(1), y)

            # 优化Critic网络
            for agent in self.agents:
                agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            for agent in self.agents:
                agent.critic_optimizer.step()

            # 计算Actor损失
            actor_loss = 0
            for i, agent in enumerate(self.agents):
                current_actions = []
                for j, a in enumerate(self.agents):
                    if j == i:
                        current_action = agent.actor(obs[:, j])
                    else:
                        current_action = actions[:, j].detach()
                    current_actions.append(current_action)
                current_actions = torch.cat(current_actions, dim=1)
                actor_loss += -agent.critic(obs.view(self.batch_size, -1), current_actions).mean()

            # 优化Actor网络
            for agent in self.agents:
                agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            for agent in self.agents:
                agent.actor_optimizer.step()

            # 软更新目标网络
            for agent in self.agents:
                agent.soft_update()

            return (critic_loss + actor_loss).item()

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


class DDPGAgent:
    """
    DDPG智能体
    - 单个无人机的智能体
    - 包含Actor和Critic网络
    """

    def __init__(self, obs_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.001, agent_idx=0, num_agents=1):
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
        action = self.actor(obs).detach().numpy()
        # 添加噪声
        action += self.noise.sample() * noise_std
        # 裁剪动作到[-1, 1]范围
        action = np.clip(action, -1, 1)
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
