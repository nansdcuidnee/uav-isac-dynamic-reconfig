import numpy as np
import torch
import torch.nn.functional as F
from src.agents.maddpg.agent import DDPGAgent
from src.agents.maddpg.replay_buffer import MultiAgentReplayBuffer

class MADDPG:
    """
    MADDPG（多智能体深度确定性策略梯度）算法
    - 多智能体管理
    - 集中式训练，分布式执行
    """

    def __init__(self, num_agents, obs_dim, action_dim, lr_actor=0.001, lr_critic=0.001, 
                 gamma=0.95, tau=0.01, batch_size=64, memory_size=1000000):
        """
        初始化MADDPG
        :param num_agents: 智能体数量
        :param obs_dim: 观察维度
        :param action_dim: 动作维度
        :param lr_actor: Actor网络学习率
        :param lr_critic: Critic网络学习率
        :param gamma: 折扣因子
        :param tau: 目标网络软更新系数
        :param batch_size: 批量大小
        :param memory_size: 经验回放缓冲区大小
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
            agent = DDPGAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                agent_idx=i,
                num_agents=num_agents
            )
            self.agents.append(agent)

        # 创建经验回放缓冲区
        self.memory = MultiAgentReplayBuffer(
            memory_size=memory_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            batch_size=batch_size
        )

    def choose_action(self, obs_list, noise_std=0.1):
        """
        选择动作
        :param obs_list: 观察列表
        :param noise_std: 噪声标准差
        :return: 动作列表
        """
        actions = []
        for i, obs in enumerate(obs_list):
            action = self.agents[i].choose_action(obs, noise_std)
            actions.append(action)
        return actions

    def learn(self):
        """
        学习
        :return: 损失值
        """
        if len(self.memory) < self.batch_size:
            return 0

        # 从经验回放缓冲区中采样
        obs, obs_full, actions, rewards, obs_, obs_full_, dones = self.memory.sample()

        # 计算目标Q值
        target_actions = []
        for i in range(self.num_agents):
            target_action_probs = self.agents[i].target_actor(obs_[:, i])
            # 从概率分布中选择动作
            target_action = torch.argmax(target_action_probs, dim=1).unsqueeze(1)
            # 转换为one-hot编码
            target_action_one_hot = torch.zeros(target_action.shape[0], self.action_dim)
            target_action_one_hot.scatter_(1, target_action, 1)
            target_actions.append(target_action_one_hot)
        target_actions = torch.cat(target_actions, dim=1)

        # 计算每个智能体的损失
        total_loss = 0
        for i in range(self.num_agents):
            # 计算目标Q值
            with torch.no_grad():
                target_q_val = self.agents[i].target_critic(obs_full_.clone(), target_actions.clone())
                target_q = rewards[:, i].view(-1, 1) + self.gamma * target_q_val * (1 - dones[:, i].view(-1, 1))

            # 计算当前Q值
            current_q = self.agents[i].critic(obs_full.clone(), actions.clone())

            # 计算Critic损失
            critic_loss = F.mse_loss(current_q, target_q)

            # 优化Critic网络
            self.agents[i].critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=False)
            self.agents[i].critic_optimizer.step()

            # 计算Actor损失
            # 构建当前智能体的动作
            agent_action_probs = self.agents[i].actor(obs[:, i])
            # 从概率分布中选择动作
            agent_action = torch.argmax(agent_action_probs, dim=1).unsqueeze(1)
            # 转换为one-hot编码
            agent_action_one_hot = torch.zeros(agent_action.shape[0], self.action_dim)
            agent_action_one_hot.scatter_(1, agent_action, 1)
            # 创建一个全新的actions张量，避免inplace操作
            actor_actions = []
            for j in range(self.num_agents):
                if j == i:
                    actor_actions.append(agent_action_one_hot)
                else:
                    actor_actions.append(actions[:, j*self.action_dim:(j+1)*self.action_dim].clone())
            actor_actions = torch.cat(actor_actions, dim=1)
            # 计算Actor损失
            actor_loss = -self.agents[i].critic(obs_full.clone(), actor_actions).mean()

            # 优化Actor网络
            self.agents[i].actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            self.agents[i].actor_optimizer.step()

            # 软更新目标网络
            self.agents[i].soft_update()

            total_loss += critic_loss.item() + actor_loss.item()

        return total_loss / self.num_agents

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