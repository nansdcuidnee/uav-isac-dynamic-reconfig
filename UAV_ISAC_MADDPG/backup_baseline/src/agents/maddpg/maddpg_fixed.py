import numpy as np
import torch
import torch.nn.functional as F
from src.agents.maddpg.agent import DDPGAgent
from src.agents.maddpg.replay_buffer import MultiAgentReplayBuffer

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax采样
    :param logits: 模型输出的logits
    :param temperature: 温度参数
    :param hard: 是否返回one-hot编码
    :return: 采样的动作概率或one-hot向量
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
    y = logits + gumbel_noise
    y_soft = F.softmax(y / temperature, dim=-1)
    
    if hard:
        # 硬采样：返回one-hot向量
        index = torch.argmax(y_soft, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        # 软采样：返回概率分布
        return y_soft

class MADDPG:
    """
    MADDPG（多智能体深度确定性策略梯度）算法 - 修复版
    - 多智能体管理
    - 集中式训练，分布式执行
    """

    def __init__(self, num_agents, obs_dim, action_dim, lr_actor=0.001, lr_critic=0.001, 
                 gamma=0.99, tau=0.001, batch_size=128, memory_size=200000):
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
        
        # 学习步数计数器
        self.learn_step = 0
        # 目标网络更新频率
        self.target_update_freq = 2

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
        学习 - 修复版
        :return: 损失值
        """
        if len(self.memory) < self.batch_size:
            return 0

        # 从经验回放缓冲区中采样
        obs, obs_full, actions, rewards, obs_, obs_full_, dones = self.memory.sample()

        total_loss = 0
        
        for i in range(self.num_agents):
            # ========== Critic更新 ==========
            # 计算目标Q值
            with torch.no_grad():
                # 使用目标Actor计算下一状态的动作（使用Gumbel-Softmax采样）
                target_actions = []
                for j in range(self.num_agents):
                    logits = self.agents[j].target_actor(obs_[:, j])
                    # 使用Gumbel-Softmax采样（硬采样，返回one-hot向量）
                    action_one_hot = gumbel_softmax(logits, temperature=0.5, hard=True)
                    target_actions.append(action_one_hot)
                target_actions = torch.cat(target_actions, dim=1)
                
                # 计算目标Q值
                target_q_val = self.agents[i].target_critic(obs_full_, target_actions)
                target_q = rewards[:, i].view(-1, 1) + self.gamma * target_q_val * (1 - dones[:, i].view(-1, 1))

            # 计算当前Q值
            current_q = self.agents[i].critic(obs_full, actions)

            # 计算Critic损失
            critic_loss = F.mse_loss(current_q, target_q)

            # 优化Critic网络
            self.agents[i].critic_optimizer.zero_grad()
            critic_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic.parameters(), 1.0)
            self.agents[i].critic_optimizer.step()

            # ========== Actor更新 ==========
            # 计算当前智能体的动作（使用Gumbel-Softmax采样）
            logits = self.agents[i].actor(obs[:, i])
            # 使用Gumbel-Softmax采样（软采样，保持梯度）
            actor_action = gumbel_softmax(logits, temperature=0.5, hard=False)
            
            # 构建所有智能体的动作（当前智能体使用新计算的动作，其他使用原动作）
            actor_actions_list = []
            for j in range(self.num_agents):
                if j == i:
                    actor_actions_list.append(actor_action)
                else:
                    actor_actions_list.append(actions[:, j*self.action_dim:(j+1)*self.action_dim])
            actor_actions = torch.cat(actor_actions_list, dim=1)
            
            # 计算Actor损失（最大化Q值）
            actor_loss = -self.agents[i].critic(obs_full, actor_actions).mean()

            # 优化Actor网络
            self.agents[i].actor_optimizer.zero_grad()
            actor_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), 1.0)
            self.agents[i].actor_optimizer.step()

            total_loss += critic_loss.item() + actor_loss.item()
        
        # 软更新目标网络（延迟更新）
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            for i in range(self.num_agents):
                self.agents[i].soft_update()

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
