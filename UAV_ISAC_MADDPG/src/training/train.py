import os
import numpy as np
import torch
import sys
import json

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.envs.uav_env import UAVEnv
from src.agents.maddpg.maddpg import MADDPG
from src.agents.ddpg.ddpg_single import DDPG
from src.agents.dqn.dqn import DQN
from src.visualization.plot_training import plot_training_curves
from src.utils import load_config, create_directory, save_results
from src.training.trainer import Trainer

def train(config_path, save_dir, algorithm='ddpg', num_drones=None, trajectory_mode='last', trajectory_episodes=10):
    """
    训练算法（兼容层，调用 Trainer 类的核心逻辑）
    :param config_path: 配置文件路径
    :param save_dir: 保存目录
    :param algorithm: 算法名称 ('ddpg', 'dqn', 'maddpg')
    :param num_drones: 无人机数量，None表示使用配置文件中的值
    :param trajectory_mode: 轨迹采集模式 ('first', 'last')，默认为 'last'
    :param trajectory_episodes: 轨迹采集的 episodes 数量，默认为 10
    """
    # 加载配置
    config = load_config(config_path)
    
    # 使用指定的无人机数量或配置文件中的值
    if num_drones is None:
        num_drones = config['env'].get('num_drones', 3)
    else:
        # 覆盖配置中的无人机数量
        config['env']['num_drones'] = num_drones
    
    # 创建保存目录
    create_directory(save_dir)
    log_dir = os.path.join(save_dir, 'logs')
    
    # 从配置中读取动态事件相关参数
    max_drones = config['env'].get('max_drones', num_drones)
    initial_active_drones = config['env'].get('initial_active_drones', num_drones)
    event_schedule = config.get('event', {}).get('schedule', []) if config.get('event', {}).get('enabled', False) else []
    
    # 读取环境参数，添加默认值
    num_users = config['env'].get('num_users', 20)
    max_steps = config['env'].get('max_steps', 128)
    alpha = config['env'].get('alpha', 1.0)
    beta = config['env'].get('beta', 2.0)
    gamma = config['env'].get('gamma', 1.0)
    
    # 创建环境
    env = UAVEnv(
        num_drones=num_drones,
        num_users=num_users,
        max_steps=max_steps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_drones=max_drones,
        initial_active_drones=initial_active_drones,
        event_schedule=event_schedule
    )
    
    # 创建智能体
    obs_dim = env.observation_space.shape[0]
    
    # 使用max_drones作为agent数量，确保与env返回的obs/reward数量一致
    agent_num = max_drones
    
    # 从配置中读取算法参数，添加默认值
    algo_config = config.get('algo', {})
    lr_actor = algo_config.get('lr_actor', 0.001)
    lr_critic = algo_config.get('lr_critic', 0.001)
    gamma = algo_config.get('gamma', 0.99)
    tau = algo_config.get('tau', 0.001)
    batch_size = algo_config.get('batch_size', 64)
    memory_size = algo_config.get('memory_size', 1000000)
    
    if algorithm == 'dqn':
        # 使用DQN算法（离散动作）
        agent = DQN(
            num_agents=agent_num,
            obs_dim=obs_dim,
            action_dim=8,  # 离散动作空间，8个方向
            lr=lr_actor,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=batch_size,
            buffer_size=memory_size
        )
    elif agent_num == 1 or algorithm == 'ddpg':
        # 单无人机使用DDPG（连续动作）
        agent = DDPG(
            obs_dim=obs_dim,
            action_dim=2,  # 连续动作空间，2维速度向量
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            buffer_size=memory_size
        )
    else:
        # 多无人机使用MADDPG
        agent = MADDPG(
            num_agents=agent_num,
            obs_dim=obs_dim,
            action_dim=8,  # 离散动作空间，8个方向
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            memory_size=memory_size
        )
    
    # 训练参数
    train_config = config.get('train', {})
    num_episodes = train_config.get('num_episodes', 1000)
    
    print(f"开始训练，无人机数量: {agent_num}, 用户数量: {num_users}")
    
    # 创建训练器并执行训练
    trainer = Trainer(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        save_dir=save_dir,
        log_dir=log_dir,
        collect_trajectory=True,
        trajectory_episodes=trajectory_episodes,
        trajectory_mode=trajectory_mode
    )
    
    # 执行训练
    score_history, coverage_history, loss_history = trainer.train()
    
    # 绘制训练曲线
    plot_path = os.path.join(save_dir, "training_curves.png")
    plot_training_curves(score_history, coverage_history, loss_history, save_path=plot_path)
    
    print("训练完成！")

if __name__ == "__main__":
    config_path = "src/configs/config.yaml"
    save_dir = "results/models"
    train(config_path, save_dir)
