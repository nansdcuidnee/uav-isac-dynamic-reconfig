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
        num_drones = config['env']['num_drones']
    else:
        # 覆盖配置中的无人机数量
        config['env']['num_drones'] = num_drones
    
    # 创建保存目录
    create_directory(save_dir)
    log_dir = os.path.join(save_dir, 'logs')
    
    # 创建环境
    env = UAVEnv(
        num_drones=num_drones,
        num_users=config['env']['num_users'],
        max_steps=config['env']['max_steps'],
        alpha=config['env']['alpha'],
        beta=config['env']['beta'],
        gamma=config['env']['gamma']
    )
    
    # 创建智能体
    obs_dim = env.observation_space.shape[0]
    
    if algorithm == 'dqn':
        # 使用DQN算法（离散动作）
        agent = DQN(
            num_agents=num_drones,
            obs_dim=obs_dim,
            action_dim=8,  # 离散动作空间，8个方向
            lr=config['algo']['lr_actor'],
            gamma=config['algo']['gamma'],
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=config['algo']['batch_size'],
            buffer_size=config['algo']['memory_size']
        )
    elif num_drones == 1 or algorithm == 'ddpg':
        # 单无人机使用DDPG（连续动作）
        agent = DDPG(
            obs_dim=obs_dim,
            action_dim=2,  # 连续动作空间，2维速度向量
            lr_actor=config['algo']['lr_actor'],
            lr_critic=config['algo']['lr_critic'],
            gamma=config['algo']['gamma'],
            tau=config['algo']['tau'],
            batch_size=config['algo']['batch_size'],
            buffer_size=config['algo']['memory_size']
        )
    else:
        # 多无人机使用MADDPG
        agent = MADDPG(
            num_agents=num_drones,
            obs_dim=obs_dim,
            action_dim=8,  # 离散动作空间，8个方向
            lr_actor=config['algo']['lr_actor'],
            lr_critic=config['algo']['lr_critic'],
            gamma=config['algo']['gamma'],
            tau=config['algo']['tau'],
            batch_size=config['algo']['batch_size'],
            memory_size=config['algo']['memory_size']
        )
    
    # 训练参数
    num_episodes = config['train']['num_episodes']
    
    print(f"开始训练，无人机数量: {config['env']['num_drones']}, 用户数量: {config['env']['num_users']}")
    
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
