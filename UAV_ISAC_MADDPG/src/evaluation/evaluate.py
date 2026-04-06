import os
import numpy as np
import sys

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)


from src.envs.uav_env import UAVEnv
from src.agents.maddpg.maddpg import MADDPG
from src.utils import load_config

def evaluate(config_path, model_path):
    """
    评估训练好的模型
    :param config_path: 配置文件路径
    :param model_path: 模型路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 根据模型路径调整无人机数量
    num_drones = config['env']['num_drones']
    inferred = False
    
    # 支持多种命名模式
    import re
    
    # 模式1: 1_drones, 3_drones, 5_drones
    match = re.search(r'(\d+)_drones', model_path.lower())
    if match:
        num_drones = int(match.group(1))
        inferred = True
    
    # 模式2: num_uav_1, num_uav_3, num_uav_5
    if not inferred:
        match = re.search(r'num_uav_(\d+)', model_path.lower())
        if match:
            num_drones = int(match.group(1))
            inferred = True
    
    # 模式3: uav_1, uav_3, uav_5
    if not inferred:
        match = re.search(r'uav_(\d+)', model_path.lower())
        if match:
            num_drones = int(match.group(1))
            inferred = True
    
    # 如果无法推断，给出警告
    if not inferred:
        print(f"警告: 无法从模型路径 '{model_path}' 推断无人机数量，使用配置文件中的值: {num_drones}")
    
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
    
    # 根据模型路径选择合适的智能体
    # 使用正则表达式确保匹配完整的单词
    import re
    
    # 检查是否包含 'maddpg' 子字符串
    maddpg_match = 'maddpg' in model_path.lower()
    # 检查是否包含 'ddpg' 子字符串但不包含 'maddpg'
    ddpg_match = 'ddpg' in model_path.lower() and not maddpg_match
    # 检查是否包含 'dqn' 子字符串
    dqn_match = 'dqn' in model_path.lower()
    
    if maddpg_match:
        # 使用MADDPG智能体
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
    elif ddpg_match:
        # 使用DDPG智能体
        from src.agents.ddpg.ddpg_single import DDPG
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
    elif dqn_match:
        # 使用DQN智能体
        from src.agents.dqn.dqn import DQN
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
    else:
        # 默认使用MADDPG智能体
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
    
    # 加载模型
    agent.load_checkpoint(model_path)
    print(f"模型已加载: {model_path}")
    
    # 评估参数
    num_episodes = config['eval']['num_episodes']
    
    # 评估历史
    total_rewards = []
    total_coverages = []
    total_rates = []
    
    print(f"开始评估，无人机数量: {num_drones}, 用户数量: {config['env']['num_users']}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=config['eval']['seed'] + episode)
        episode_reward = 0
        episode_coverage = 0
        episode_rate = 0
        
        for step in range(config['env']['max_steps']):
            # 选择动作（无噪声）
            if isinstance(obs, list):
                action = agent.choose_action(obs, noise_std=0.0)
            else:
                action = agent.choose_action([obs], noise_std=0.0)
            
            # 执行动作
            next_obs, reward, done, _, info = env.step(action)
            
            # 记录数据
            episode_reward += sum(reward) if isinstance(reward, list) else reward
            episode_coverage += info['coverage_rate']
            episode_rate += info['total_rate']
            
            # 更新状态
            obs = next_obs
            
            if done:
                break
        
        # 记录历史
        total_rewards.append(episode_reward / config['env']['max_steps'])
        total_coverages.append(episode_coverage / config['env']['max_steps'])
        total_rates.append(episode_rate / config['env']['max_steps'])
        
        print(f"Episode {episode+1}/{num_episodes}: Reward={total_rewards[-1]:.2f}, Coverage={total_coverages[-1]:.2f}, Rate={total_rates[-1]:.2f}")
    
    # 计算平均值
    avg_reward = np.mean(total_rewards)
    avg_coverage = np.mean(total_coverages)
    avg_rate = np.mean(total_rates)
    
    print(f"\n评估结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均覆盖率: {avg_coverage:.2f}")
    print(f"平均通信速率: {avg_rate:.2f}")
    
    return avg_reward, avg_coverage, avg_rate

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument('--config', type=str, default='src/configs/config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='results/models/ddpg_1_drones/models/model_600.pt', help='模型路径')
    
    args = parser.parse_args()
    
    evaluate(args.config, args.model)
