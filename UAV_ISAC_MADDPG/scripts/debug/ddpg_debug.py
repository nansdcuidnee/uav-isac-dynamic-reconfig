import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入新的DDPG类
from src.agents.ddpg.ddpg_single import DDPG
# 导入UAV环境
from src.envs.uav_env import UAVEnv

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def train_ddpg():
    """
    训练DDPG算法
    """
    # 环境参数
    num_drones = 1
    num_users = 20
    max_steps = 64
    
    # 算法参数
    lr_actor = 0.0001
    lr_critic = 0.0001  # 降低critic学习率到与actor相同，减少震荡
    gamma = 0.99
    tau = 0.001
    batch_size = 128
    buffer_size = 2000000
    
    # 训练参数
    num_episodes = 3000
    save_interval = 100
    log_interval = 10
    
    # 创建环境
    env = UAVEnv(num_drones=num_drones, num_users=num_users, max_steps=max_steps)
    
    # 获取观察维度和动作维度
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = 2  # 2D速度控制
    print(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # 打印观察向量样本，检查状态归一化是否生效
    obs_sample = env._get_obs()
    print("Obs sample (first 10 values):", obs_sample[:10])
    print("SNR part (log compressed):", obs_sample[7:7+env.num_users][:5])  # 只打印前5个SNR值
    print("Connection part:", obs_sample[-env.num_users:][:5])  # 只打印前5个连接状态
    
    # 创建DDPG智能体
    agent = DDPG(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=buffer_size
    )
    
    # 训练历史
    score_history = []
    coverage_history = []
    loss_history = []
    
    # 创建保存目录
    save_dir = "results/drone_number/exp3_num_uav_1_ddpg"
    plots_dir = os.path.join(save_dir, "plots")
    models_dir = os.path.join(save_dir, "models")
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"开始训练，无人机数量: {num_drones}, 用户数量: {num_users}")
    
    for episode in range(num_episodes):
        # 正确处理reset返回的元组
        obs, _ = env.reset()
        # 调试：打印obs的结构
        print(f"Episode {episode+1}, Reset obs type: {type(obs)}, shape: {getattr(obs, 'shape', 'N/A')}, len: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
        if isinstance(obs, (list, np.ndarray)):
            print(f"First element type: {type(obs[0])}, shape: {getattr(obs[0], 'shape', 'N/A')}")
        episode_reward = 0
        episode_coverage = 0
        episode_loss = 0
        
        # 噪声衰减
        noise_std = 0.2 * (0.998 ** episode)  # 初始0.2，衰减更慢
        
        for step in range(max_steps):
            # 选择动作
            action = agent.choose_action(obs, noise_std=noise_std)
            
            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 4:
                # gym格式
                next_obs, reward, done, info = step_result
            else:
                # gymnasium格式
                next_obs, reward, done, truncated, info = step_result
            
            # 提取标量奖励（如果 reward 是列表）
            if isinstance(reward, list):
                scalar_reward = reward[0]
            else:
                scalar_reward = reward
            
            # 记录数据
            episode_reward += scalar_reward
            episode_coverage += info['coverage_rate']
            
            # 每10个episode的第一步打印奖励分量
            if step == 0 and episode % 10 == 0:
                print(f"Episode {episode+1}, Step 0: r_com={info['r_com_norm']:.2f}, r_cover={info['r_cover']:.2f}, distance_reward={info['distance_reward']:.2f}, penalty={info['r_penalty']:.2f}, reward={scalar_reward:.2f}")
            
            # 计算完整观察
            obs_full = obs
            next_obs_full = next_obs
            
            # 处理动作
            actions_flat = action
            
            # 存储经验
            agent.add_experience(obs, obs_full, actions_flat, reward, next_obs, next_obs_full, done)
            
            # 学习
            critic_loss, actor_loss, total_loss = agent.learn()
            episode_loss += total_loss
            
            # 更新状态
            obs = next_obs
            
            if done:
                break
        
        # 记录历史
        score_history.append(episode_reward / max_steps)
        coverage_history.append(episode_coverage / max_steps)
        loss_history.append(episode_loss / max_steps)
        
        # 打印日志
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(score_history[-log_interval:])
            avg_coverage = np.mean(coverage_history[-log_interval:])
            avg_loss = np.mean(loss_history[-log_interval:])
            print(f"Episode {episode+1}/{num_episodes}: Reward={avg_reward:.2f}, Coverage={avg_coverage:.2f}, Loss={avg_loss:.2f}")
        
        # 每20个episode打印详细数据
        if (episode + 1) % 20 == 0:
            print(f"\n===== Episode {episode+1} 详细数据 =====")
            print(f"平均奖励: {episode_reward / max_steps:.2f}")
            print(f"平均覆盖率: {episode_coverage / max_steps:.2f}")
            print(f"平均损失: {episode_loss / max_steps:.2f}")
            print("====================================\n")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f"model_{episode+1}")
            agent.save_checkpoint(model_path)
            print(f"模型已保存到 {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(models_dir, "final_model")
    agent.save_checkpoint(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 保存训练历史
    np.save(os.path.join(logs_dir, "score_history.npy"), score_history)
    np.save(os.path.join(logs_dir, "coverage_history.npy"), coverage_history)
    np.save(os.path.join(logs_dir, "loss_history.npy"), loss_history)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 15))  # 调整画布大小，适应纵向排列
    
    # 奖励曲线
    plt.subplot(3, 1, 1)  # 3行1列，第1个位置
    plt.plot(score_history)
    plt.title('Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # 覆盖率曲线
    plt.subplot(3, 1, 2)  # 3行1列，第2个位置
    plt.plot(coverage_history)
    plt.title('Coverage Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Coverage')
    
    # 损失曲线
    plt.subplot(3, 1, 3)  # 3行1列，第3个位置
    plt.plot(loss_history)
    plt.title('Loss Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    
    # 避免覆盖现有的训练曲线图
    curve_file_path = os.path.join(plots_dir, "training_curves.png")
    if os.path.exists(curve_file_path):
        # 如果文件存在，添加时间戳
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        curve_file_path = os.path.join(plots_dir, f"training_curves_{timestamp}.png")
    
    plt.savefig(curve_file_path)
    print(f"训练曲线已保存到 {curve_file_path}")
    plt.show()
    
    print("训练完成！")

if __name__ == "__main__":
    train_ddpg()
