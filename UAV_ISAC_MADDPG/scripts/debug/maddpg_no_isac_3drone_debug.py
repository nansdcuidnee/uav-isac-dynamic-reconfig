import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# 导入修复后的MADDPG类
from agents.maddpg.maddpg_fixed import MADDPG
# 导入UAV环境
from envs.uav_env import UAVEnv

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def train_maddpg_no_isac_3drone():
    """
    训练MADDPG算法（3个无人机）- 无ISAC版本
    """
    # 环境参数
    num_drones = 3
    num_users = 20
    max_steps = 128
    
    # 算法参数 - 调整学习率以获得更好的收敛
    lr_actor = 0.0003  # 增加学习率
    lr_critic = 0.0003  # 增加学习率
    gamma = 0.99
    tau = 0.005  # 增加软更新系数
    batch_size = 128
    buffer_size = 200000
    
    # 训练参数
    num_episodes = 3000
    save_interval = 100
    log_interval = 10
    
    # 创建环境
    env = UAVEnv(num_drones=num_drones, num_users=num_users, max_steps=max_steps)
    
    # 获取观察维度和动作维度
    obs, _ = env.reset()
    if not isinstance(obs, list):
        obs = [obs]
    obs_dim = len(obs[0])
    action_dim = 8
    print(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # 打印观察向量样本
    obs_sample = env._get_obs()
    if obs_sample and isinstance(obs_sample[0], np.ndarray):
        print("Obs sample (first 10 values):", obs_sample[0][:10])
        print("SNR part (log compressed):", obs_sample[0][7:7+env.num_users][:5])
        print("Connection part:", obs_sample[0][-env.num_users:][:5])
    else:
        print("Obs sample:", obs_sample)
    
    # 创建MADDPG智能体（使用修复版）
    agent = MADDPG(
        num_agents=num_drones,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        memory_size=buffer_size
    )
    
    # 训练历史 - 使用列表保存完整数据
    score_history = []
    coverage_history = []
    loss_history = []
    
    # 创建保存目录
    save_dir = "results/drone_number/exp3_num_uav_3_no_isac"
    plots_dir = os.path.join(save_dir, "plots")
    models_dir = os.path.join(save_dir, "models")
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"开始训练，无人机数量: {num_drones}, 用户数量: {num_users}")
    print(f"学习率: Actor={lr_actor}, Critic={lr_critic}, Tau={tau}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        if not isinstance(obs, list):
            obs = [obs]
        
        episode_reward = 0
        episode_coverage = 0
        episode_loss = 0
        
        # 噪声衰减 - 使用更慢的衰减
        noise_std = max(0.1, 0.3 * (0.995 ** episode))
        
        for step in range(max_steps):
            # 选择动作
            action = agent.choose_action(obs, noise_std=noise_std)
            
            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                next_obs, reward, done, truncated, info = step_result
            
            if not isinstance(next_obs, list):
                next_obs = [next_obs]
            if not isinstance(reward, list):
                reward = [reward] * num_drones
            
            episode_reward += sum(reward)
            episode_coverage += info['coverage_rate']
            
            # 降低日志输出频率
            if step == 0 and episode % 50 == 0:
                print(f"Episode {episode+1}, Step 0: min_dist={info['min_dist']:.2f}, "
                      f"r_cover={info['r_cover']:.2f}, distance_reward={info['distance_reward']:.2f}, "
                      f"penalty={info['r_penalty']:.2f}, reward={sum(reward):.2f}, noise={noise_std:.3f}")
            
            # 计算完整观察
            obs_full = np.concatenate(obs) if len(obs) > 1 else obs[0]
            next_obs_full = np.concatenate(next_obs) if len(next_obs) > 1 else next_obs[0]
            
            # 处理动作 - 转换为one-hot编码
            actions_flat = []
            for a in action:
                one_hot = np.zeros(action_dim, dtype=np.float32)
                one_hot[a] = 1.0
                actions_flat.append(one_hot)
            actions_flat = np.concatenate(actions_flat)
            
            # 存储经验
            agent.memory.add(obs, obs_full, actions_flat, reward, next_obs, next_obs_full, [done]*len(reward))
            
            # 学习
            loss = agent.learn()
            episode_loss += loss
            
            obs = next_obs
            
            if done:
                break
        
        # 每100回合释放内存
        if (episode + 1) % 100 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Episode {episode+1}: 内存已释放")
        
        # 记录历史
        score_history.append(episode_reward / max_steps)
        coverage_history.append(episode_coverage / max_steps)
        loss_history.append(episode_loss / max_steps)
        
        # 打印日志
        if (episode + 1) % log_interval == 0:
            scores = list(score_history)
            coverages = list(coverage_history)
            losses = list(loss_history)

            avg_reward = np.mean(scores[-log_interval:])
            avg_coverage = np.mean(coverages[-log_interval:])
            avg_loss = np.mean(losses[-log_interval:])

            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={avg_reward:.3f}, Coverage={avg_coverage:.3f}, Loss={avg_loss:.3f}, "
                  f"Noise={noise_std:.3f}")
        
        # 每50个episode打印详细数据
        if (episode + 1) % 50 == 0:
            scores = list(score_history)
            coverages = list(coverage_history)
            
            # 计算最近50回合的平均值
            recent_reward = np.mean(scores[-50:])
            recent_coverage = np.mean(coverages[-50:])
            
            print(f"\n===== Episode {episode+1} 详细数据 =====")
            print(f"最近50回合平均奖励: {recent_reward:.3f}")
            print(f"最近50回合平均覆盖率: {recent_coverage:.3f}")
            print(f"当前回合平均奖励: {episode_reward / max_steps:.3f}")
            print(f"当前回合平均覆盖率: {episode_coverage / max_steps:.3f}")
            print(f"当前探索噪声: {noise_std:.3f}")
            print("====================================\n")
        
        # 保存模型和历史数据
        if (episode + 1) % (save_interval * 2) == 0:
            model_path = os.path.join(models_dir, f"model_{episode+1}")
            agent.save_checkpoint(model_path)
            print(f"模型已保存到 {model_path}")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            np.save(os.path.join(logs_dir, f"score_history_{timestamp}.npy"), list(score_history))
            np.save(os.path.join(logs_dir, f"coverage_history_{timestamp}.npy"), list(coverage_history))
            np.save(os.path.join(logs_dir, f"loss_history_{timestamp}.npy"), list(loss_history))
            print(f"历史数据已保存")
    
    # 保存最终模型
    final_model_path = os.path.join(models_dir, "final_model")
    agent.save_checkpoint(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 保存训练历史
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(logs_dir, f"score_history_{timestamp}.npy"), list(score_history))
    np.save(os.path.join(logs_dir, f"coverage_history_{timestamp}.npy"), list(coverage_history))
    np.save(os.path.join(logs_dir, f"loss_history_{timestamp}.npy"), list(loss_history))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 12))
    
    # 奖励曲线
    plt.subplot(3, 1, 1)
    plt.plot(list(score_history))
    plt.title('Reward Curve (MADDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    # 覆盖率曲线
    plt.subplot(3, 1, 2)
    plt.plot(list(coverage_history))
    plt.title('Coverage Curve (MADDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Average Coverage')
    plt.grid(True, alpha=0.3)
    
    # 损失曲线
    plt.subplot(3, 1, 3)
    plt.plot(list(loss_history))
    plt.title('Loss Curve (MADDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 使用时间戳避免覆盖现有文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    curve_file_path = os.path.join(plots_dir, f"training_curves_{timestamp}.png")
    
    plt.savefig(curve_file_path, dpi=150)
    print(f"训练曲线已保存到 {curve_file_path}")
    plt.close()
    print("图形已关闭，内存已释放")
    
    print("训练完成！")

if __name__ == "__main__":
    train_maddpg_no_isac_3drone()
