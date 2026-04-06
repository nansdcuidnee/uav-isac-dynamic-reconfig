import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# 导入DQN类
from agents.dqn.dqn import DQN
# 导入UAV环境
from envs.uav_env import UAVEnv

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def train_dqn():
    """
    训练DQN算法
    """
    # 环境参数
    num_drones = 1
    num_users = 20
    max_steps = 128  # 增加最大步数
    
    # 算法参数
    lr = 0.0003  # 降低学习率
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.999  # 更慢的衰减
    batch_size = 128  # 增大批量大小
    buffer_size = 200000  # 减小经验回放缓冲区，降低内存占用
    
    # 训练参数
    num_episodes = 3000
    save_interval = 100
    log_interval = 10
    
    # 创建环境
    env = UAVEnv(num_drones=num_drones, num_users=num_users, max_steps=max_steps)
    
    # 获取观察维度和动作维度
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = 8  # 8个离散动作方向
    print(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # 打印观察向量样本，检查状态归一化是否生效
    obs_sample = env._get_obs()
    print("Obs sample (first 10 values):", obs_sample[:10])
    print("SNR part (log compressed):", obs_sample[7:7+env.num_users][:5])  # 只打印前5个SNR值
    print("Connection part:", obs_sample[-env.num_users:][:5])  # 只打印前5个连接状态
    
    # 创建DQN智能体
    agent = DQN(
        num_agents=num_drones,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size
    )
    
    # 训练历史 - 使用 deque 限制长度
    from collections import deque
    max_history_length = 1000  # 只保存最近 1000 回合的历史
    score_history = deque(maxlen=max_history_length)
    coverage_history = deque(maxlen=max_history_length)
    loss_history = deque(maxlen=max_history_length)
    epsilon_history = deque(maxlen=max_history_length)
    
    # 创建保存目录
    save_dir = "results/drone_number/exp3_num_uav_1_dqn"
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
        # 确保obs是单个数组
        if isinstance(obs, list):
            obs = obs[0]
        
        # 调试：打印obs的结构
        print(f"Episode {episode+1}, Reset obs type: {type(obs)}, shape: {getattr(obs, 'shape', 'N/A')}, len: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
        if isinstance(obs, (list, np.ndarray)):
            print(f"First element type: {type(obs[0])}, shape: {getattr(obs[0], 'shape', 'N/A')}")
        
        episode_reward = 0
        episode_coverage = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # 选择动作（DQN需要传入列表形式的obs）
            action = agent.choose_action([obs])
            
            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 4:
                # gym格式
                next_obs, reward, done, info = step_result
            else:
                # gymnasium格式
                next_obs, reward, done, truncated, info = step_result
            
            # 确保next_obs是单个数组
            if isinstance(next_obs, list):
                next_obs = next_obs[0]
            
            # 提取标量奖励（如果 reward 是列表）
            if isinstance(reward, list):
                scalar_reward = reward[0]
            else:
                scalar_reward = reward
            
            # 记录数据
            episode_reward += scalar_reward
            episode_coverage += info['coverage_rate']
            
            # 降低日志输出频率
            if step == 0 and episode % 50 == 0:  # 每50个episode打印一次
                print(f"Episode {episode+1}, Step 0: min_dist={info['min_dist']:.2f}, r_cover={info['r_cover']:.2f}, distance_reward={info['distance_reward']:.2f}, penalty={info['r_penalty']:.2f}, reward={scalar_reward:.2f}")
            
            # 计算完整观察
            obs_full = obs
            next_obs_full = next_obs
            
            # 处理动作
            actions_flat = np.array(action, dtype=np.float32)
            
            # 存储经验
            agent.add_experience([obs], obs_full, actions_flat, [reward], [next_obs], next_obs_full, [done])
            
            # 学习
            loss = agent.learn()
            episode_loss += loss
            
            # 更新状态
            obs = next_obs
            
            if done:
                break
        
        # 每100回合手动释放内存
        if (episode + 1) % 100 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("内存已释放")
        
        # 记录历史
        score_history.append(episode_reward / max_steps)
        coverage_history.append(episode_coverage / max_steps)
        loss_history.append(episode_loss / max_steps)
        epsilon_history.append(agent.epsilon)
        
        # 打印日志
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(score_history[-log_interval:])
            avg_coverage = np.mean(coverage_history[-log_interval:])
            avg_loss = np.mean(loss_history[-log_interval:])
            current_epsilon = agent.epsilon
            print(f"Episode {episode+1}/{num_episodes}: Reward={avg_reward:.2f}, Coverage={avg_coverage:.2f}, Loss={avg_loss:.2f}, Epsilon={current_epsilon:.3f}")
        
        # 每20个episode打印详细数据
        if (episode + 1) % 20 == 0:
            print(f"\n===== Episode {episode+1} 详细数据 =====")
            print(f"平均奖励: {episode_reward / max_steps:.2f}")
            print(f"平均覆盖率: {episode_coverage / max_steps:.2f}")
            print(f"平均损失: {episode_loss / max_steps:.2f}")
            print(f"当前探索率: {agent.epsilon:.3f}")
            print("====================================\n")
        
        # 保存模型和历史数据 - 降低保存频率
        if (episode + 1) % (save_interval * 2) == 0:  # 每 200 回合保存一次
            model_path = os.path.join(models_dir, f"model_{episode+1}")
            agent.save_checkpoint(model_path)
            print(f"模型已保存到 {model_path}")
            
            # 定期保存历史数据
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            np.save(os.path.join(logs_dir, f"score_history_{timestamp}.npy"), list(score_history))
            np.save(os.path.join(logs_dir, f"coverage_history_{timestamp}.npy"), list(coverage_history))
            np.save(os.path.join(logs_dir, f"loss_history_{timestamp}.npy"), list(loss_history))
            np.save(os.path.join(logs_dir, f"epsilon_history_{timestamp}.npy"), list(epsilon_history))
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
    np.save(os.path.join(logs_dir, f"epsilon_history_{timestamp}.npy"), list(epsilon_history))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 16))  # 调整画布大小，适应纵向排列
    
    # 奖励曲线
    plt.subplot(4, 1, 1)  # 4行1列，第1个位置
    plt.plot(list(score_history))
    plt.title('Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # 覆盖率曲线
    plt.subplot(4, 1, 2)  # 4行1列，第2个位置
    plt.plot(list(coverage_history))
    plt.title('Coverage Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Coverage')
    
    # 损失曲线
    plt.subplot(4, 1, 3)  # 4行1列，第3个位置
    plt.plot(list(loss_history))
    plt.title('Loss Curve')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    
    # 探索率曲线
    plt.subplot(4, 1, 4)  # 4行1列，第4个位置
    plt.plot(list(epsilon_history))
    plt.title('Epsilon Curve')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    
    # 使用时间戳避免覆盖现有文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    curve_file_path = os.path.join(plots_dir, f"training_curves_{timestamp}.png")
    
    plt.savefig(curve_file_path)
    print(f"训练曲线已保存到 {curve_file_path}")
    plt.close()  # 关闭图形，释放内存
    print("图形已关闭，内存已释放")
    
    print("训练完成！")

if __name__ == "__main__":
    train_dqn()
