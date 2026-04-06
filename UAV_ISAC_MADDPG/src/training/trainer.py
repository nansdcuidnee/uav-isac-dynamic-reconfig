import os
import numpy as np
import torch
from src.training.logger import Logger
from src.training.checkpoint import Checkpoint
from src.agents.maddpg.maddpg import MADDPG

class Trainer:
    """
    训练器
    - 主训练循环
    - 模型保存
    - 日志记录
    """

    def __init__(self, env, agent, num_episodes, save_dir, log_dir, collect_trajectory=True, trajectory_episodes=10, trajectory_mode='first'):
        """
        初始化训练器
        :param env: 环境
        :param agent: 智能体
        :param num_episodes: 训练回合数
        :param save_dir: 保存目录
        :param log_dir: 日志目录
        :param collect_trajectory: 是否收集轨迹数据
        :param trajectory_episodes: 轨迹采集回合数
        :param trajectory_mode: 轨迹采集模式 ('first' 或 'last')
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.collect_trajectory = collect_trajectory
        self.trajectory_episodes = trajectory_episodes
        self.trajectory_mode = trajectory_mode

        # 创建必要的目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 初始化日志记录器
        self.logger = Logger(log_dir)
        # 初始化检查点管理器
        self.checkpoint = Checkpoint(save_dir)

        # 训练历史
        self.score_history = []
        self.coverage_history = []
        self.loss_history = []
        # 轨迹数据
        self.trajectory_data = []

    def train(self):
        """
        开始训练
        :return: 训练历史
        """
        print(f"开始训练，共 {self.num_episodes} 回合")

        for episode in range(self.num_episodes):
            # 重置环境
            reset_result = self.env.reset(seed=episode)
            if len(reset_result) == 2:
                # gymnasium格式
                obs, _ = reset_result
            else:
                # gym格式
                obs = reset_result
            # 确保obs是列表格式
            if not isinstance(obs, list):
                obs = [obs]

            score = 0
            done = False
            episode_loss = 0
            episode_coverage = []
            step = 0

            while not done:
                # 选择动作
                actions = self.agent.choose_action(obs, noise_std=0.1)
                # 执行动作
                step_result = self.env.step(actions)
                if len(step_result) == 4:
                    # gym格式
                    obs_, rewards, done, info = step_result
                else:
                    # gymnasium格式
                    obs_, rewards, done, truncated, info = step_result
                # 确保obs_是列表格式
                if not isinstance(obs_, list):
                    obs_ = [obs_]
                # 确保rewards是列表格式
                if not isinstance(rewards, list):
                    rewards = [rewards]

                # 计算完整观察
                obs_full = np.concatenate(obs)
                obs_full_ = np.concatenate(obs_)
                
                # 处理动作，确保可以连接
                if isinstance(actions, list):
                    # 检查列表中的元素类型
                    if len(actions) > 0:
                        if isinstance(actions[0], (int, np.integer)):
                            # 检查是否为 MADDPG 智能体
                            if isinstance(self.agent, MADDPG):
                                # MADDPG 离散动作，转换为 one-hot 编码
                                num_agents = self.agent.num_agents
                                action_dim = self.agent.action_dim
                                actions_one_hot = []
                                for action in actions:
                                    # 转成 int 并做范围裁剪
                                    action_idx = int(action)
                                    action_idx = max(0, min(action_idx, action_dim - 1))
                                    one_hot = np.zeros(action_dim)
                                    one_hot[action_idx] = 1
                                    actions_one_hot.extend(one_hot)
                                actions_flat = np.array(actions_one_hot, dtype=np.float32)
                            else:
                                # DQN 离散动作，转换为数组
                                actions_flat = np.array(actions, dtype=np.float32)
                        else:
                            # 尝试连接动作
                            try:
                                actions_flat = np.concatenate(actions)
                            except ValueError:
                                # 如果连接失败，可能是单无人机零维动作
                                actions_flat = np.array(actions, dtype=np.float32)
                    else:
                        # 空列表，使用空数组
                        actions_flat = np.array([], dtype=np.float32)
                elif isinstance(actions, np.ndarray):
                    # DDPG 单智能体返回的一维数组
                    actions_flat = actions.astype(np.float32)
                else:
                    # 其他类型，尝试转换为数组
                    actions_flat = np.array(actions, dtype=np.float32)

                # 存储经验
                self.agent.memory.add(
                    obs, obs_full, actions_flat, rewards, obs_, obs_full_, [done]*len(rewards)
                )

                # 学习
                loss = self.agent.learn()
                # 处理不同类型的返回值
                if isinstance(loss, tuple):
                    # 如果返回的是元组，取第一个值或计算平均值
                    episode_loss += sum(loss) / len(loss)
                else:
                    # 如果返回的是标量，直接累加
                    episode_loss += loss

                # 收集轨迹数据
                if self.collect_trajectory:
                    # 检查是否在采集范围内
                    if (self.trajectory_mode == 'first' and episode < self.trajectory_episodes) or \
                       (self.trajectory_mode == 'last' and episode >= self.num_episodes - self.trajectory_episodes):
                        drone_pos, user_pos = self.env.get_positions()
                        for uav_id, pos in enumerate(drone_pos):
                            self.trajectory_data.append({
                                'episode': episode,
                                'step': step,
                                'uav_id': uav_id,
                                'x': pos[0],
                                'y': pos[1],
                                'z': pos[2]
                            })

                # 更新观察
                obs = obs_
                # 累积分数
                score += sum(rewards)
                # 记录覆盖率
                coverage_rate = info.get('coverage_rate', 0)
                episode_coverage.append(coverage_rate)
                # 增加步数
                step += 1

            # 记录历史
            self.score_history.append(score)
            avg_coverage = np.mean(episode_coverage)
            self.coverage_history.append(avg_coverage)
            self.loss_history.append(episode_loss)

            # 记录日志
            self.logger.log_scalar('score', score, episode)
            self.logger.log_scalar('coverage', avg_coverage, episode)
            self.logger.log_scalar('loss', episode_loss, episode)

            # 打印进度
            if episode % 10 == 0:
                avg_score = np.mean(self.score_history[-10:])
                avg_loss = np.mean(self.loss_history[-10:])
                print(f"回合 {episode}/{self.num_episodes}: 得分 = {score:.2f}, 平均得分 = {avg_score:.2f}, 覆盖率 = {avg_coverage:.2f}, 损失 = {avg_loss:.2f}")

            # 保存模型
            if episode % 100 == 0:
                model_path = os.path.join(self.save_dir, 'models', f'model_{episode}.pt')
                self.agent.save_checkpoint(model_path)
                print(f"模型已保存到 {model_path}")

        # 保存最终模型
        final_model_path = os.path.join(self.save_dir, 'models', 'best_model.pt')
        self.agent.save_checkpoint(final_model_path)
        print(f"最终模型已保存到 {final_model_path}")

        # 保存训练历史
        np.save(os.path.join(self.log_dir, 'score_history.npy'), self.score_history)
        np.save(os.path.join(self.log_dir, 'coverage_history.npy'), self.coverage_history)
        np.save(os.path.join(self.log_dir, 'loss_history.npy'), self.loss_history)

        # 保存轨迹数据
        if self.collect_trajectory:
            self.save_trajectory_data()

        print("训练完成！")
        return self.score_history, self.coverage_history, self.loss_history

    def save_trajectory_data(self):
        """
        保存轨迹数据和用户位置数据
        """
        if not self.trajectory_data:
            return
        
        import json
        # 保存轨迹数据
        trajectory_path = os.path.join(self.save_dir, "trajectory_data.json")
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)
        print(f"轨迹数据已保存到 {trajectory_path}")
        
        # 保存用户位置数据（固定位置）
        drone_pos, user_pos = self.env.get_positions()
        user_positions_path = os.path.join(self.save_dir, "user_positions.json")
        user_data = []
        for user_id, pos in enumerate(user_pos):
            user_data.append({
                'user_id': user_id,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2]
            })
        with open(user_positions_path, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        print(f"用户位置数据已保存到 {user_positions_path}")