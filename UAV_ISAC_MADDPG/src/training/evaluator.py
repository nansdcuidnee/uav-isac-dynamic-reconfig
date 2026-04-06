import numpy as np

class Evaluator:
    """
    评估器
    - 测试和评估模型性能
    - 生成轨迹和帧数据
    """

    def __init__(self, env, agent, num_episodes):
        """
        初始化评估器
        :param env: 环境
        :param agent: 智能体
        :param num_episodes: 评估回合数
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def evaluate(self):
        """
        开始评估
        :return: 评估结果
        """
        print(f"开始评估，共 {self.num_episodes} 回合")

        total_score = 0
        total_coverage = 0
        all_trajectories = []
        all_frames_data = []

        for episode in range(self.num_episodes):
            # 重置环境
            obs, _ = self.env.reset(seed=episode)
            # 确保obs是列表格式
            if not isinstance(obs, list):
                obs = [obs]

            score = 0
            done = False
            episode_coverage = []
            trajectories = [[] for _ in range(self.env.num_drones)]
            frames_data = []

            while not done:
                # 选择动作（无噪声）
                actions = self.agent.choose_action(obs, noise_std=0.0)
                # 执行动作
                obs_, rewards, done, truncated, info = self.env.step(actions)
                # 确保obs_是列表格式
                if not isinstance(obs_, list):
                    obs_ = [obs_]
                # 确保rewards是列表格式
                if not isinstance(rewards, list):
                    rewards = [rewards]

                # 记录轨迹
                drone_pos = self.env.get_positions()[0]
                for i in range(self.env.num_drones):
                    trajectories[i].append(drone_pos[i])

                # 记录帧数据
                frame_data = {
                    'drone_pos': drone_pos,
                    'user_pos': self.env.get_positions()[1],
                    'connections': self.env.user_connections,
                    'trajectories': trajectories
                }
                frames_data.append(frame_data)

                # 更新观察
                obs = obs_
                # 累积分数
                score += sum(rewards)
                # 记录覆盖率
                coverage_rate = info.get('coverage_rate', 0)
                episode_coverage.append(coverage_rate)

            # 累积总分数和覆盖率
            total_score += score
            total_coverage += np.mean(episode_coverage)
            all_trajectories.append(trajectories)
            all_frames_data.extend(frames_data)

            # 打印进度
            print(f"评估回合 {episode+1}/{self.num_episodes}: 得分 = {score:.2f}, 覆盖率 = {np.mean(episode_coverage):.2f}")

        # 计算平均值
        avg_score = total_score / self.num_episodes
        avg_coverage = total_coverage / self.num_episodes

        print(f"评估完成！平均得分: {avg_score:.2f}, 平均覆盖率: {avg_coverage:.2f}")
        return avg_score, avg_coverage, all_trajectories, all_frames_data