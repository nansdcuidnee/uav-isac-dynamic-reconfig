import numpy as np
import gym
from gym import spaces
from src.envs.models.communication_model import CommunicationModel
from src.envs.models.sensing_model import SensingModel
from src.envs.models.uav_model import UAVModel
from src.envs.models.user_model import UserModel
from src.envs.models.environment_model import EnvironmentModel
from src.utils import calculate_distance, clip_position, calculate_battery_consumption, generate_random_position

class UAVEnv(gym.Env):
    """
    UAV通信感知一体化环境
    - 多无人机轨迹规划
    - 通信速率计算
    - 感知SNR计算
    - 覆盖率评估
    """

    def __init__(self, num_drones=3, num_users=20, max_steps=128, alpha=1.0, beta=2.0, gamma=1.0):
        super(UAVEnv, self).__init__()

        self.num_drones = num_drones
        self.num_users = num_users
        self.max_steps = max_steps
        self.alpha = alpha  # 通信速率权重
        self.beta = beta    # 覆盖率权重
        self.gamma = gamma  # 惩罚项权重

        # 动作空间设置
        # 对于 DQN，使用离散动作空间
        self.action_space = spaces.Discrete(8)
        
        # 同时保持连续动作的兼容性（用于 DDPG 等算法）
        self.continuous_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 方向映射
        self.DIRECTIONS = np.array([
            [1, 0],   # 前
            [-1, 0],  # 后
            [0, 1],   # 左
            [0, -1],  # 右
            [1, 1],   # 左前
            [1, -1],  # 左后
            [-1, 1],  # 右前
            [-1, -1]  # 右后
        ])

        # 状态空间
        # 每个智能体的观察维度
        obs_dim = 2 + 2 + 1 + 1 + 1 + num_users + num_users  # 2(位置) + 2(速度) + 1(越界) + 1(碰撞) + 1(电池) + num_users(SNR) + num_users(连接)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 通信参数
        self.P = 100.0  # 发射功率
        self.noise = 1e-3  # 噪声功率
        self.snr_threshold = 0.01  # 感知信噪比阈值
        self.safe_distance = 5.0  # 安全距离
        self.communication_range = 100.0  # 通信范围
        self.sensing_range = 120.0  # 感知范围
        self.max_speed = 10.0  # 最大速度
        self.max_acc = 15.0  # 最大加速度，增大到15.0
        self.dt = 1.0  # 时间步长
        self.area_size = 200.0  # 区域大小

        # 状态变量
        self.out_of_bounds_count = 0
        self.collision_count = 0
        self.step_count = 0

        # 电池模型
        self.battery = np.ones(self.num_drones) * 100.0  # 初始电量100%

        # 奖励分量，用于调试
        self.r_com_raw = 0.0
        self.r_cover = 0.0
        self.r_penalty = 0.0
        self.dist_reward = 0.0
        self.min_dist = 0.0

        # 生成固定用户位置
        self.user_pos = np.array([generate_random_position(self.area_size, 0.0) for _ in range(self.num_users)])

        # 初始化环境模型
        self.env_model = EnvironmentModel(area_size=self.area_size, time_steps=self.max_steps)
        
        # 初始化 UAV 模型列表
        self.uav_models = []
        for i in range(self.num_drones):
            initial_position = generate_random_position(self.area_size, 50.0)
            uav_model = UAVModel(
                uav_id=i,
                initial_position=initial_position,
                max_speed=self.max_speed,
                communication_range=self.communication_range,
                sensing_range=self.sensing_range
            )
            self.uav_models.append(uav_model)
        
        # 初始化用户模型列表
        self.user_models = []
        for i in range(self.num_users):
            user_model = UserModel(
                user_id=i,
                position=self.user_pos[i]
            )
            self.user_models.append(user_model)

        # 初始化通信和感知模型
        self.channel_model = CommunicationModel(P=self.P, noise=self.noise)
        self.sensing_model = SensingModel(P=self.P, noise=self.noise, snr_threshold=self.snr_threshold)

        # ISAC模块集成测试
        self._test_isac_integration()

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # 无人机位置（3D）
        self.drone_pos = np.array([generate_random_position(self.area_size, 50.0) for _ in range(self.num_drones)])
        self.drone_vel = np.zeros((self.num_drones, 2))

        # 用户位置（3D）- 保持固定，不再重新生成

        # 重置状态变量
        self.out_of_bounds_count = 0
        self.collision_count = 0
        self.step_count = 0

        # 重置电池电量
        self.battery = np.ones(self.num_drones) * 100.0

        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        # 确保actions是numpy数组
        actions = np.array(actions)

        # 处理actions的维度
        if actions.ndim == 1:
            # 对于单无人机，使用一维数组作为连续动作
            if self.num_drones == 1:
                actions = np.expand_dims(actions, axis=0)
        elif actions.ndim == 2:
            # 对于多无人机，确保形状正确
            if actions.shape[0] != self.num_drones:
                actions = actions.reshape(self.num_drones, -1)

        # 确保actions的形状与drone数量匹配
        actions = actions[:self.num_drones]

        # 更新速度和位置
        for i in range(self.num_drones):
            action = actions[i]
            if isinstance(action, np.ndarray) and len(action) >= 2:
                # 加速度控制
                self.drone_vel[i] += action[:2] * self.max_acc * self.dt
                self.drone_vel[i] = np.clip(self.drone_vel[i], -self.max_speed, self.max_speed)
                self.drone_pos[i, :2] += self.drone_vel[i] * self.dt
            else:
                # 兼容离散动作
                try:
                    # 尝试直接使用整数动作索引
                    if isinstance(action, (np.ndarray, np.integer, np.floating)):
                        action_idx = int(action.item())
                    else:
                        action_idx = int(action)
                    action_idx = np.clip(action_idx, 0, 7)
                except (ValueError, TypeError):
                    # 如果失败，尝试归一化处理（兼容其他算法）
                    if isinstance(action, (np.ndarray, np.integer, np.floating)):
                        action_val = action.item()
                    else:
                        action_val = action
                    action_idx = int((action_val + 1) / 2 * 7)
                    action_idx = np.clip(action_idx, 0, 7)
                direction = self.DIRECTIONS[action_idx]
                # 直接设置速度，而不是平滑更新
                self.drone_vel[i] = direction * self.max_speed
                self.drone_pos[i, :2] += self.drone_vel[i] * self.dt

        # 边界检测
        out_of_bounds = 0
        for i in range(self.num_drones):
            # 检查是否越界
            pos = self.drone_pos[i, :2]
            if np.any(pos < 0) or np.any(pos > self.area_size):
                out_of_bounds += 1
            # 裁剪位置
            clipped_pos = clip_position(self.drone_pos[i], self.area_size)
            self.drone_pos[i] = clipped_pos
        out_penalty = out_of_bounds
        self.out_of_bounds_count += out_penalty

        # 碰撞检测
        collision_penalty = 0
        current_collisions = 0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = calculate_distance(self.drone_pos[i], self.drone_pos[j])
                if dist < self.safe_distance:
                    collision_penalty += 1
                    current_collisions += 1
        self.collision_count += current_collisions

        # 使用通信模型计算总通信速率和用户连接情况
        drone_pos_list = self.drone_pos.tolist()
        user_pos_list = self.user_pos.tolist()
        total_rate, connections, snr_matrix, rate_matrix = self.channel_model.calculate_total_rate(
            drone_pos_list, user_pos_list, self.communication_range
        )
        self.user_connections = connections.tolist()

        # 使用感知模型计算覆盖率
        covered_users, coverage_rate, coverage_status = self.sensing_model.calculate_coverage(
            drone_pos_list, user_pos_list, self.sensing_range
        )
        served_users = covered_users

        # 计算能量消耗
        energy_cost = np.sum(self.drone_vel**2)

        # 基于速度的电池消耗模型
        battery_consumption = np.zeros(self.num_drones)
        for i in range(self.num_drones):
            battery_consumption[i] = calculate_battery_consumption(self.drone_vel[i], self.dt, 0.05)
        self.battery = np.clip(self.battery - battery_consumption, 0, 100)

        # 构建奖励函数 - 与论文一致
        # r = α*r_com + β*r_cover - γ*r_penalty
        
        # 计算最小距离
        min_dist = min([calculate_distance(self.drone_pos[0], u) for u in self.user_pos])
        # 平方惩罚 - 距离奖励（负值，越接近0越好）
        dist_reward = - (min_dist / self.area_size) ** 2
        
        # 通信奖励（原始值）
        r_com_raw = total_rate
        # 通信奖励归一化（假设最大速率约为50）
        r_com_norm = np.clip(r_com_raw / 50.0, 0, 1)
        
        # 覆盖率
        r_cover = coverage_rate
        
        # 惩罚项
        speed_penalty = np.sum(np.clip(np.abs(self.drone_vel) - self.max_speed, 0, np.inf)) / self.max_speed
        r_penalty = (out_penalty + collision_penalty) * 0.5 + speed_penalty * 0.05
        
        # 总奖励 - 调整权重，增加覆盖率权重，减小距离惩罚权重
        # reward = 0.2 * r_com_raw + 0.3 * r_cover + 0.5 * dist_reward - r_penalty
        reward = 0.3 * dist_reward + 0.7 * r_cover - r_penalty
        
        # 存储奖励分量，用于调试
        self.r_com_raw = r_com_raw
        self.r_com_norm = r_com_norm
        self.r_cover = r_cover
        self.r_penalty = r_penalty
        self.dist_reward = dist_reward
        self.min_dist = min_dist

        # 检查是否结束
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # 计算平均通信速率
        avg_rate = total_rate / self.num_users if self.num_users > 0 else 0

        info = {
            'served_users': served_users,
            'total_system_rate': total_rate,
            'avg_rate': avg_rate,
            'coverage_rate': coverage_rate,
            'total_rate': total_rate,
            'min_dist': min_dist,
            'dist_reward': dist_reward,
            'distance_reward': dist_reward,  # 别名，兼容训练脚本
            'r_com_raw': r_com_raw,
            'r_com_norm': r_com_norm,
            'r_cover': r_cover,
            'r_penalty': r_penalty,
            'penalty': r_penalty
        }

        obs = self._get_obs()
        return obs, [reward]*self.num_drones, done, False, info

    def _get_obs(self):
        obs_list = []

        # 为每个无人机生成单独的观察
        for drone_idx in range(self.num_drones):
            obs = []

            # 当前无人机位置（只使用x和y坐标）
            obs.extend(self.drone_pos[drone_idx, :2])

            # 当前无人机速度
            obs.extend(self.drone_vel[drone_idx])

            # 越界次数
            obs.append(self.out_of_bounds_count)

            # 碰撞计数
            obs.append(self.collision_count)

            # 电池电量
            obs.append(self.battery[drone_idx])

            # 感知SNR（当前无人机对每个用户）
            for j in range(self.num_users):
                d = calculate_distance(self.drone_pos[drone_idx], self.user_pos[j])
                snr = self.sensing_model.calculate_snr(d)
                obs.append(snr)

            # 用户连接状态（当前无人机对每个用户）
            drone_pos_list = self.drone_pos.tolist()
            user_pos_list = self.user_pos.tolist()
            _, connections, _, _ = self.channel_model.calculate_total_rate(
                drone_pos_list, user_pos_list, self.communication_range
            )
            connection = connections
            for j in range(self.num_users):
                if connection[j] == drone_idx:
                    obs.append(1.0)
                else:
                    obs.append(0.0)

            obs = np.array(obs, dtype=np.float32)
            # 对位置、速度除以区域大小
            obs[:2] /= self.area_size
            obs[2:4] /= self.max_speed  # 速度归一化
            # 越界次数、碰撞计数、电池电量已经在合理范围，可以保持原样或除以最大值
            obs[4] = min(obs[4], 10) / 10  # 越界次数限制在10以内
            obs[5] = min(obs[5], 10) / 10  # 碰撞次数限制在10以内
            obs[6] /= 100  # 电池电量归一化到[0,1]
            # SNR 通常很大，用log压缩
            snr_start = 7
            snr_end = snr_start + self.num_users
            obs[snr_start:snr_end] = np.log1p(obs[snr_start:snr_end]) / np.log(1 + 1e6)  # 压缩到[0,1]左右
            # 连接状态已经是0/1，不需要处理
            obs_list.append(obs)

        # 对于单无人机情况，返回第一个观察
        if self.num_drones == 1:
            return obs_list[0]
        return obs_list

    def get_positions(self):
        """获取无人机和用户的3D位置"""
        drone_pos = self.drone_pos.tolist()
        user_pos = self.user_pos.tolist()
        return drone_pos, user_pos

    def close(self):
        """关闭环境"""
        pass

    def _test_isac_integration(self):
        """
        测试ISAC模块集成
        确保通信和感知模型正确初始化并能正常工作
        """
        print("开始ISAC模块集成测试...")
        
        # 检查通信和感知模型是否被正确初始化
        comm_model_exists = hasattr(self, 'channel_model') and self.channel_model is not None
        sense_model_exists = hasattr(self, 'sensing_model') and self.sensing_model is not None
        
        print(f"通信模型是否存在: {comm_model_exists}")
        print(f"感知模型是否存在: {sense_model_exists}")
        
        # 测试基本功能
        if comm_model_exists and sense_model_exists:
            # 生成测试位置
            test_drone_pos = [[50, 50, 50]]  # 单个无人机在中心位置
            test_user_pos = [[25, 25, 0], [75, 75, 0]]  # 两个用户
            
            # 测试通信模型
            try:
                total_rate, connections, snr_matrix, rate_matrix = self.channel_model.calculate_total_rate(
                    test_drone_pos, test_user_pos, self.communication_range
                )
                print(f"通信模型测试成功，总通信速率: {total_rate:.2f}")
            except Exception as e:
                print(f"通信模型测试失败: {str(e)}")
            
            # 测试感知模型
            try:
                covered_users, coverage_rate, coverage_status = self.sensing_model.calculate_coverage(
                    test_drone_pos, test_user_pos, self.sensing_range
                )
                print(f"感知模型测试成功，覆盖率: {coverage_rate:.2f}")
            except Exception as e:
                print(f"感知模型测试失败: {str(e)}")
            
            print("ISAC模块集成测试完成！")
        else:
            print("ISAC模块集成测试失败：模型未正确初始化")