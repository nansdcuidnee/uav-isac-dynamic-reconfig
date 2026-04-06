import numpy as np

class UAVDynamics:
    """
    无人机运动模型
    - 位置更新
    - 速度限制
    - 碰撞检测
    """

    def __init__(self, max_speed=5.0, safe_distance=5.0, dt=1.0):
        """
        初始化无人机运动模型
        :param max_speed: 最大速度
        :param safe_distance: 安全距离
        :param dt: 时间步长
        """
        self.max_speed = max_speed
        self.safe_distance = safe_distance
        self.dt = dt

    def update_position(self, drone_pos, drone_vel, actions):
        """
        更新无人机位置
        :param drone_pos: 无人机位置
        :param drone_vel: 无人机速度
        :param actions: 动作
        :return: 更新后的位置和速度
        """
        # 确保actions是numpy数组
        actions = np.array(actions)

        # 处理actions的维度
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        elif actions.ndim == 2:
            if actions.shape[0] != drone_pos.shape[0]:
                actions = actions.reshape(drone_pos.shape[0], -1)
        elif actions.ndim == 3:
            actions = actions.reshape(drone_pos.shape[0], -1)

        # 确保actions的形状与drone_vel匹配
        actions = actions[:drone_pos.shape[0], :2]
        actions = np.clip(actions, -1, 1)

        # 更新速度和位置
        drone_vel = 0.9 * drone_vel + 0.1 * (actions * self.max_speed)
        drone_vel = np.clip(drone_vel, -self.max_speed, self.max_speed)
        drone_pos[:, :2] += drone_vel * self.dt

        return drone_pos, drone_vel

    def check_collisions(self, drone_pos):
        """
        检测碰撞
        :param drone_pos: 无人机位置
        :return: 碰撞次数
        """
        collision_count = 0
        for i in range(drone_pos.shape[0]):
            for j in range(i + 1, drone_pos.shape[0]):
                dist = np.linalg.norm(drone_pos[i, :2] - drone_pos[j, :2])
                if dist < self.safe_distance:
                    collision_count += 1
        return collision_count

    def check_boundaries(self, drone_pos, area_size):
        """
        检测边界
        :param drone_pos: 无人机位置
        :param area_size: 区域大小
        :return: 越界次数和裁剪后的位置
        """
        out_of_bounds = np.logical_or(drone_pos[:, :2] < 0, drone_pos[:, :2] > area_size)
        out_penalty = np.sum(out_of_bounds)
        drone_pos[:, :2] = np.clip(drone_pos[:, :2], 0, area_size)
        return out_penalty, drone_pos