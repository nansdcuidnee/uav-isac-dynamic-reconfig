import numpy as np

class UAVModel:
    """
    UAV模型 - 描述论文中的无人机系统
    """
    
    def __init__(self, uav_id, initial_position, max_speed=10.0, communication_range=100.0, sensing_range=120.0):
        """
        初始化UAV模型
        :param uav_id: 无人机ID
        :param initial_position: 初始位置 [x, y, z]
        :param max_speed: 最大速度 (m/s)
        :param communication_range: 通信范围 (m)
        :param sensing_range: 感知范围 (m)
        """
        self.uav_id = uav_id
        self.position = np.array(initial_position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)  # 2D速度
        self.max_speed = max_speed
        self.communication_range = communication_range
        self.sensing_range = sensing_range
        self.trajectory = [initial_position.copy()]
    
    def get_id(self):
        """
        获取无人机ID
        :return: 无人机ID
        """
        return self.uav_id
    
    def get_position(self):
        """
        获取当前位置
        :return: 位置 [x, y, z]
        """
        return self.position.copy()
    
    def get_velocity(self):
        """
        获取当前速度
        :return: 速度 [vx, vy]
        """
        return self.velocity.copy()
    
    def get_communication_range(self):
        """
        获取通信范围
        :return: 通信范围
        """
        return self.communication_range
    
    def get_sensing_range(self):
        """
        获取感知范围
        :return: 感知范围
        """
        return self.sensing_range
    
    def get_trajectory(self):
        """
        获取飞行轨迹
        :return: 轨迹列表
        """
        return np.array(self.trajectory)
    
    def update_position(self, new_position):
        """
        更新位置
        :param new_position: 新位置 [x, y, z]
        """
        self.position = np.array(new_position, dtype=np.float32)
        self.trajectory.append(new_position.copy())
    
    def update_velocity(self, new_velocity):
        """
        更新速度
        :param new_velocity: 新速度 [vx, vy]
        """
        self.velocity = np.clip(np.array(new_velocity, dtype=np.float32), -self.max_speed, self.max_speed)
    
    def calculate_distance(self, other_position):
        """
        计算到其他位置的距离
        :param other_position: 其他位置 [x, y, z]
        :return: 距离
        """
        return np.sqrt(np.sum((self.position - np.array(other_position))**2))
    
    def is_in_communication_range(self, other_position):
        """
        检查是否在通信范围内
        :param other_position: 其他位置 [x, y, z]
        :return: 是否在通信范围内
        """
        return self.calculate_distance(other_position) <= self.communication_range
    
    def is_in_sensing_range(self, other_position):
        """
        检查是否在感知范围内
        :param other_position: 其他位置 [x, y, z]
        :return: 是否在感知范围内
        """
        return self.calculate_distance(other_position) <= self.sensing_range
