import numpy as np

class SensingModel:
    """
    感知模型 - 描述论文中的感知系统
    """
    
    def __init__(self, P=100.0, noise=1e-3, snr_threshold=0.5):
        """
        初始化感知模型
        :param P: 发射功率
        :param noise: 噪声功率
        :param snr_threshold: 感知信噪比阈值
        """
        self.P = P
        self.noise = noise
        self.snr_threshold = snr_threshold
    
    def calculate_snr(self, distance):
        """
        计算感知SNR
        :param distance: 距离
        :return: SNR值
        """
        if distance == 0:
            return self.P / self.noise
        return self.P / (distance**2 + self.noise)
    
    def is_within_sensing_range(self, distance, sensing_range):
        """
        判断是否在感知范围内
        :param distance: 距离
        :param sensing_range: 感知范围
        :return: 是否在感知范围内
        """
        return distance <= sensing_range
    
    def calculate_coverage(self, drone_pos, user_pos, sensing_range):
        """
        计算感知覆盖情况
        :param drone_pos: 无人机位置
        :param user_pos: 用户位置
        :param sensing_range: 感知范围
        :return: 覆盖用户数和覆盖状态
        """
        num_drones = len(drone_pos)
        num_users = len(user_pos)
        coverage_status = np.zeros(num_users, dtype=bool)

        for j in range(num_users):
            for i in range(num_drones):
                distance = np.linalg.norm(np.array(drone_pos[i][:2]) - np.array(user_pos[j][:2]))
                if self.is_within_sensing_range(distance, sensing_range):
                    snr = self.calculate_snr(distance)
                    if snr >= self.snr_threshold:
                        coverage_status[j] = True
                        break

        covered_users = np.sum(coverage_status)
        coverage_rate = covered_users / num_users if num_users > 0 else 0

        return covered_users, coverage_rate, coverage_status
    
    def get_sensing_coverage(self, drone_pos, user_pos, sensing_range):
        """
        获取感知覆盖状态
        :param drone_pos: 无人机位置
        :param user_pos: 用户位置
        :param sensing_range: 感知范围
        :return: 覆盖状态
        """
        _, _, coverage_status = self.calculate_coverage(drone_pos, user_pos, sensing_range)
        return coverage_status
