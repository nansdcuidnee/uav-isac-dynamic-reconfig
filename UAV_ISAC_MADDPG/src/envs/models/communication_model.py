import numpy as np

class CommunicationModel:
    """
    通信模型 - 描述论文中的通信系统
    """
    
    def __init__(self, P=100.0, noise=1e-3):
        """
        初始化通信模型
        :param P: 发射功率
        :param noise: 噪声功率
        """
        self.P = P
        self.noise = noise
    
    def calculate_channel_gain(self, distance):
        """
        计算信道增益
        :param distance: 距离
        :return: 信道增益
        """
        if distance == 0:
            return self.P / self.noise
        return self.P / (distance**2 + self.noise)
    
    def calculate_data_rate(self, snr):
        """
        计算数据速率（香农公式）
        :param snr: 信噪比
        :return: 数据速率
        """
        return np.log2(1 + snr)
    
    def calculate_total_rate(self, drone_pos, user_pos, communication_range):
        """
        计算总通信速率
        :param drone_pos: 无人机位置
        :param user_pos: 用户位置
        :param communication_range: 通信范围
        :return: 总通信速率和用户连接情况
        """
        num_drones = len(drone_pos)
        num_users = len(user_pos)
        snr_matrix = np.zeros((num_drones, num_users))
        rate_matrix = np.zeros((num_drones, num_users))

        # 计算每个无人机到每个用户的SNR和速率
        for i in range(num_drones):
            for j in range(num_users):
                distance = np.linalg.norm(np.array(drone_pos[i][:2]) - np.array(user_pos[j][:2]))
                if distance <= communication_range:
                    snr = self.calculate_channel_gain(distance)
                    rate = self.calculate_data_rate(snr)
                else:
                    snr = 0
                    rate = 0
                snr_matrix[i, j] = snr
                rate_matrix[i, j] = rate

        # 用户连接到SNR最大的无人机
        connections = np.argmax(snr_matrix, axis=0)
        total_rate = 0
        served_users = 0

        for j in range(num_users):
            best_drone = connections[j]
            total_rate += rate_matrix[best_drone, j]
            if snr_matrix[best_drone, j] > 0:
                served_users += 1

        return total_rate, connections, snr_matrix, rate_matrix
    
    def get_user_connections(self, drone_pos, user_pos, communication_range):
        """
        获取用户连接情况
        :param drone_pos: 无人机位置
        :param user_pos: 用户位置
        :param communication_range: 通信范围
        :return: 用户连接情况
        """
        _, connections, _, _ = self.calculate_total_rate(drone_pos, user_pos, communication_range)
        return connections
