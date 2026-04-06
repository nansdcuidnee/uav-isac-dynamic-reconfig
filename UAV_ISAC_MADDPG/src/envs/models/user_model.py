import numpy as np

class UserModel:
    """
    用户模型 - 描述论文中的用户系统
    """
    
    def __init__(self, user_id, position):
        """
        初始化用户模型
        :param user_id: 用户ID
        :param position: 用户位置 [x, y, z]
        """
        self.user_id = user_id
        self.position = np.array(position, dtype=np.float32)
        self.is_covered = False  # 是否被通信覆盖
        self.is_sensed = False   # 是否被感知覆盖
        self.coverage_history = []  # 覆盖历史
        self.sensing_history = []   # 感知历史
    
    def get_id(self):
        """
        获取用户ID
        :return: 用户ID
        """
        return self.user_id
    
    def get_position(self):
        """
        获取用户位置
        :return: 位置 [x, y, z]
        """
        return self.position.copy()
    
    def is_covered_by_uav(self):
        """
        检查是否被通信覆盖
        :return: 是否被覆盖
        """
        return self.is_covered
    
    def is_sensed_by_uav(self):
        """
        检查是否被感知覆盖
        :return: 是否被感知
        """
        return self.is_sensed
    
    def set_coverage_status(self, covered):
        """
        设置覆盖状态
        :param covered: 是否被覆盖
        """
        self.is_covered = covered
        self.coverage_history.append(covered)
    
    def set_sensing_status(self, sensed):
        """
        设置感知状态
        :param sensed: 是否被感知
        """
        self.is_sensed = sensed
        self.sensing_history.append(sensed)
    
    def get_coverage_history(self):
        """
        获取覆盖历史
        :return: 覆盖历史列表
        """
        return self.coverage_history
    
    def get_sensing_history(self):
        """
        获取感知历史
        :return: 感知历史列表
        """
        return self.sensing_history
    
    def calculate_distance(self, other_position):
        """
        计算到其他位置的距离
        :param other_position: 其他位置 [x, y, z]
        :return: 距离
        """
        return np.sqrt(np.sum((self.position - np.array(other_position))**2))
