import numpy as np

class EnvironmentModel:
    """
    环境模型 - 描述论文中的系统环境
    """
    
    def __init__(self, area_size=200.0, time_steps=1000):
        """
        初始化环境模型
        :param area_size: 环境区域大小 (m)
        :param time_steps: 总时间步数
        """
        self.area_size = area_size
        self.time_steps = time_steps
        self.current_time = 0
        
    def get_area_size(self):
        """
        获取环境区域大小
        :return: 区域大小
        """
        return self.area_size
    
    def get_time_steps(self):
        """
        获取总时间步数
        :return: 总时间步数
        """
        return self.time_steps
    
    def get_current_time(self):
        """
        获取当前时间步
        :return: 当前时间步
        """
        return self.current_time
    
    def advance_time(self):
        """
        时间步进
        """
        if self.current_time < self.time_steps - 1:
            self.current_time += 1
            return True
        return False
    
    def reset_time(self):
        """
        重置时间
        """
        self.current_time = 0
    
    def is_position_valid(self, position):
        """
        检查位置是否在有效区域内
        :param position: 位置 [x, y, z]
        :return: 是否有效
        """
        x, y, z = position
        return 0 <= x <= self.area_size and 0 <= y <= self.area_size and z >= 0
    
    def clip_position(self, position):
        """
        裁剪位置到有效区域
        :param position: 位置 [x, y, z]
        :return: 裁剪后的位置
        """
        x, y, z = position
        x = max(0, min(self.area_size, x))
        y = max(0, min(self.area_size, y))
        z = max(0, z)
        return [x, y, z]
