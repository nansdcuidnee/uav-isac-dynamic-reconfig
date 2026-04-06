import numpy as np

def calculate_distance(pos1, pos2):
    """
    计算两点之间的距离
    :param pos1: 第一个点的位置
    :param pos2: 第二个点的位置
    :return: 距离
    """
    return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

def normalize_vector(vector):
    """
    归一化向量
    :param vector: 输入向量
    :return: 归一化后的向量
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def clip_position(position, area_size):
    """
    裁剪位置到区域内
    :param position: 位置
    :param area_size: 区域大小
    :return: 裁剪后的位置
    """
    position = np.array(position)
    position[:2] = np.clip(position[:2], 0, area_size)
    return position

def calculate_battery_consumption(velocity, dt, consumption_rate=0.05):
    """
    计算电池消耗
    :param velocity: 速度
    :param dt: 时间步长
    :param consumption_rate: 消耗率
    :return: 电池消耗
    """
    distance = np.linalg.norm(velocity) * dt
    return distance * consumption_rate

def generate_random_position(area_size, height=50.0):
    """
    生成随机位置
    :param area_size: 区域大小
    :param height: 高度
    :return: 随机位置
    """
    x = np.random.uniform(0, area_size)
    y = np.random.uniform(0, area_size)
    return [x, y, height]
