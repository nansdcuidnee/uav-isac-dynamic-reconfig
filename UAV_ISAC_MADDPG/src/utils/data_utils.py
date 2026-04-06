import numpy as np

def smooth_curve(data, window_size=10):
    """
    平滑曲线
    :param data: 数据
    :param window_size: 窗口大小
    :return: 平滑后的数据
    """
    if window_size < 2:
        return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def calculate_statistics(data_list):
    """
    计算数据的统计信息
    :param data_list: 数据列表
    :return: 包含均值、标准差、最大值、最小值的字典
    """
    data = np.array(data_list)
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'max': np.max(data),
        'min': np.min(data)
    }
