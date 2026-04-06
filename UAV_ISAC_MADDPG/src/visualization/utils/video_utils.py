import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_downsampling_rate(frames_data):
    """
    计算合理的降采样率
    :param frames_data: 帧数据列表
    :return: 降采样率
    """
    if len(frames_data) > 5000:
        return 20  # 对于超长序列，每20帧取一帧
    elif len(frames_data) > 2000:
        return 10  # 对于长序列，每10帧取一帧
    elif len(frames_data) > 1000:
        return 5  # 对于中等长度序列，每5帧取一帧
    else:
        return 3  # 对于短序列，每3帧取一帧

def save_frame(fig, writer):
    """
    保存帧到视频写入器
    :param fig: Matplotlib图形对象
    :param writer: imageio写入器
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    image = imageio.imread(buf)
    writer.append_data(image)
    plt.close()
    plt.clf()
    plt.close('all')

def get_axis_range(frames_data, dimensions=2):
    """
    获取坐标轴范围
    :param frames_data: 帧数据列表
    :param dimensions: 维度数（2或3）
    :return: 坐标轴范围
    """
    all_x = []
    all_y = []
    all_z = []
    
    for frame_data in frames_data:
        user_pos = np.array(frame_data['user_pos'])
        if user_pos.size > 0:
            all_x.extend(user_pos[:, 0])
            all_y.extend(user_pos[:, 1])
            if dimensions == 3 and user_pos.shape[1] > 2:
                all_z.extend(user_pos[:, 2])
        
        drone_pos = np.array(frame_data['drone_pos'])
        if drone_pos.size > 0:
            all_x.extend(drone_pos[:, 0])
            all_y.extend(drone_pos[:, 1])
            if dimensions == 3 and drone_pos.shape[1] > 2:
                all_z.extend(drone_pos[:, 2])
    
    # 设置固定的坐标轴范围，留出一定的边距
    if all_x and all_y:
        x_min, x_max = min(all_x) - 10, max(all_x) + 10
        y_min, y_max = min(all_y) - 10, max(all_y) + 10
    else:
        x_min, x_max, y_min, y_max = 0, 100, 0, 100
    
    if dimensions == 3:
        if all_z:
            z_min, z_max = min(all_z) - 5, max(all_z) + 5
        else:
            z_min, z_max = 0, 100
        return x_min, x_max, y_min, y_max, z_min, z_max
    else:
        return x_min, x_max, y_min, y_max

def get_height_range(frames_data):
    """
    获取高度范围
    :param frames_data: 帧数据列表
    :return: 最小高度、最大高度、高度范围
    """
    all_heights = []
    for frame_data in frames_data:
        drone_pos = np.array(frame_data['drone_pos'])
        if drone_pos.size > 0:
            all_heights.extend(drone_pos[:, 2])
    
    if all_heights:
        min_height = min(all_heights)
        max_height = max(all_heights)
        height_range = max_height - min_height
        if height_range == 0:
            height_range = 1  # 避免除零
    else:
        min_height = 40
        max_height = 60
        height_range = 20
    
    return min_height, max_height, height_range

def create_directory(dir_path):
    """
    创建目录
    :param dir_path: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)