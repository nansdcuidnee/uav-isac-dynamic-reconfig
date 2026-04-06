import numpy as np

def generate_terrain(width, height, resolution, scale=50, octaves=4, persistence=0.5):
    """
    使用简单的噪声生成地形高度图
    :param width: 地形宽度
    :param height: 地形高度
    :param resolution: 分辨率
    :param scale: 噪声缩放因子
    :param octaves: 噪声 octaves
    :param persistence: 噪声持久性
    :return: 地形高度图
    """
    # 使用numpy实现简单的噪声生成
    def simple_noise(x, y, scale):
        """简单的噪声生成函数"""
        return np.sin(x / scale) * np.cos(y / scale) + 0.5 * np.sin(x / (scale * 2)) * np.cos(y / (scale * 2))
    
    terrain = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise = 0
            freq = scale
            amp = 1.0
            for _ in range(octaves):
                noise += simple_noise(x, y, freq) * amp
                freq *= 2
                amp *= persistence
            # 调整高度范围，创建山谷效果
            terrain[y, x] = noise * 30 - 20  # 负值表示山谷
    
    return terrain

def generate_terrain_mesh(x_range, y_range, resolution=50):
    """
    生成地形网格
    :param x_range: X轴范围
    :param y_range: Y轴范围
    :param resolution: 分辨率
    :return: x, y, z 网格
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    width = int((x_max - x_min) / 2)
    height = int((y_max - y_min) / 2)
    
    terrain = generate_terrain(width, height, resolution)
    
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    x, y = np.meshgrid(x, y)
    z = terrain
    
    return x, y, z

def add_terrain_features(ax, x, y, z):
    """
    添加地形特征
    :param ax: 3D轴
    :param x: X坐标网格
    :param y: Y坐标网格
    :param z: Z坐标网格
    """
    # 添加岩石
    rock_positions = []
    for i in range(50):
        rx = np.random.uniform(x.min(), x.max())
        ry = np.random.uniform(y.min(), y.max())
        # 找到对应的地形高度
        xi = int((rx - x.min()) / (x.max() - x.min()) * x.shape[1])
        yi = int((ry - y.min()) / (y.max() - y.min()) * x.shape[0])
        if xi < x.shape[1] and yi < x.shape[0]:
            rz = z[yi, xi] + np.random.uniform(2, 5)
            rock_positions.append((rx, ry, rz))
    
    # 绘制岩石
    if rock_positions:
        rock_x, rock_y, rock_z = zip(*rock_positions)
        ax.scatter(rock_x, rock_y, rock_z, s=30, c='gray', marker='^', label='岩石')
    
    # 添加植被
    tree_positions = []
    for i in range(100):
        tx = np.random.uniform(x.min(), x.max())
        ty = np.random.uniform(y.min(), y.max())
        # 找到对应的地形高度
        xi = int((tx - x.min()) / (x.max() - x.min()) * x.shape[1])
        yi = int((ty - y.min()) / (y.max() - y.min()) * x.shape[0])
        if xi < x.shape[1] and yi < x.shape[0]:
            tz = z[yi, xi] + 1
            tree_positions.append((tx, ty, tz))
    
    # 绘制植被
    if tree_positions:
        tree_x, tree_y, tree_z = zip(*tree_positions)
        ax.scatter(tree_x, tree_y, tree_z, s=20, c='green', marker='^', label='植被')
    
    # 添加水体（低洼地区）
    water_mask = z < -10
    water_x = x[water_mask]
    water_y = y[water_mask]
    water_z = np.full_like(water_x, -10)
    
    if len(water_x) > 0:
        ax.scatter(water_x, water_y, water_z, s=5, c='blue', alpha=0.5, label='水体')