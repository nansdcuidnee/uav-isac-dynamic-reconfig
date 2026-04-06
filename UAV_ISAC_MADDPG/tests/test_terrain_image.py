import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 地形生成函数
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
    :param x_range: x坐标范围 [min, max]
    :param y_range: y坐标范围 [min, max]
    :param resolution: 分辨率
    :return: (x, y, z) 网格
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    x, y = np.meshgrid(x, y)
    
    # 生成地形高度
    z = generate_terrain(resolution, resolution, resolution)
    
    return x, y, z

def add_terrain_features(ax, terrain_x, terrain_y, terrain_z):
    """
    添加地形特征
    :param ax: 3D轴对象
    :param terrain_x: 地形x坐标
    :param terrain_y: 地形y坐标
    :param terrain_z: 地形z坐标
    """
    # 添加岩石
    rock_count = 50
    for _ in range(rock_count):
        rx = np.random.uniform(terrain_x.min(), terrain_x.max())
        ry = np.random.uniform(terrain_y.min(), terrain_y.max())
        # 找到对应的地形高度
        xi = int((rx - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
        yi = int((ry - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
        if xi < terrain_x.shape[1] and yi < terrain_x.shape[0]:
            rz = terrain_z[yi, xi] + np.random.uniform(1, 3)
            # 绘制岩石
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
            rock_radius = np.random.uniform(0.5, 1.5)
            rock_x = rx + rock_radius * np.cos(u) * np.sin(v)
            rock_y = ry + rock_radius * np.sin(u) * np.sin(v)
            rock_z = rz + rock_radius * np.cos(v)
            ax.plot_wireframe(rock_x, rock_y, rock_z, color='gray', alpha=0.7)
    
    # 添加植被
    vegetation_count = 100
    for _ in range(vegetation_count):
        vx = np.random.uniform(terrain_x.min(), terrain_x.max())
        vy = np.random.uniform(terrain_y.min(), terrain_y.max())
        # 找到对应的地形高度
        xi = int((vx - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
        yi = int((vy - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
        if xi < terrain_x.shape[1] and yi < terrain_x.shape[0]:
            vz = terrain_z[yi, xi] + 1
            # 绘制植被（简单的圆柱体）
            height = np.random.uniform(1, 3)
            radius = np.random.uniform(0.2, 0.5)
            z = np.linspace(vz, vz + height, 10)
            theta = np.linspace(0, 2*np.pi, 10)
            theta, z = np.meshgrid(theta, z)
            veg_x = vx + radius * np.cos(theta)
            veg_y = vy + radius * np.sin(theta)
            veg_z = z
            ax.plot_surface(veg_x, veg_y, veg_z, color='green', alpha=0.6)
    
    # 添加水体（在低洼地区）
    water_level = -10
    water_mask = terrain_z < water_level
    if np.any(water_mask):
        # 找到水体区域
        water_x = terrain_x[water_mask]
        water_y = terrain_y[water_mask]
        water_z = np.full_like(water_x, water_level)
        # 绘制水体
        ax.scatter(water_x, water_y, water_z, s=1, c='blue', alpha=0.5)

if __name__ == "__main__":
    # 生成地形网格
    x_range = [0, 200]
    y_range = [0, 200]
    terrain_x, terrain_y, terrain_z = generate_terrain_mesh(x_range, y_range, resolution=50)
    
    # 创建3D图像
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地形
    ax.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
    
    # 添加地形特征
    add_terrain_features(ax, terrain_x, terrain_y, terrain_z)
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Terrain with Features')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 确保保存目录存在
    save_dir = 'results/training_videos'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    save_path = os.path.join(save_dir, 'terrain_test.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"地形测试图像已保存到: {save_path}")
    
    # 显示图像
    plt.show()
