import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.animation.animation_3d import draw_quadcopter, create_directory
from visualization.utils.video_utils import calculate_downsampling_rate, save_frame, get_axis_range
from visualization.terrain_generator import generate_terrain_mesh, add_terrain_features

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_trajectory_data(csv_path):
    """
    从CSV文件加载轨迹数据
    :param csv_path: CSV文件路径
    :return: 帧数据列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 按时间戳分组
    grouped = df.groupby('timestamp')
    
    frames_data = []
    trajectories = {}
    
    # 初始化轨迹字典
    drone_ids = df['drone_id'].unique()
    for drone_id in drone_ids:
        trajectories[drone_id] = []
    
    # 处理每个时间戳的数据
    for timestamp, group in grouped:
        # 提取无人机位置
        drone_pos = []
        for drone_id in sorted(drone_ids):
            drone_data = group[group['drone_id'] == drone_id]
            if not drone_data.empty:
                row = drone_data.iloc[0]
                pos = [row['pos_x'], row['pos_y'], row['pos_z']]
                drone_pos.append(pos)
                # 更新轨迹
                trajectories[drone_id].append(pos)
        
        # 生成用户位置（假设有10个用户，位置固定）
        user_pos = [
            [25, 25, 0], [75, 75, 0], [125, 125, 0],
            [175, 175, 0], [225, 225, 0], [275, 275, 0],
            [325, 325, 0], [375, 375, 0], [425, 425, 0],
            [475, 475, 0]
        ]
        
        # 生成连接关系（简单的最近距离分配）
        connections = []
        if drone_pos:
            for user in user_pos:
                # 找到最近的无人机
                min_dist = float('inf')
                closest_drone = None
                for i, drone in enumerate(drone_pos):
                    dist = np.sqrt((user[0] - drone[0])**2 + (user[1] - drone[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_drone = i
                connections.append(closest_drone)
        
        # 准备轨迹数据
        frame_trajectories = []
        for drone_id in sorted(drone_ids):
            frame_trajectories.append(trajectories[drone_id].copy())
        
        # 创建帧数据
        frame_data = {
            'drone_pos': drone_pos,
            'user_pos': user_pos,
            'connections': connections,
            'trajectories': frame_trajectories,
            'step': len(frames_data) + 1
        }
        
        frames_data.append(frame_data)
    
    return frames_data

def create_isac_comparison_animation(with_isac_frames, without_isac_frames, save_path='results/training_videos/3d_isac_comparison_animation.mp4'):
    """
    创建3D对比动画，展示有无辅助通信感知一体化的对比
    :param with_isac_frames: 有辅助通信感知一体化的帧数据
    :param without_isac_frames: 无辅助通信感知一体化的帧数据
    :param save_path: 保存路径
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 降采样
    with_isac_frames_downsampled = with_isac_frames[::3]
    without_isac_frames_downsampled = without_isac_frames[::3]
    total_frames = min(len(with_isac_frames_downsampled), len(without_isac_frames_downsampled))
    print(f"降采样后帧数: {total_frames}")
    
    # 预计算坐标范围
    all_frames = with_isac_frames_downsampled + without_isac_frames_downsampled
    x_min, x_max, y_min, y_max, z_min, z_max = get_axis_range(all_frames, dimensions=3)
    
    # 生成对比动画
    try:
        writer = imageio.get_writer(save_path, fps=24, quality=8, codec='libx264')
        
        # 生成地形网格
        terrain_x, terrain_y, terrain_z = generate_terrain_mesh([x_min, x_max], [y_min, y_max], resolution=50)
        
        for i in range(total_frames):
            fig = plt.figure(figsize=(20, 8), dpi=72)
            
            # 无辅助通信感知一体化场景
            ax1 = fig.add_subplot(121, projection='3d')
            without_frame = without_isac_frames_downsampled[i]
            
            # 绘制地形
            ax1.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
            
            # 添加地形特征
            add_terrain_features(ax1, terrain_x, terrain_y, terrain_z)
            
            # 绘制用户（确保用户在地形上）
            user_pos = np.array(without_frame['user_pos'])
            if user_pos.size > 0:
                # 调整用户位置，确保在地形上
                adjusted_user_pos = user_pos.copy()
                for j in range(len(adjusted_user_pos)):
                    # 找到对应的地形高度
                    xi = int((adjusted_user_pos[j, 0] - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
                    yi = int((adjusted_user_pos[j, 1] - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
                    if xi < terrain_x.shape[1] and yi < terrain_x.shape[0]:
                        adjusted_user_pos[j, 2] = terrain_z[yi, xi] + 1  # 高于地形1米
                ax1.scatter(adjusted_user_pos[:, 0], adjusted_user_pos[:, 1], adjusted_user_pos[:, 2], s=50, c='gray', label='用户')
            
            # 绘制无人机
            drone_pos = np.array(without_frame['drone_pos'])
            if drone_pos.size > 0:
                colors = ['red', 'green', 'blue']
                for j, d in enumerate(drone_pos):
                    color = colors[j % len(colors)]
                    # 绘制四旋翼无人机模型
                    draw_quadcopter(ax1, d, color=color, size=3)
                    
                    # 绘制通信范围
                    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                    r = 50  # 通信范围半径
                    x = d[0] + r * np.cos(u) * np.sin(v)
                    y = d[1] + r * np.sin(u) * np.sin(v)
                    z = d[2] + r * np.cos(v)
                    ax1.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')
                    
                    # 绘制感知范围
                    r_sense = 60  # 感知范围半径
                    x = d[0] + r_sense * np.cos(u) * np.sin(v)
                    y = d[1] + r_sense * np.sin(u) * np.sin(v)
                    z = d[2] + r_sense * np.cos(v)
                    ax1.plot_surface(x, y, z, color='green', alpha=0.08, edgecolor='none')
            
            # 绘制轨迹
            trajectories = without_frame.get('trajectories', [])
            if trajectories:
                for j, traj in enumerate(trajectories):
                    traj = np.array(traj)
                    if traj.size > 0:
                        color = colors[j % len(colors)]
                        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='-', linewidth=3, color=color, label=f'轨迹{j+1}')
            
            # 绘制连接线
            connections = without_frame.get('connections')
            if connections is not None and drone_pos.size > 0 and user_pos.size > 0:
                for user_idx, drone_idx in enumerate(connections):
                    if drone_idx is not None:
                        # 调整用户位置，确保在地形上
                        xi = int((user_pos[user_idx, 0] - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
                        yi = int((user_pos[user_idx, 1] - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
                        user_z = terrain_z[yi, xi] + 1 if xi < terrain_x.shape[1] and yi < terrain_x.shape[0] else user_pos[user_idx, 2]
                        
                        xs = [user_pos[user_idx, 0], drone_pos[drone_idx, 0]]
                        ys = [user_pos[user_idx, 1], drone_pos[drone_idx, 1]]
                        zs = [user_z, drone_pos[drone_idx, 2]]
                        ax1.plot(xs, ys, zs, linewidth=1, color='green')
            
            ax1.set_title('无辅助通信感知一体化')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])
            ax1.set_zlim([z_min, z_max])
            ax1.view_init(elev=30, azim=45)
            
            # 有辅助通信感知一体化场景
            ax2 = fig.add_subplot(122, projection='3d')
            with_frame = with_isac_frames_downsampled[i]
            
            # 绘制地形
            ax2.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
            
            # 添加地形特征
            add_terrain_features(ax2, terrain_x, terrain_y, terrain_z)
            
            # 绘制用户（确保用户在地形上）
            user_pos = np.array(with_frame['user_pos'])
            if user_pos.size > 0:
                # 调整用户位置，确保在地形上
                adjusted_user_pos = user_pos.copy()
                for j in range(len(adjusted_user_pos)):
                    # 找到对应的地形高度
                    xi = int((adjusted_user_pos[j, 0] - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
                    yi = int((adjusted_user_pos[j, 1] - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
                    if xi < terrain_x.shape[1] and yi < terrain_x.shape[0]:
                        adjusted_user_pos[j, 2] = terrain_z[yi, xi] + 1  # 高于地形1米
                ax2.scatter(adjusted_user_pos[:, 0], adjusted_user_pos[:, 1], adjusted_user_pos[:, 2], s=50, c='gray', label='用户')
            
            # 绘制无人机
            drone_pos = np.array(with_frame['drone_pos'])
            if drone_pos.size > 0:
                colors = ['red', 'green', 'blue']
                for j, d in enumerate(drone_pos):
                    color = colors[j % len(colors)]
                    # 绘制四旋翼无人机模型
                    draw_quadcopter(ax2, d, color=color, size=3)
                    
                    # 绘制通信范围（更大的范围）
                    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                    r = 70  # 更大的通信范围半径
                    x = d[0] + r * np.cos(u) * np.sin(v)
                    y = d[1] + r * np.sin(u) * np.sin(v)
                    z = d[2] + r * np.cos(v)
                    ax2.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')
                    
                    # 绘制感知范围（更大的范围）
                    r_sense = 80  # 更大的感知范围半径
                    x = d[0] + r_sense * np.cos(u) * np.sin(v)
                    y = d[1] + r_sense * np.sin(u) * np.sin(v)
                    z = d[2] + r_sense * np.cos(v)
                    ax2.plot_surface(x, y, z, color='green', alpha=0.08, edgecolor='none')
            
            # 绘制轨迹
            trajectories = with_frame.get('trajectories', [])
            if trajectories:
                for j, traj in enumerate(trajectories):
                    traj = np.array(traj)
                    if traj.size > 0:
                        color = colors[j % len(colors)]
                        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='-', linewidth=3, color=color, label=f'轨迹{j+1}')
            
            # 绘制连接线
            connections = with_frame.get('connections')
            if connections is not None and drone_pos.size > 0 and user_pos.size > 0:
                for user_idx, drone_idx in enumerate(connections):
                    if drone_idx is not None:
                        # 调整用户位置，确保在地形上
                        xi = int((user_pos[user_idx, 0] - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
                        yi = int((user_pos[user_idx, 1] - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
                        user_z = terrain_z[yi, xi] + 1 if xi < terrain_x.shape[1] and yi < terrain_x.shape[0] else user_pos[user_idx, 2]
                        
                        xs = [user_pos[user_idx, 0], drone_pos[drone_idx, 0]]
                        ys = [user_pos[user_idx, 1], drone_pos[drone_idx, 1]]
                        zs = [user_z, drone_pos[drone_idx, 2]]
                        ax2.plot(xs, ys, zs, linewidth=1, color='green')
            
            ax2.set_title('有辅助通信感知一体化')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Z (m)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim([x_min, x_max])
            ax2.set_ylim([y_min, y_max])
            ax2.set_zlim([z_min, z_max])
            ax2.view_init(elev=30, azim=45)
            
            plt.suptitle(f'有无辅助通信感知一体化对比 - 第{i+1}帧')
            plt.tight_layout()
            
            # 保存为图像
            save_frame(fig, writer)
            
            # 每20帧打印一次进度
            if (i + 1) % 20 == 0 or (i + 1) == total_frames:
                print(f"已生成 {i + 1}/{total_frames} 帧")
        
        writer.close()
        print(f"3D对比动画已保存到 {save_path}")
    except Exception as e:
        print(f"生成3D对比动画失败: {e}")

def generate_test_data(num_drones=3, num_steps=300):
    """
    生成测试轨迹数据
    :param num_drones: 无人机数量
    :param num_steps: 步数
    :return: 帧数据列表
    """
    frames_data = []
    trajectories = {}
    
    # 初始化轨迹
    for i in range(num_drones):
        trajectories[i] = []
    
    for step in range(num_steps):
        drone_pos = []
        for i in range(num_drones):
            # 生成简单的轨迹数据
            x = 100 + i * 100 + step * 0.5
            y = 100 + i * 50
            z = 100 + step * 0.1
            pos = [x, y, z]
            drone_pos.append(pos)
            trajectories[i].append(pos)
        
        # 生成用户位置
        user_pos = [
            [25, 25, 0], [75, 75, 0], [125, 125, 0],
            [175, 175, 0], [225, 225, 0], [275, 275, 0],
            [325, 325, 0], [375, 375, 0], [425, 425, 0],
            [475, 475, 0]
        ]
        
        # 生成连接关系
        connections = []
        if drone_pos:
            for user in user_pos:
                min_dist = float('inf')
                closest_drone = None
                for j, drone in enumerate(drone_pos):
                    dist = np.sqrt((user[0] - drone[0])**2 + (user[1] - drone[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_drone = j
                connections.append(closest_drone)
        
        # 准备轨迹数据
        frame_trajectories = []
        for i in range(num_drones):
            frame_trajectories.append(trajectories[i].copy())
        
        # 创建帧数据
        frame_data = {
            'drone_pos': drone_pos,
            'user_pos': user_pos,
            'connections': connections,
            'trajectories': frame_trajectories,
            'step': step + 1
        }
        
        frames_data.append(frame_data)
    
    return frames_data

def main():
    print("开始生成有无辅助通信感知一体化对比视频...")
    
    # 生成测试数据
    print("生成测试轨迹数据...")
    with_isac_frames = generate_test_data(num_drones=3, num_steps=300)
    without_isac_frames = generate_test_data(num_drones=3, num_steps=300)
    
    # 生成3D对比视频
    save_path = 'd:\UAV-Communication-MADDPG\UAV_ISAC_MADDPG\results\training_videos\3d_isac_comparison_animation.mp4'
    print("生成3D对比视频...")
    create_isac_comparison_animation(with_isac_frames, without_isac_frames, save_path=save_path)
    
    print("视频生成完成！")

if __name__ == "__main__":
    main()