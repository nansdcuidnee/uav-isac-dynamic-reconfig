import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_directory(dir_path):
    """
    创建目录
    :param dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def calculate_downsampling_rate(frames_data):
    """
    计算降采样率
    :param frames_data: 帧数据列表
    :return: 降采样率
    """
    total_frames = len(frames_data)
    if total_frames > 1000:
        return 5
    elif total_frames > 500:
        return 3
    elif total_frames > 200:
        return 2
    else:
        return 1

def get_axis_range(frames_data, dimensions=2):
    """
    获取坐标轴范围
    :param frames_data: 帧数据列表
    :param dimensions: 维度
    :return: 坐标轴范围
    """
    all_pos = []
    for frame in frames_data:
        drone_pos = frame.get('drone_pos', [])
        user_pos = frame.get('user_pos', [])
        all_pos.extend(drone_pos)
        all_pos.extend(user_pos)
    
    if not all_pos:
        return 0, 100, 0, 100
    
    all_pos = np.array(all_pos)
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    
    # 添加一些边距
    margin = (x_max - x_min) * 0.1 if x_max > x_min else 10
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    if dimensions == 3:
        z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()
        z_margin = (z_max - z_min) * 0.1 if z_max > z_min else 10
        z_min -= z_margin
        z_max += z_margin
        return x_min, x_max, y_min, y_max, z_min, z_max
    else:
        return x_min, x_max, y_min, y_max

def get_height_range(frames_data):
    """
    获取高度范围
    :param frames_data: 帧数据列表
    :return: 最小高度, 最大高度, 高度范围
    """
    all_heights = []
    for frame in frames_data:
        drone_pos = frame.get('drone_pos', [])
        for pos in drone_pos:
            if len(pos) >= 3:
                all_heights.append(pos[2])
    
    if not all_heights:
        return 0, 100, 100
    
    min_height = min(all_heights)
    max_height = max(all_heights)
    height_range = max_height - min_height if max_height > min_height else 1
    
    return min_height, max_height, height_range

def save_frame(fig, writer):
    """
    保存帧
    :param fig: 图形对象
    :param writer: 写入器
    """
    import io
    from PIL import Image
    
    # 将图形保存到内存
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    # 读取图像并添加到视频
    image = Image.open(buffer)
    frame = np.array(image)
    writer.append_data(frame)
    
    # 清理
    plt.close(fig)

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
        
        # 生成用户位置（假设有3个用户，位置固定）
        user_pos = [[25, 25, 0], [75, 75, 0], [125, 125, 0]]
        
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

def create_trajectory_animation(frames_data, save_path='results/training_videos/trajectory_animation.mp4', max_steps=None):
    """
    创建轨迹动画
    :param frames_data: 帧数据列表
    :param save_path: 保存路径
    :param max_steps: 每个回合的最大步数
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 只保存最后一个回合的帧
    if max_steps and max_steps > 0 and len(frames_data) > max_steps:
        last_episode_frames = frames_data[-max_steps:]
        print(f"只保存最后一个回合的 {len(last_episode_frames)} 帧")
        frames_data = last_episode_frames
    
    # 优化帧率设置
    fps = 24
    
    # 优化降采样策略
    downsampling_rate = calculate_downsampling_rate(frames_data)
    
    frames_data_downsampled = frames_data[::downsampling_rate]
    total_frames = len(frames_data_downsampled)
    print(f"降采样后帧数: {total_frames}")
    print(f"降采样率: 每{downsampling_rate}帧取一帧")
    
    # 计算预计视频时长
    estimated_duration = total_frames / fps
    print(f"预计视频时长: {estimated_duration:.1f}秒 ({estimated_duration/60:.1f}分钟)")
    
    # 预计算高度范围
    min_height, max_height, height_range = get_height_range(frames_data)
    print(f"高度范围: {min_height:.1f}m - {max_height:.1f}m")
    
    # 使用imageio.get_writer逐帧写入
    try:
        writer = imageio.get_writer(save_path, fps=fps, quality=8, codec='libx264')
        
        # 预计算所有帧的坐标范围
        x_min, x_max, y_min, y_max = get_axis_range(frames_data_downsampled, dimensions=2)
        
        for i, frame_data in enumerate(frames_data_downsampled):
            # 优化绘图设置
            fig = plt.figure(figsize=(8, 6), dpi=72)
            ax = fig.add_subplot(111)
            
            # 绘制用户
            user_pos = np.array(frame_data['user_pos'])
            if user_pos.size > 0:
                ax.scatter(user_pos[:, 0], user_pos[:, 1], s=50, c='gray', label='用户')
            
            # 绘制无人机
            drone_pos = np.array(frame_data['drone_pos'])
            if drone_pos.size > 0:
                for j, d in enumerate(drone_pos):
                    # 优化高度归一化
                    height_normalized = (d[2] - min_height) / height_range
                    height_normalized = max(0, min(1, height_normalized))
                    
                    # 使用更明显的颜色映射
                    color = plt.cm.jet(height_normalized)
                    
                    # 优化无人机大小随高度变化
                    base_size = 80
                    size_range = 120
                    marker_size = base_size + height_normalized * size_range
                    
                    ax.scatter(d[0], d[1], s=marker_size, c=[color], 
                               label=f'无人机{j+1} (高度: {d[2]:.1f}m)')
                    
                    # 绘制通信范围
                    circle = plt.Circle((d[0], d[1]), 50, fill=False, color='blue', linestyle='--', alpha=0.5)
                    ax.add_patch(circle)
                    # 绘制感知范围
                    circle = plt.Circle((d[0], d[1]), 60, fill=False, color='red', linestyle='--', alpha=0.3)
                    ax.add_patch(circle)
            
            # 绘制轨迹
            trajectories = frame_data.get('trajectories', [])
            for j, traj in enumerate(trajectories):
                traj = np.array(traj)
                if traj.size > 0:
                    for k in range(len(traj) - 1):
                        height1 = traj[k, 2]
                        height2 = traj[k+1, 2]
                        h1_norm = max(0, min(1, (height1 - min_height) / height_range))
                        h2_norm = max(0, min(1, (height2 - min_height) / height_range))
                        color1 = plt.cm.jet(h1_norm)
                        color2 = plt.cm.jet(h2_norm)
                        
                        ax.plot([traj[k, 0], traj[k+1, 0]], [traj[k, 1], traj[k+1, 1]], 
                                 linestyle='-', linewidth=3, color=color1, alpha=0.8)
            
            # 绘制连接线
            connections = frame_data.get('connections')
            if connections is not None and drone_pos.size > 0 and user_pos.size > 0:
                for user_idx, drone_idx in enumerate(connections):
                    if drone_idx is not None:
                        xs = [user_pos[user_idx, 0], drone_pos[drone_idx, 0]]
                        ys = [user_pos[user_idx, 1], drone_pos[drone_idx, 1]]
                        ax.plot(xs, ys, linewidth=1, color='green')
            
            ax.set_title(f'无人机轨迹动画 - 第{i+1}帧')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.grid(True)
            ax.set_aspect('equal')
            
            # 设置固定的坐标轴范围
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            
            # 调整布局
            plt.tight_layout()
            
            # 保存为图像
            save_frame(fig, writer)
            
            # 每20帧打印一次进度
            if (i + 1) % 20 == 0 or (i + 1) == total_frames:
                print(f"已生成 {i + 1}/{total_frames} 帧")
        writer.close()
        print(f"轨迹动画已保存到 {save_path}")
    except Exception as e:
        print(f"生成轨迹动画失败: {e}")

def main():
    # 加载MADDPG轨迹数据
    maddpg_csv_path = 'results/training_videos/maddpg_trajectory.csv'
    
    if not os.path.exists(maddpg_csv_path):
        print(f"文件不存在: {maddpg_csv_path}")
        return
    
    print("加载MADDPG轨迹数据...")
    maddpg_frames = load_trajectory_data(maddpg_csv_path)
    print(f"加载完成，共 {len(maddpg_frames)} 帧")
    
    # 生成2D轨迹动画
    print("\n生成2D轨迹动画...")
    create_trajectory_animation(maddpg_frames)

if __name__ == "__main__":
    main()