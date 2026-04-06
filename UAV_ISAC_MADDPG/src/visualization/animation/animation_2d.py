import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.visualization.utils.video_utils import (
    calculate_downsampling_rate,
    save_frame,
    get_axis_range,
    get_height_range,
    create_directory
)

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_trajectory_animation(frames_data, save_path='results/training_videos/trajectory_animation.mp4', max_steps=None):
    """
    创建轨迹动画
    :param frames_data: 帧数据列表，每个元素是 {'drone_pos': [...], 'user_pos': [...], 'connections': [...], 'trajectories': [...]}
    :param save_path: 保存路径
    :param max_steps: 每个回合的最大步数，如果提供，只保存最后一个回合的帧
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 只保存最后一个回合的帧（如果提供了max_steps）
    if max_steps and max_steps > 0 and len(frames_data) > max_steps:
        last_episode_frames = frames_data[-max_steps:]
        print(f"只保存最后一个回合的 {len(last_episode_frames)} 帧")
        frames_data = last_episode_frames
    
    # 优化帧率设置，确保平滑播放
    fps = 24  # 保持24 FPS
    
    # 优化降采样策略，确保运动连续性同时减小视频体积
    downsampling_rate = calculate_downsampling_rate(frames_data)
    
    frames_data_downsampled = frames_data[::downsampling_rate]
    total_frames = len(frames_data_downsampled)
    print(f"降采样后帧数: {total_frames}")
    print(f"降采样率: 每{downsampling_rate}帧取一帧")
    
    # 计算预计视频时长
    estimated_duration = total_frames / fps
    print(f"预计视频时长: {estimated_duration:.1f}秒 ({estimated_duration/60:.1f}分钟)")
    
    # 预计算高度范围，确保高度变化显示更准确
    min_height, max_height, height_range = get_height_range(frames_data)
    print(f"高度范围: {min_height:.1f}m - {max_height:.1f}m")
    
    # 使用imageio.get_writer逐帧写入，避免内存问题
    try:
        # 调整视频编码参数，使用更高效的编码器
        writer = imageio.get_writer(save_path, fps=fps, quality=8, codec='libx264')  # 使用H.264编码器，平衡质量和大小
        
        # 预计算所有帧的坐标范围，确保所有帧使用相同的坐标轴范围
        x_min, x_max, y_min, y_max = get_axis_range(frames_data_downsampled, dimensions=2)
        
        for i, frame_data in enumerate(frames_data_downsampled):
            # 优化绘图设置，减小图像分辨率以减小视频体积
            fig = plt.figure(figsize=(8, 6), dpi=72)  # 减小图形尺寸和dpi
            ax = fig.add_subplot(111)
            
            # 绘制用户
            user_pos = np.array(frame_data['user_pos'])
            if user_pos.size > 0:
                ax.scatter(user_pos[:, 0], user_pos[:, 1], s=50, c='gray', label='用户')
            
            # 绘制无人机
            drone_pos = np.array(frame_data['drone_pos'])
            if drone_pos.size > 0:
                for j, d in enumerate(drone_pos):
                    # 优化高度归一化，使用实际高度范围
                    height_normalized = (d[2] - min_height) / height_range
                    height_normalized = max(0, min(1, height_normalized))  # 归一化到0-1
                    
                    # 使用更明显的颜色映射，从蓝色（低）到红色（高）
                    color = plt.cm.jet(height_normalized)  # 使用jet颜色映射，对比更明显
                    
                    # 优化无人机大小随高度变化，使变化更明显
                    base_size = 80
                    size_range = 120
                    marker_size = base_size + height_normalized * size_range  # 高度越高，标记越大
                    
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
                    # 为轨迹添加高度信息，使用颜色渐变
                    for k in range(len(traj) - 1):
                        # 计算两个点之间的高度
                        height1 = traj[k, 2]
                        height2 = traj[k+1, 2]
                        # 归一化高度
                        h1_norm = max(0, min(1, (height1 - min_height) / height_range))
                        h2_norm = max(0, min(1, (height2 - min_height) / height_range))
                        # 获取颜色
                        color1 = plt.cm.jet(h1_norm)
                        color2 = plt.cm.jet(h2_norm)
                        
                        # 绘制带有颜色渐变的线段
                        # 为了提高性能，我们直接使用起始点颜色
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
            
            # 调整布局，确保所有帧的大小一致
            plt.tight_layout()
            
            # 保存为图像，优化保存参数
            save_frame(fig, writer)
            
            # 每20帧打印一次进度
            if (i + 1) % 20 == 0 or (i + 1) == total_frames:
                print(f"已生成 {i + 1}/{total_frames} 帧")
        writer.close()
        print(f"轨迹动画已保存到 {save_path}")
        print("视频播放流畅度优化：")
        print(f"1. 帧率设置为 {fps} FPS")
        print(f"2. 智能降采样率：每{downsampling_rate}帧取一帧")
        print("3. 优化渲染速度和视频质量")
        print("4. 确保一致的帧生成速度")
        print("高度变化显示优化：")
        print("1. 使用实际数据的高度范围进行归一化")
        print("2. 使用jet颜色映射，从蓝色（低）到红色（高）")
        print("3. 增大无人机大小随高度的变化范围")
        print("4. 轨迹使用颜色渐变显示高度变化")
        print("5. 保持高度变化的平滑过渡")
    except Exception as e:
        print(f"生成轨迹动画失败: {e}")

def create_comparison_animation(maddpg_frames, isac_maddpg_frames, save_path='results/training_videos/comparison_animation.mp4'):
    """
    创建对比动画
    :param maddpg_frames: MADDPG的帧数据
    :param isac_maddpg_frames: ISAC-MADDPG的帧数据
    :param save_path: 保存路径
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 降采样
    maddpg_frames_downsampled = maddpg_frames[::5]
    isac_maddpg_frames_downsampled = isac_maddpg_frames[::5]
    total_frames = min(len(maddpg_frames_downsampled), len(isac_maddpg_frames_downsampled))
    print(f"降采样后帧数: {total_frames}")
    
    # 使用imageio.get_writer逐帧写入
    try:
        writer = imageio.get_writer(save_path, fps=10, quality=8, codec='libx264')
        
        for i in range(total_frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
            
            # 绘制MADDPG
            maddpg_frame = maddpg_frames_downsampled[i]
            user_pos = np.array(maddpg_frame['user_pos'])
            if user_pos.size > 0:
                ax1.scatter(user_pos[:, 0], user_pos[:, 1], s=50, c='gray', label='用户')
            
            drone_pos = np.array(maddpg_frame['drone_pos'])
            if drone_pos.size > 0:
                colors = ['blue', 'green', 'red', 'purple', 'orange']
                for j, d in enumerate(drone_pos):
                    ax1.scatter(d[0], d[1], s=100, c=colors[j % len(colors)], label=f'无人机{j+1}')
                    circle = plt.Circle((d[0], d[1]), 30, fill=False, color='blue', linestyle='--', alpha=0.5)
                    ax1.add_patch(circle)
            
            trajectories = maddpg_frame.get('trajectories', [])
            for j, traj in enumerate(trajectories):
                traj = np.array(traj)
                if traj.size > 0:
                    ax1.plot(traj[:, 0], traj[:, 1], linestyle='--', color=colors[j % len(colors)], label=f'轨迹{j+1}')
            
            ax1.set_title('MADDPG')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.legend()
            ax1.grid(True)
            ax1.set_aspect('equal')
            
            # 绘制ISAC-MADDPG
            isac_maddpg_frame = isac_maddpg_frames_downsampled[i]
            user_pos = np.array(isac_maddpg_frame['user_pos'])
            if user_pos.size > 0:
                ax2.scatter(user_pos[:, 0], user_pos[:, 1], s=50, c='gray', label='用户')
            
            drone_pos = np.array(isac_maddpg_frame['drone_pos'])
            if drone_pos.size > 0:
                colors = ['blue', 'green', 'red', 'purple', 'orange']
                for j, d in enumerate(drone_pos):
                    ax2.scatter(d[0], d[1], s=100, c=colors[j % len(colors)], label=f'无人机{j+1}')
                    circle = plt.Circle((d[0], d[1]), 30, fill=False, color='blue', linestyle='--', alpha=0.5)
                    ax2.add_patch(circle)
            
            trajectories = isac_maddpg_frame.get('trajectories', [])
            for j, traj in enumerate(trajectories):
                traj = np.array(traj)
                if traj.size > 0:
                    ax2.plot(traj[:, 0], traj[:, 1], linestyle='--', color=colors[j % len(colors)], label=f'轨迹{j+1}')
            
            ax2.set_title('ISAC-MADDPG')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_aspect('equal')
            
            plt.suptitle(f'算法对比动画 - 第{i+1}帧')
            plt.tight_layout()
            
            # 保存为图像
            save_frame(fig, writer)
            
            # 每100帧打印一次进度
            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                print(f"已生成 {i + 1}/{total_frames} 帧")
        
        writer.close()
        print(f"对比动画已保存到 {save_path}")
    except Exception as e:
        print(f"生成对比动画失败: {e}")