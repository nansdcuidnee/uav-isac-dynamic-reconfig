import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.visualization.utils.video_utils import (
    calculate_downsampling_rate,
    save_frame,
    get_axis_range,
    create_directory
)
from src.visualization.terrain_generator import generate_terrain_mesh, add_terrain_features

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_quadcopter(ax, pos, color='red', size=5):
    """
    绘制四旋翼无人机模型
    :param ax: 3D轴
    :param pos: 无人机位置 [x, y, z]
    :param color: 无人机颜色
    :param size: 无人机大小
    """
    x, y, z = pos
    
    # 绘制机身
    body_length = size * 0.8
    body_radius = size * 0.2
    body_z = np.linspace(z - body_length/2, z + body_length/2, 10)
    body_theta = np.linspace(0, 2*np.pi, 10)
    body_theta, body_z = np.meshgrid(body_theta, body_z)
    body_x = x + body_radius * np.cos(body_theta)
    body_y = y + body_radius * np.sin(body_theta)
    ax.plot_surface(body_x, body_y, body_z, color=color, alpha=0.8)
    
    # 绘制四个螺旋桨臂
    arm_length = size * 1.5
    arm_radius = size * 0.05
    
    # 前臂
    arm1_z = np.linspace(z - arm_length/2, z + arm_length/2, 10)
    arm1_x = np.linspace(x, x + arm_length, 10)
    arm1_x, arm1_z = np.meshgrid(arm1_x, arm1_z)
    arm1_y = y + arm_radius * np.cos(np.linspace(0, 2*np.pi, 10))[:, np.newaxis]
    ax.plot_surface(arm1_x, arm1_y, arm1_z, color=color, alpha=0.6)
    
    # 后臂
    arm2_z = np.linspace(z - arm_length/2, z + arm_length/2, 10)
    arm2_x = np.linspace(x, x - arm_length, 10)
    arm2_x, arm2_z = np.meshgrid(arm2_x, arm2_z)
    arm2_y = y + arm_radius * np.cos(np.linspace(0, 2*np.pi, 10))[:, np.newaxis]
    ax.plot_surface(arm2_x, arm2_y, arm2_z, color=color, alpha=0.6)
    
    # 右臂
    arm3_z = np.linspace(z - arm_length/2, z + arm_length/2, 10)
    arm3_y = np.linspace(y, y + arm_length, 10)
    arm3_y, arm3_z = np.meshgrid(arm3_y, arm3_z)
    arm3_x = x + arm_radius * np.cos(np.linspace(0, 2*np.pi, 10))[:, np.newaxis]
    ax.plot_surface(arm3_x, arm3_y, arm3_z, color=color, alpha=0.6)
    
    # 左臂
    arm4_z = np.linspace(z - arm_length/2, z + arm_length/2, 10)
    arm4_y = np.linspace(y, y - arm_length, 10)
    arm4_y, arm4_z = np.meshgrid(arm4_y, arm4_z)
    arm4_x = x + arm_radius * np.cos(np.linspace(0, 2*np.pi, 10))[:, np.newaxis]
    ax.plot_surface(arm4_x, arm4_y, arm4_z, color=color, alpha=0.6)
    
    # 绘制螺旋桨
    propeller_radius = size * 0.4
    propeller_thickness = size * 0.02
    
    # 前螺旋桨
    prop1_theta = np.linspace(0, 2*np.pi, 20)
    prop1_r = np.linspace(0, propeller_radius, 10)
    prop1_theta, prop1_r = np.meshgrid(prop1_theta, prop1_r)
    prop1_x = x + arm_length + prop1_r * np.cos(prop1_theta)
    prop1_y = y
    prop1_z = z + propeller_thickness * np.sin(np.linspace(0, np.pi, 10))[:, np.newaxis]
    ax.plot_surface(prop1_x, prop1_y, prop1_z, color='white', alpha=0.9)
    
    # 后螺旋桨
    prop2_theta = np.linspace(0, 2*np.pi, 20)
    prop2_r = np.linspace(0, propeller_radius, 10)
    prop2_theta, prop2_r = np.meshgrid(prop2_theta, prop2_r)
    prop2_x = x - arm_length + prop2_r * np.cos(prop2_theta)
    prop2_y = y
    prop2_z = z + propeller_thickness * np.sin(np.linspace(0, np.pi, 10))[:, np.newaxis]
    ax.plot_surface(prop2_x, prop2_y, prop2_z, color='white', alpha=0.9)
    
    # 右螺旋桨
    prop3_theta = np.linspace(0, 2*np.pi, 20)
    prop3_r = np.linspace(0, propeller_radius, 10)
    prop3_theta, prop3_r = np.meshgrid(prop3_theta, prop3_r)
    prop3_x = x
    prop3_y = y + arm_length + prop3_r * np.sin(prop3_theta)
    prop3_z = z + propeller_thickness * np.sin(np.linspace(0, np.pi, 10))[:, np.newaxis]
    ax.plot_surface(prop3_x, prop3_y, prop3_z, color='white', alpha=0.9)
    
    # 左螺旋桨
    prop4_theta = np.linspace(0, 2*np.pi, 20)
    prop4_r = np.linspace(0, propeller_radius, 10)
    prop4_theta, prop4_r = np.meshgrid(prop4_theta, prop4_r)
    prop4_x = x
    prop4_y = y - arm_length + prop4_r * np.sin(prop4_theta)
    prop4_z = z + propeller_thickness * np.sin(np.linspace(0, np.pi, 10))[:, np.newaxis]
    ax.plot_surface(prop4_x, prop4_y, prop4_z, color='white', alpha=0.9)

def create_3d_trajectory_animation(frames_data, save_path='results/training_videos/3d_trajectory_animation.mp4', max_steps=None, fps=24, downsample='auto'):
    """
    创建3D轨迹动画
    :param frames_data: 帧数据列表，每个元素是 {'drone_pos': [...], 'user_pos': [...], 'connections': [...], 'trajectories': [...], 'step': step}
    :param save_path: 保存路径
    :param max_steps: 每个回合的最大步数，如果提供，只保存最后一个回合的帧
    :param fps: 帧率
    :param downsample: 降采样率，'auto'表示自动计算，整数表示强制使用该值
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 只保存最后一个回合的帧（如果提供了max_steps）
    if max_steps and max_steps > 0 and len(frames_data) > max_steps:
        last_episode_frames = frames_data[-max_steps:]
        print(f"只保存最后一个回合的 {len(last_episode_frames)} 帧")
        frames_data = last_episode_frames
    
    # 优化降采样策略，确保运动连续性同时减小视频体积
    if downsample == 'auto':
        downsampling_rate = calculate_downsampling_rate(frames_data)
    else:
        downsampling_rate = downsample
    
    frames_data_downsampled = frames_data[::downsampling_rate]
    total_frames = len(frames_data_downsampled)
    print(f"降采样后帧数: {total_frames}")
    print(f"降采样率: 每{downsampling_rate}帧取一帧")
    
    # 计算预计视频时长
    estimated_duration = total_frames / fps
    print(f"预计视频时长: {estimated_duration:.1f}秒 ({estimated_duration/60:.1f}分钟)")
    
    # 预计算坐标范围，确保所有帧使用相同的坐标轴范围
    x_min, x_max, y_min, y_max, z_min, z_max = get_axis_range(frames_data, dimensions=3)
    print(f"坐标范围: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
    
    # 使用imageio.get_writer逐帧写入，避免内存问题
    try:
        # 调整视频编码参数，使用更高效的编码器
        writer = imageio.get_writer(save_path, fps=fps, quality=8, codec='libx264')  # 使用H.264编码器，平衡质量和大小
        
        # 生成地形网格
        terrain_x, terrain_y, terrain_z = generate_terrain_mesh([x_min, x_max], [y_min, y_max], resolution=50)
        
        for i, frame_data in enumerate(frames_data_downsampled):
            # 优化绘图设置，减小图像分辨率以减小视频体积
            fig = plt.figure(figsize=(10, 8), dpi=72)  # 减小图形尺寸和dpi
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制地形
            ax.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
            
            # 添加地形特征
            add_terrain_features(ax, terrain_x, terrain_y, terrain_z)
            
            # 绘制用户（确保用户在地形上）
            user_pos = np.array(frame_data['user_pos'])
            if user_pos.size > 0:
                # 调整用户位置，确保在地形上
                adjusted_user_pos = user_pos.copy()
                for j in range(len(adjusted_user_pos)):
                    # 找到对应的地形高度
                    xi = int((adjusted_user_pos[j, 0] - terrain_x.min()) / (terrain_x.max() - terrain_x.min()) * terrain_x.shape[1])
                    yi = int((adjusted_user_pos[j, 1] - terrain_y.min()) / (terrain_y.max() - terrain_y.min()) * terrain_x.shape[0])
                    if xi < terrain_x.shape[1] and yi < terrain_x.shape[0]:
                        adjusted_user_pos[j, 2] = terrain_z[yi, xi] + 1  # 高于地形1米
                ax.scatter(adjusted_user_pos[:, 0], adjusted_user_pos[:, 1], adjusted_user_pos[:, 2], s=50, c='gray', label='用户')
            
            # 绘制无人机
            drone_pos = np.array(frame_data['drone_pos'])
            if drone_pos.size > 0:
                colors = ['red', 'green', 'blue', 'purple', 'orange']
                for j, d in enumerate(drone_pos):
                    color = colors[j % len(colors)]
                    # 绘制四旋翼无人机模型
                    draw_quadcopter(ax, d, color=color, size=3)
                    
                    # 绘制通信范围（在3D中绘制球体，使用更美观的半透明效果）
                    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                    r = 50  # 通信范围半径
                    x = d[0] + r * np.cos(u) * np.sin(v)
                    y = d[1] + r * np.sin(u) * np.sin(v)
                    z = d[2] + r * np.cos(v)
                    ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')
                    
                    # 绘制感知范围（使用不同的颜色和透明度）
                    r_sense = 60  # 感知范围半径
                    x = d[0] + r_sense * np.cos(u) * np.sin(v)
                    y = d[1] + r_sense * np.sin(u) * np.sin(v)
                    z = d[2] + r_sense * np.cos(v)
                    ax.plot_surface(x, y, z, color='green', alpha=0.08, edgecolor='none')
            
            # 绘制轨迹
            trajectories = frame_data.get('trajectories', [])
            if trajectories:
                for j, traj in enumerate(trajectories):
                    traj = np.array(traj)
                    if traj.size > 0:
                        color = colors[j % len(colors)]
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='-', linewidth=3, color=color, label=f'轨迹{j+1}')
            
            # 绘制连接线
            connections = frame_data.get('connections')
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
                        ax.plot(xs, ys, zs, linewidth=1, color='green')
            
            # 获取当前步骤
            step = frame_data.get('step', i+1)
            
            ax.set_title(f'3D无人机轨迹动画 - 第{step}步')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.legend()
            ax.grid(True)
            
            # 设置固定的坐标轴范围
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            
            # 设置视角，确保每次都从相同的角度观察
            ax.view_init(elev=30, azim=45)
            
            # 调整布局，确保所有帧的大小一致
            plt.tight_layout()
            
            # 保存为图像，优化保存参数
            save_frame(fig, writer)
            
            # 每20帧打印一次进度
            if (i + 1) % 20 == 0 or (i + 1) == total_frames:
                print(f"已生成 {i + 1}/{total_frames} 帧")
        writer.close()
        print(f"3D轨迹动画已保存到 {save_path}")
        print("视频播放流畅度优化：")
        print(f"1. 帧率设置为 {fps} FPS")
        print(f"2. 智能降采样率：每{downsampling_rate}帧取一帧")
        print("3. 优化渲染速度和视频质量")
        print("4. 确保一致的帧生成速度")
    except Exception as e:
        print(f"生成3D轨迹动画失败: {e}")

def create_3d_comparison_animation(single_drone_frames, three_drones_frames, save_path='results/training_videos/3d_comparison_animation.mp4', fps=24, downsample='auto'):
    """
    创建3D对比动画，展示1个无人机和3个无人机的对比
    :param single_drone_frames: 1个无人机的帧数据
    :param three_drones_frames: 3个无人机的帧数据
    :param save_path: 保存路径
    :param fps: 帧率
    :param downsample: 降采样率，'auto'表示自动计算，整数表示强制使用该值
    """
    # 创建保存目录
    dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    create_directory(dir_path)
    
    # 降采样
    if downsample == 'auto':
        # 自动计算降采样率
        from src.visualization.utils.video_utils import calculate_downsampling_rate
        downsampling_rate = calculate_downsampling_rate(single_drone_frames)
    else:
        downsampling_rate = downsample
    
    single_drone_frames_downsampled = single_drone_frames[::downsampling_rate]
    three_drones_frames_downsampled = three_drones_frames[::downsampling_rate]
    total_frames = min(len(single_drone_frames_downsampled), len(three_drones_frames_downsampled))
    print(f"降采样后帧数: {total_frames}")
    print(f"降采样率: 每{downsampling_rate}帧取一帧")
    
    # 计算预计视频时长
    estimated_duration = total_frames / fps
    print(f"预计视频时长: {estimated_duration:.1f}秒 ({estimated_duration/60:.1f}分钟)")
    
    # 预计算坐标范围
    all_frames = single_drone_frames_downsampled + three_drones_frames_downsampled
    x_min, x_max, y_min, y_max, z_min, z_max = get_axis_range(all_frames, dimensions=3)
    
    # 生成对比动画
    try:
        writer = imageio.get_writer(save_path, fps=fps, quality=8, codec='libx264')
        
        # 生成地形网格
        terrain_x, terrain_y, terrain_z = generate_terrain_mesh([x_min, x_max], [y_min, y_max], resolution=50)
        
        for i in range(total_frames):
            fig = plt.figure(figsize=(20, 8), dpi=72)
            
            # 1个无人机场景
            ax1 = fig.add_subplot(121, projection='3d')
            single_frame = single_drone_frames_downsampled[i]
            
            # 绘制地形
            ax1.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
            
            # 添加地形特征
            add_terrain_features(ax1, terrain_x, terrain_y, terrain_z)
            
            # 绘制用户（确保用户在地形上）
            user_pos = np.array(single_frame['user_pos'])
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
            drone_pos = np.array(single_frame['drone_pos'])
            if drone_pos.size > 0:
                # 绘制四旋翼无人机模型
                draw_quadcopter(ax1, drone_pos[0], color='red', size=3)
                
                # 绘制通信范围
                u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                r = 50
                x = drone_pos[0, 0] + r * np.cos(u) * np.sin(v)
                y = drone_pos[0, 1] + r * np.sin(u) * np.sin(v)
                z = drone_pos[0, 2] + r * np.cos(v)
                ax1.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')
                
                # 绘制感知范围
                r_sense = 60
                x = drone_pos[0, 0] + r_sense * np.cos(u) * np.sin(v)
                y = drone_pos[0, 1] + r_sense * np.sin(u) * np.sin(v)
                z = drone_pos[0, 2] + r_sense * np.cos(v)
                ax1.plot_surface(x, y, z, color='green', alpha=0.08, edgecolor='none')
            
            # 绘制轨迹
            trajectories = single_frame.get('trajectories', [])
            if trajectories:
                traj = np.array(trajectories[0])
                if traj.size > 0:
                    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='-', linewidth=3, color='red', label='轨迹')
            
            ax1.set_title('1个无人机')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])
            ax1.set_zlim([z_min, z_max])
            ax1.view_init(elev=30, azim=45)
            
            # 3个无人机场景
            ax2 = fig.add_subplot(122, projection='3d')
            three_frame = three_drones_frames_downsampled[i]
            
            # 绘制地形
            ax2.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.8, edgecolor='none')
            
            # 添加地形特征
            add_terrain_features(ax2, terrain_x, terrain_y, terrain_z)
            
            # 绘制用户（确保用户在地形上）
            user_pos = np.array(three_frame['user_pos'])
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
            drone_pos = np.array(three_frame['drone_pos'])
            if drone_pos.size > 0:
                colors = ['red', 'green', 'blue']
                for j, d in enumerate(drone_pos):
                    color = colors[j % len(colors)]
                    # 绘制四旋翼无人机模型
                    draw_quadcopter(ax2, d, color=color, size=3)
                    
                    # 绘制通信范围
                    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                    r = 50
                    x = d[0] + r * np.cos(u) * np.sin(v)
                    y = d[1] + r * np.sin(u) * np.sin(v)
                    z = d[2] + r * np.cos(v)
                    ax2.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')
                    
                    # 绘制感知范围
                    r_sense = 60
                    x = d[0] + r_sense * np.cos(u) * np.sin(v)
                    y = d[1] + r_sense * np.sin(u) * np.sin(v)
                    z = d[2] + r_sense * np.cos(v)
                    ax2.plot_surface(x, y, z, color='green', alpha=0.08, edgecolor='none')
            
            # 绘制轨迹
            trajectories = three_frame.get('trajectories', [])
            if trajectories:
                for j, traj in enumerate(trajectories):
                    traj = np.array(traj)
                    if traj.size > 0:
                        color = colors[j % len(colors)]
                        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle='-', linewidth=3, color=color, label=f'轨迹{j+1}')
            
            ax2.set_title('3个无人机')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Z (m)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim([x_min, x_max])
            ax2.set_ylim([y_min, y_max])
            ax2.set_zlim([z_min, z_max])
            ax2.view_init(elev=30, azim=45)
            
            plt.suptitle(f'1个无人机 vs 3个无人机 - 第{i+1}帧')
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