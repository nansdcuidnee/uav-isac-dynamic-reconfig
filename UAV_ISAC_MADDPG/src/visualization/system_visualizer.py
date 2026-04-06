import numpy as np
import matplotlib.pyplot as plt
from src.visualization.base_visualizer import BaseVisualizer
from src.visualization.animation.animation_2d import create_trajectory_animation, create_comparison_animation
from src.visualization.animation.animation_3d import create_3d_trajectory_animation, create_3d_comparison_animation

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SystemVisualizer(BaseVisualizer):
    """
    系统可视化器
    """
    def __init__(self):
        """
        初始化系统可视化器
        """
        super().__init__()
    
    def visualize_uav_trajectory(self, uav_list, save_path='results/visualization/uav_trajectory.png'):
        """
        可视化无人机轨迹
        :param uav_list: 无人机列表
        :param save_path: 保存路径
        """
        fig = self.create_figure()
        ax = fig.add_subplot(111)
        
        # 绘制每个无人机的轨迹
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, uav in enumerate(uav_list):
            trajectory = uav.get_trajectory()
            if trajectory.size > 0:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                        linestyle='-', linewidth=2, 
                        color=colors[i % len(colors)], 
                        label=f'无人机{i+1}轨迹')
                # 绘制起点和终点
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                           s=100, c=colors[i % len(colors)], 
                           marker='o', label=f'无人机{i+1}起点')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                           s=100, c=colors[i % len(colors)], 
                           marker='*', label=f'无人机{i+1}终点')
        
        ax.set_title('无人机轨迹')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        self.save_figure(fig, save_path)
        self.close_figure()
    
    def visualize_user_distribution(self, user_list, save_path='results/visualization/user_distribution.png'):
        """
        可视化用户分布
        :param user_list: 用户列表
        :param save_path: 保存路径
        """
        fig = self.create_figure()
        ax = fig.add_subplot(111)
        
        # 绘制用户位置
        user_positions = np.array([user.get_position() for user in user_list])
        if user_positions.size > 0:
            ax.scatter(user_positions[:, 0], user_positions[:, 1], 
                       s=50, c='gray', marker='x', label='用户')
        
        ax.set_title('用户分布')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        self.save_figure(fig, save_path)
        self.close_figure()
    
    def visualize_communication_coverage(self, uav_list, user_list, save_path='results/visualization/communication_coverage.png'):
        """
        可视化通信覆盖
        :param uav_list: 无人机列表
        :param user_list: 用户列表
        :param save_path: 保存路径
        """
        fig = self.create_figure()
        ax = fig.add_subplot(111)
        
        # 绘制用户位置
        user_positions = np.array([user.get_position() for user in user_list])
        if user_positions.size > 0:
            ax.scatter(user_positions[:, 0], user_positions[:, 1], 
                       s=50, c='gray', marker='x', label='用户')
        
        # 绘制无人机和通信范围
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, uav in enumerate(uav_list):
            pos = uav.get_position()
            comm_range = uav.get_communication_range()
            
            # 绘制无人机
            ax.scatter(pos[0], pos[1], 
                       s=100, c=colors[i % len(colors)], 
                       label=f'无人机{i+1}')
            
            # 绘制通信范围
            circle = plt.Circle((pos[0], pos[1]), comm_range, 
                               fill=False, color=colors[i % len(colors)], 
                               linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_title('通信覆盖')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        self.save_figure(fig, save_path)
        self.close_figure()
    
    def visualize_sensing_coverage(self, uav_list, user_list, save_path='results/visualization/sensing_coverage.png'):
        """
        可视化感知覆盖
        :param uav_list: 无人机列表
        :param user_list: 用户列表
        :param save_path: 保存路径
        """
        fig = self.create_figure()
        ax = fig.add_subplot(111)
        
        # 绘制用户位置
        user_positions = np.array([user.get_position() for user in user_list])
        if user_positions.size > 0:
            ax.scatter(user_positions[:, 0], user_positions[:, 1], 
                       s=50, c='gray', marker='x', label='用户')
        
        # 绘制无人机和感知范围
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, uav in enumerate(uav_list):
            pos = uav.get_position()
            sense_range = uav.get_sensing_range()
            
            # 绘制无人机
            ax.scatter(pos[0], pos[1], 
                       s=100, c=colors[i % len(colors)], 
                       label=f'无人机{i+1}')
            
            # 绘制感知范围
            circle = plt.Circle((pos[0], pos[1]), sense_range, 
                               fill=False, color=colors[i % len(colors)], 
                               linestyle='--', alpha=0.3)
            ax.add_patch(circle)
        
        ax.set_title('感知覆盖')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        self.save_figure(fig, save_path)
        self.close_figure()
    
    def create_dynamic_animation(self, frames_data, save_path='results/videos/dynamic_animation.mp4', use_3d=False):
        """
        创建动态动画
        :param frames_data: 帧数据列表
        :param save_path: 保存路径
        :param use_3d: 是否使用3D动画
        """
        if use_3d:
            create_3d_trajectory_animation(frames_data, save_path)
        else:
            create_trajectory_animation(frames_data, save_path)
    
    def create_comparison_animation(self, maddpg_frames, isac_maddpg_frames, save_path='results/videos/comparison_animation.mp4', use_3d=False):
        """
        创建对比动画
        :param maddpg_frames: MADDPG的帧数据
        :param isac_maddpg_frames: ISAC-MADDPG的帧数据
        :param save_path: 保存路径
        :param use_3d: 是否使用3D动画
        """
        if use_3d:
            create_3d_comparison_animation(maddpg_frames, isac_maddpg_frames, save_path)
        else:
            create_comparison_animation(maddpg_frames, isac_maddpg_frames, save_path)