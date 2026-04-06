import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from visualization.animation.animation_2d import create_trajectory_animation, create_comparison_animation
from visualization.animation.animation_3d import create_3d_trajectory_animation, create_3d_comparison_animation

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
    
    # 生成3D轨迹动画
    print("\n生成3D轨迹动画...")
    create_3d_trajectory_animation(maddpg_frames)
    
    # 生成对比动画（使用相同数据作为示例）
    print("\n生成2D对比动画...")
    create_comparison_animation(maddpg_frames, maddpg_frames)
    
    # 生成3D对比动画（使用相同数据作为示例）
    print("\n生成3D对比动画...")
    create_3d_comparison_animation(maddpg_frames, maddpg_frames)

if __name__ == "__main__":
    main()