import numpy as np
from visualization.animate_3d_uav import create_3d_trajectory_animation
import os

# 生成测试帧数据
def generate_test_frames():
    frames_data = []
    num_frames = 50  # 生成50帧
    num_drones = 3
    num_users = 20
    
    # 生成用户位置（固定）
    user_pos = np.random.uniform(0, 200, (num_users, 3))
    # 设置用户高度为0
    user_pos[:, 2] = 0
    
    # 生成无人机轨迹
    for i in range(num_frames):
        # 生成多个无人机的位置
        drone_pos = []
        for j in range(num_drones):
            # 每个无人机使用不同的初始角度
            angle = i * 0.05 + j * (2 * np.pi / num_drones)
            x = 100 + 50 * np.cos(angle)
            y = 100 + 50 * np.sin(angle)
            # 高度随时间正弦变化，范围50-100米
            height = 75 + 25 * np.sin(i * 0.02 + j * 0.1)
            drone_pos.append([x, y, height])
        
        drone_pos = np.array(drone_pos)
        
        # 生成轨迹历史
        trajectories = []
        for j in range(num_drones):
            traj = []
            for k in range(max(0, i-20), i+1):
                traj_angle = k * 0.05 + j * (2 * np.pi / num_drones)
                traj_x = 100 + 50 * np.cos(traj_angle)
                traj_y = 100 + 50 * np.sin(traj_angle)
                traj_height = 75 + 25 * np.sin(k * 0.02 + j * 0.1)
                traj.append([traj_x, traj_y, traj_height])
            trajectories.append(traj)
        
        # 生成连接（简单的最近邻）
        connections = []
        for user in user_pos:
            # 找到最近的无人机
            distances = np.linalg.norm(drone_pos[:, :2] - user[:2], axis=1)
            nearest_drone = np.argmin(distances)
            connections.append(nearest_drone)
        
        frame_data = {
            'drone_pos': drone_pos.tolist(),
            'user_pos': user_pos.tolist(),
            'connections': connections,
            'trajectories': trajectories,
            'step': i+1
        }
        frames_data.append(frame_data)
    
    return frames_data

if __name__ == "__main__":
    # 生成测试数据
    print("生成测试数据...")
    frames_data = generate_test_frames()
    print(f"生成了 {len(frames_data)} 帧数据")
    
    # 确保保存目录存在
    save_path = 'results/training_videos/3d_terrain_animation.mp4'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 生成3D动画
    print("生成3D地形动画...")
    create_3d_trajectory_animation(frames_data, save_path)
    print("3D地形动画生成完成！")
    print(f"动画已保存到: {save_path}")
