import numpy as np
from visualization.animate_uav import create_trajectory_animation
import os

# 生成大量的帧数据
def generate_large_test_frames():
    frames_data = []
    num_frames = 6400  # 生成6400帧，模拟原始数据
    num_drones = 1
    num_users = 10
    
    # 生成用户位置（固定）
    user_pos = np.random.uniform(0, 100, (num_users, 2))
    
    # 生成无人机轨迹
    for i in range(num_frames):
        # 无人机位置：圆形轨迹，高度随时间变化
        angle = i * 0.01
        x = 50 + 30 * np.cos(angle)
        y = 50 + 30 * np.sin(angle)
        # 高度随时间正弦变化，范围40-60米
        height = 50 + 10 * np.sin(i * 0.005)
        
        drone_pos = np.array([[x, y, height]])
        
        # 生成轨迹历史
        trajectories = []
        for j in range(num_drones):
            traj = []
            for k in range(max(0, i-20), i+1):
                traj_angle = k * 0.01
                traj_x = 50 + 30 * np.cos(traj_angle)
                traj_y = 50 + 30 * np.sin(traj_angle)
                traj_height = 50 + 10 * np.sin(k * 0.005)
                traj.append([traj_x, traj_y, traj_height])
            trajectories.append(traj)
        
        # 生成连接（简单的最近邻）
        connections = []
        for user in user_pos:
            # 找到最近的无人机
            distances = np.linalg.norm(drone_pos[:, :2] - user, axis=1)
            nearest_drone = np.argmin(distances)
            connections.append(nearest_drone)
        
        frame_data = {
            'drone_pos': drone_pos.tolist(),
            'user_pos': user_pos.tolist(),
            'connections': connections,
            'trajectories': trajectories
        }
        frames_data.append(frame_data)
    
    return frames_data

if __name__ == "__main__":
    # 生成测试数据
    print("生成测试数据...")
    frames_data = generate_large_test_frames()
    print(f"生成了 {len(frames_data)} 帧数据")
    
    # 确保保存目录存在
    save_path = 'results/exp3_num_uav_1_optimized/videos/trajectory_animation_large.mp4'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 生成动画
    print("生成轨迹动画...")
    create_trajectory_animation(frames_data, save_path, max_steps=1000)  # 只保存最后1000帧
    print("动画生成完成！")
