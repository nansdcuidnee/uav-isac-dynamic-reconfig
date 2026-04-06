import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

from src.envs.models.environment_model import EnvironmentModel
from src.envs.models.uav_model import UAVModel
from src.envs.models.user_model import UserModel
from src.envs.models.communication_model import CommunicationModel
from src.envs.models.sensing_model import SensingModel
from src.visualization.system_visualizer import SystemVisualizer

def main():
    """
    系统模型演示
    """
    print("=" * 60)
    print("系统建模与可视化演示")
    print("=" * 60)
    
    # 1. 创建环境模型
    print("\n1. 创建环境模型...")
    environment = EnvironmentModel(area_size=200.0, time_steps=1000)
    print(f"   环境区域大小: {environment.get_area_size()}m × {environment.get_area_size()}m")
    print(f"   总时间步数: {environment.get_time_steps()}")
    
    # 2. 创建用户模型
    print("\n2. 创建用户模型...")
    user_positions = [
        [50, 50, 0],
        [80, 70, 0],
        [110, 90, 0],
        [140, 110, 0],
        [170, 130, 0]
    ]
    user_models = []
    for i, pos in enumerate(user_positions):
        user = UserModel(user_id=i, position=pos)
        user_models.append(user)
    print(f"   创建了 {len(user_models)} 个用户")
    
    # 3. 创建UAV模型
    print("\n3. 创建UAV模型...")
    initial_uav_positions = [
        [150, 100, 50],
        [170, 100, 60],
        [190, 100, 70]
    ]
    uav_models = []
    for i, pos in enumerate(initial_uav_positions):
        uav = UAVModel(
            uav_id=i,
            initial_position=pos,
            max_speed=10.0,
            communication_range=100.0,
            sensing_range=120.0
        )
        uav_models.append(uav)
    print(f"   创建了 {len(uav_models)} 架无人机")
    
    # 4. 模拟UAV移动
    print("\n4. 模拟UAV移动...")
    total_steps = 100
    for step in range(total_steps):
        for i, uav in enumerate(uav_models):
            current_pos = uav.get_position()
            # 简单的移动逻辑
            new_x = current_pos[0] - 0.5
            new_y = current_pos[1] + 0.3
            new_z = current_pos[2]
            new_position = environment.clip_position([new_x, new_y, new_z])
            uav.update_position(new_position)
        
        if (step + 1) % 20 == 0:
            print(f"   已完成 {step + 1}/{total_steps} 步移动")
    
    # 5. 创建可视化器
    print("\n5. 创建可视化...")
    visualizer = SystemVisualizer()
    
    # 6. 生成可视化
    print("\n6. 生成可视化结果...")
    
    # UAV轨迹可视化
    visualizer.visualize_uav_trajectory(uav_models)
    
    # 用户分布可视化
    visualizer.visualize_user_distribution(user_models)
    
    # 通信覆盖可视化
    visualizer.visualize_communication_coverage(uav_models, user_models)
    
    # 感知覆盖可视化
    visualizer.visualize_sensing_coverage(uav_models, user_models)
    
    # 动态动画 - 这里需要创建帧数据
    # 简化版本：只传递无人机模型，让可视化器处理
    visualizer.create_dynamic_animation([{'drone_pos': [uav.get_position() for uav in uav_models], 'user_pos': [user.get_position() for user in user_models]}])
    
    print("\n" + "=" * 60)
    print("系统建模与可视化演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
