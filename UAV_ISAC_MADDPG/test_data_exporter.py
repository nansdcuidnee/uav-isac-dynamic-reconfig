import os
import sys
import numpy as np

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.uav_env import UAVEnv
from data_exporter import export_simulation_data

def generate_test_trajectory_data(num_drones=3, num_steps=100):
    """
    生成测试轨迹数据
    :param num_drones: 无人机数量
    :param num_steps: 步数
    :return: 轨迹数据
    """
    trajectory_data = []
    for episode in range(1):
        for step in range(num_steps):
            for uav_id in range(num_drones):
                # 生成简单的轨迹数据
                x = 100 + uav_id * 50 + step * 0.5
                y = 10 + uav_id * 10
                z = 200 + step * 0.1
                trajectory_data.append({
                    'episode': episode,
                    'step': step,
                    'uav_id': uav_id,
                    'x': x,
                    'y': y,
                    'z': z
                })
    return trajectory_data

def main():
    """
    测试数据导出
    """
    print("开始生成测试数据...")
    
    # 创建环境
    env = UAVEnv(
        num_drones=3,
        num_users=10,
        max_steps=100
    )
    
    # 生成测试轨迹数据
    trajectory_data = generate_test_trajectory_data(num_drones=3, num_steps=100)
    
    # 导出数据
    export_simulation_data(env, trajectory_data, export_dir='unity_data')
    
    print("测试数据导出完成！")

if __name__ == "__main__":
    main()
