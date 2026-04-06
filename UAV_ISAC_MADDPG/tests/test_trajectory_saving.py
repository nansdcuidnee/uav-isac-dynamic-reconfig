import os
import sys
import json
import math

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 使用一个简单的配置文件进行测试
    config_path = "configs/config.yaml"
    save_dir = "results/test_trajectory"
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始测试轨迹数据保存...")
    
    # 运行训练（使用较少的 episodes 和 steps 进行测试）
    # 注意：这里需要修改 config.yaml 文件，将训练参数设置为较小的值
    # 或者直接在代码中修改训练参数
    
    try:
        # 这里我们不实际运行训练，而是模拟轨迹数据的生成
        # 因为实际训练可能需要较长时间
        
        # 模拟轨迹数据
        trajectory_data = []
        user_positions = []
        
        # 生成模拟的轨迹数据
        for episode in range(2):
            for step in range(10):
                for uav_id in range(3):
                    # 生成简单的圆形轨迹
                    angle = step * 0.1 + episode * 2 * math.pi
                    radius = 50 + uav_id * 20
                    x = 100 + radius * math.cos(angle)
                    y = 100 + radius * math.sin(angle)
                    z = 50 + uav_id * 10
                    
                    trajectory_data.append({
                        'episode': episode,
                        'step': step,
                        'uav_id': uav_id,
                        'x': x,
                        'y': y,
                        'z': z
                    })
        
        # 生成模拟的用户位置
        for user_id in range(5):
            user_positions.append({
                'user_id': user_id,
                'x': 50 + user_id * 30,
                'y': 50 + user_id * 20,
                'z': 0.0
            })
        
        # 保存轨迹数据
        trajectory_path = os.path.join(save_dir, "trajectory_data.json")
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        print(f"模拟轨迹数据已保存到 {trajectory_path}")
        
        # 保存用户位置数据
        user_positions_path = os.path.join(save_dir, "user_positions.json")
        with open(user_positions_path, 'w', encoding='utf-8') as f:
            json.dump(user_positions, f, indent=2, ensure_ascii=False)
        print(f"模拟用户位置数据已保存到 {user_positions_path}")
        
        print("轨迹数据保存测试完成！")
        print("请将生成的数据文件复制到 Unity 项目的 StreamingAssets 文件夹中进行测试。")
        
    except Exception as e:
        print(f"测试失败: {e}")
