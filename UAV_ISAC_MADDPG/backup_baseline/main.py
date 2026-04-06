import os
import sys
import argparse

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def main():
    """
    项目主入口
    """
    parser = argparse.ArgumentParser(description="UAV MADDPG 轨迹规划项目")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'animate'], help='运行模式')
    parser.add_argument('--config', type=str, default='src/configs/config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='results/models/ddpg_1_drones/models/model_600.pt', help='模型路径')
    parser.add_argument('--num_drones', type=int, default=3, help='无人机数量')
    parser.add_argument('--algorithm', type=str, default='maddpg', choices=['ddpg', 'dqn', 'maddpg'], help='算法名称')
    parser.add_argument('--trajectory_mode', type=str, default='last', choices=['first', 'last'], help='轨迹采集模式')
    parser.add_argument('--trajectory_episodes', type=int, default=10, help='轨迹采集的 episodes 数量')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        try:
            from src.training.train import train
        except ImportError:
            from train.train import train
        save_dir = f"results/models/{args.algorithm}_{args.num_drones}_drones"
        train(args.config, save_dir, algorithm=args.algorithm, num_drones=args.num_drones, trajectory_mode=args.trajectory_mode, trajectory_episodes=args.trajectory_episodes)
    
    elif args.mode == 'eval':
        # 评估模式
        try:
            from src.evaluation.evaluate import evaluate
        except ImportError:
            from evaluation.evaluate import evaluate
        evaluate(args.config, args.model)
    
    elif args.mode == 'animate':
        # 动画模式
        try:
            from src.visualization.animate_3d_uav import create_3d_trajectory_animation
        except ImportError:
            from visualization.animate_3d_uav import create_3d_trajectory_animation
        # 这里需要实现动画生成逻辑
        print("动画生成功能正在开发中...")

if __name__ == "__main__":
    main()
