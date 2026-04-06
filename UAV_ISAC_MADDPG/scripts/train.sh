#!/bin/bash

# 训练脚本

# 普通MADDPG训练
echo "开始训练普通MADDPG..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --experiment_name "maddpg"

# ISAC-MADDPG训练
echo "开始训练ISAC-MADDPG..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --experiment_name "isac_maddpg"

# 不同无人机数量实验
echo "开始训练1个无人机..."
python main.py --mode train --num_drones 1 --num_users 20 --num_episodes 1000 --experiment_name "1_drone"

echo "开始训练3个无人机..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --experiment_name "3_drones"

echo "开始训练5个无人机..."
python main.py --mode train --num_drones 5 --num_users 20 --num_episodes 1000 --experiment_name "5_drones"

# 不同用户数量实验
echo "开始训练10个用户..."
python main.py --mode train --num_drones 3 --num_users 10 --num_episodes 1000 --experiment_name "10_users"

echo "开始训练20个用户..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --experiment_name "20_users"

echo "开始训练50个用户..."
python main.py --mode train --num_drones 3 --num_users 50 --num_episodes 1000 --experiment_name "50_users"

# 不同随机种子实验
echo "开始训练种子1..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --seed 1 --experiment_name "seed_1"

echo "开始训练种子42..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --seed 42 --experiment_name "seed_42"

echo "开始训练种子123..."
python main.py --mode train --num_drones 3 --num_users 20 --num_episodes 1000 --seed 123 --experiment_name "seed_123"

echo "所有训练任务完成！"