#!/bin/bash

# 测试脚本

# 测试普通MADDPG模型
echo "开始测试普通MADDPG模型..."
python main.py --mode test --model_path results/models/maddpg_best_model.pt --experiment_name "test_maddpg"

# 测试ISAC-MADDPG模型
echo "开始测试ISAC-MADDPG模型..."
python main.py --mode test --model_path results/models/isac_maddpg_best_model.pt --experiment_name "test_isac_maddpg"

# 测试不同无人机数量模型
echo "开始测试1个无人机模型..."
python main.py --mode test --model_path results/models/1_drone_best_model.pt --num_drones 1 --experiment_name "test_1_drone"

echo "开始测试3个无人机模型..."
python main.py --mode test --model_path results/models/3_drones_best_model.pt --num_drones 3 --experiment_name "test_3_drones"

echo "开始测试5个无人机模型..."
python main.py --mode test --model_path results/models/5_drones_best_model.pt --num_drones 5 --experiment_name "test_5_drones"

echo "所有测试任务完成！"