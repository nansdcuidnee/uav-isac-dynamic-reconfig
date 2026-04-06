#!/bin/bash

# 可视化脚本

# 生成训练曲线
echo "生成训练曲线..."
python main.py --mode visualize --visualization_type "training_curves" --experiment_name "visualize_training"

# 生成轨迹图
echo "生成轨迹图..."
python main.py --mode visualize --visualization_type "trajectory" --model_path results/models/best_model.pt --experiment_name "visualize_trajectory"

# 生成动画
echo "生成动画..."
python main.py --mode visualize --visualization_type "animation" --model_path results/models/best_model.pt --experiment_name "visualize_animation"

# 生成覆盖率曲线
echo "生成覆盖率曲线..."
python main.py --mode visualize --visualization_type "coverage_curve" --experiment_name "visualize_coverage"

# 生成对比曲线
echo "生成对比曲线..."
python main.py --mode visualize --visualization_type "comparison" --experiment_name "visualize_comparison"

echo "所有可视化任务完成！"