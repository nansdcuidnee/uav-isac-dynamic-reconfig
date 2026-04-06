# UAV_ISAC_MADDPG

UAV_ISAC_MADDPG 是一个基于 PyTorch 和 Gym 的多智能体集成感知通信（ISAC）实验平台，包含 MAV 工况下的 MADDPG、DDPG、DQN 等策略，以及对训练过程、评估指标与可视化结果的统一编排。

## 快速开始

### 环境准备
- Python 3.10/3.11
- CUDA (可选)：用于加速训练和可视化
- 推荐创建虚拟环境，保持依赖隔离

### 安装依赖
```bash
pip install -r requirements.txt
```

### 主要入口
| 目标 | 命令示例 |
| --- | --- |
| 单机训练（1 架无人机） | `python main.py --mode train --num_drones 1 --algorithm ddpg` |
| 多机训练（3 架无人机） | `python main.py --mode train --num_drones 3 --algorithm maddpg` |
| 评估指定模型 | `python main.py --mode eval --model results/models/ddpg_1_drones/models/model_600.pt` |
| 多机训练（采集更多轨迹） | `python main.py --mode train --num_drones 3 --algorithm maddpg --trajectory_mode last --trajectory_episodes 50` |
| 生成3D动画 | `python src/visualization/animation/generate_animation_from_data.py --trajectory_json results/models/maddpg_3_drones/trajectory_data.json --user_json results/models/maddpg_3_drones/user_positions.json --output results/videos/demo_3d.mp4` |
| 生成更长3D视频 | `python src/visualization/animation/generate_animation_from_data.py --trajectory_json results/models/maddpg_3_drones/trajectory_data.json --user_json results/models/maddpg_3_drones/user_positions.json --mode 3d --fps 8 --downsample 1 --output results/videos/demo_3d_long.mp4` |

## 项目结构
- `main.py`：统一入口，负责解析命令行参数，调用训练/评估/动画子流程。
- `requirements.txt`：依赖列表。
- `docs/`：项目说明文档。
- `src/`：核心代码目录。
  - `configs/`：所有 YAML 配置，分为 `env_config.yaml`、`algo_config.yaml`、`train_config.yaml`、`config.yaml` 等。
  - `agents/`：各类策略及辅助组件。
    - `maddpg/`：MADDPG 实现（actor/critic/noise/replay）。
    - `ddpg/`：DDPG 实现，支持单机/多机设定。
    - `dqn/`：DQN 基线。
    - `random_agent.py`：随机基线。
  - `envs/`：环境编排和状态逻辑。
    - `uav_env.py`：主环境，负责 `reset`、`step`、状态/奖励/终止逻辑。
    - `dynamics/`：包含 `uav_dynamics.py`，提供飞行动力学与运动约束。
    - `models/`：通信、感知、用户、UAV、环境等模块化实体。
    - `channel_model.py` / `sensing_model.py`：保留的旧入口（实际依赖 models/ 中对应实现）。
  - `training/`：训练控制器、日志、评估器和检查点管理。
  - `evaluation/`：独立评估脚本，用于固定策略/模型的指标计算。
  - `visualization/`：动画、可视化工具，与 `animation/` 和 `utils/` 子目录协作。
  - `utils/`：常用工具，包含文件加载、环境辅助、数据与绘图辅助函数。
- `results/`：训练结果、模型和生成的图像/视频。
- `scripts/`、`data/` 等目录仅在特定流程下使用。

## 模块职责
1. **环境层（`src/envs/`）**：管理 UAV 系统的状态、导航约束、通信/感知/用户模型的集成。`uav_env.py` 负责统一的 `step`/`reset`，并通过 `models/` 提供更细粒度的实体模拟。`channel_model.py`、`sensing_model.py` 仍被训练脚本引用，但内部由 `src/envs/models/communication_model.py` 与 `src/envs/models/sensing_model.py` 完成实际逻辑。
2. **策略放置（`src/agents/`）**：每个策略模块提供 `Agent.choose_action()`、`learn()`、ReplayBuffer 以及网络定义。`maddpg` 为多智能体，`ddpg` 和 `dqn` 为单体/独立 agent 对照组。
3. **训练控制（`src/training/`）**：`train.py` 封装训练循环，`trainer.py` 组织单 epoch 的采样与更新，`logger.py` / `checkpoint.py` 负责日志与模型周期保存，`evaluator.py` 嵌入训练过程中的评估逻辑。
4. **评估/可视化（`src/evaluation/` + `src/visualization/`）**：评估脚本以固定模型运行，输出覆盖率、平均奖励等。可视化目录提供 2D/3D 动画生成脚本以及视频工具。
5. **辅助（`src/utils/`）**：包含配置加载、环境参数、绘图与数据处理，供各模块复用。

## 常见流程
1. 运行 `python main.py --mode train`，根据 `--num_drones` 读取对应配置，创建 `UAVEnv`，加载 `src/agents/` 指定策略，进入训练循环。
2. 训练中由 `src/training/trainer.py` 与 `src/training/evaluator.py` 协同完成多 agent 的动作采样、loss 计算与评估。
3. 完成模型后，通过 `python main.py --mode eval` 或 `python src/evaluation/evaluate.py` 计算各类指标。
4. 可视化脚本（`src/visualization/animation/*.py`）可根据实验输出生成 2D/3D 动画并保存至 `results/`。

## 配置说明
| 配置文件 | 说明 |
| --- | --- |
| `src/configs/config.yaml` | 入口配置，引用环境/算法/训练子配置。|
| `src/configs/env_config.yaml` | 环境维度、奖励结构、场景参数。|
| `src/configs/algo_config.yaml` | 算法选择（MADDPG/DDPG/DQN）及超参。|
| `src/configs/train_config.yaml` | 训练批次、保存频率、日志。|
| `src/configs/default.yaml` | 默认值，避免缺省字段抛出。|

## 结果与日志
- `results/models/`：保存阶段性模型权重、轨迹数据和用户位置数据。
- `results/plots/`：训练/评估曲线。
- `results/videos/`：可视化动画输出。
- `src/training/logger.py` 将信息打印到控制台并写入日志文件，辅助 `results/` 目录分析。

## 参考资料
- [MADDPG: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- [ISAC: Integrated Sensing and Communication](https://arxiv.org/abs/2001.07405)
