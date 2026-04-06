# 项目架构

## 总览
`main.py` 是统一入口，依据 CLI 参数选择训练/评估/动画路径，依赖 `src/configs/config.yaml` 将环境、算法、训练配置拼接后交给核心模块。项目以 Gym 风格的 `UAVEnv` 为基础，策略模块通过 `step`/`reset` 与环境交互，训练循环收集经验并驱动 agent 更新。

## 关键目录与边界
- `main.py`：启动流程、解析参数、负责日志目录和模型保存路径。
- `src/configs/`：管理 YAML 配置；入口 `config.yaml` 会引用 `env_config.yaml`、`algo_config.yaml`、`train_config.yaml` 并补齐 `default.yaml` 的默认值。
- `src/envs/`：环境编排核心，包含 `uav_env.py`、`dynamics/uav_dynamics.py`、`models/*` 以及保留的旧接口 `channel_model.py`、`sensing_model.py`、`user_model.py`。
- `src/agents/`：放置 MADDPG、DDPG、DQN 及随机策略，提供 Actor/Critic/Replay/Noise 等通用组件。
- `src/training/`：封装训练管理（`train.py`）、Trainer/Evaluator/Logger/Checkpoint 组件，协调数据采集和模型更新。
- `src/evaluation/`：独立评估脚本，依赖训练结果计算覆盖率、能耗、奖励等指标。
- `src/visualization/`：生成 2D/3D 动画（`animation/`）、绘图辅助（`utils/`）并驱动 `results/` 的图像与视频输出。
- `src/utils/`：配置加载、环境参数、数据与绘图工具，供各层复用。

## 模块职责
### `src/envs/`
- `uav_env.py` 负责 `reset`、`step`、奖励、终止判断、动态调整；与 `src/envs/models/` 统一调用通信、感知、UAV、用户、环境等实体。
- `dynamics/uav_dynamics.py` 封装飞行器运动约束与动力学函数。
- `models/` 中的 `communication_model.py`、`sensing_model.py`、`uav_model.py`、`environment_model.py`、`user_model.py`、`demo_system.py` 提供可插拔实体和多维观测。
- `channel_model.py` / `sensing_model.py` 目前作为兼容 shim，调用 models 下的新实现；`user_model.py` 仍保留旧接口以便逐步迁移。

### `src/agents/`
- `maddpg/`：多智能体策略，actor/critic、replay buffer、noise、固定版本等都在此模块管理。
- `ddpg/` 与 `dqn/`：提供单 agent 或独立 agent 的对照实验。
- `random_agent.py` 用于基线实验。

### `src/training/`
- `train.py` 构建训练循环、采样与评估的调度。
- `trainer.py` 负责按 step/episode 采集交互数据并触发 agent 的 `learn()`。
- `logger.py`、`checkpoint.py`、`evaluator.py` 分别处理日志记录、模型保存与周期性评估。
- `training/evaluator.py` 也可以在训练中独立调用，产生日志供 `results/plots/` 使用。

### `src/evaluation/`
- `evaluate.py` 运行固定模型并输出单次或多次覆盖率、奖励等指标，方便和训练曲线对齐。

### `src/visualization/`
- `base_visualizer.py` 与 `system_visualizer.py` 负责绘图抽象；`terrain_generator.py` 生成地形数据。
- `animation/` 下的 `generate_animation.py`、`generate_animation_from_data.py`、`generate_isac_comparison.py` 生成 2D/3D、ISAC 对比视频。
- `utils/video_utils.py` 提供视频编码与帧操作工具。

### `src/utils/`
- `file_utils.py` 提供配置加载与路径工具。
- `env_utils.py` 包含环境参数、奖励函数、动作/观察空间辅助代码。
- `data_utils.py` 与 `plot_utils.py` 支撑训练曲线与统计图。

### `src/configs/`
- `config.yaml` 结合 `env_config.yaml`、`algo_config.yaml`、`train_config.yaml`，通过 `default.yaml` 补充缺省字段。
- 训练/评估流程从 `config` 中读取 `env`、`algo`、`train`、`eval` 四套配置，完成 CLI 传参后的最终构建。

## 数据与控制流
1. CLI 解析器在 `main.py` 读取 `--mode`、`--num_drones`、`--algo` 等参数并决定调用 `src/training/train.py`、`src/evaluation/evaluate.py` 或 `src/visualization/animation/*`。
2. 训练时依次创建 `UAVEnv`、agent（MADDPG/DDPG/DQN），以 Trainer 采样 `step`，把采样的 transition 送入 Agent 的 replay buffer。
3. Agent 在 `trainer.learn()` 中执行网络前向与优化，`logger`/`checkpoint` 自动记录信息并在设定频率保存模型。
4. `evaluator` 与 `evaluate.py` 复用训练出的模型路径进行覆盖率、平均 reward 等指标评估，结果写入 `results/plots/`。
5. 所有模块统一通过 `src/utils/file_utils.py` 中的 `load_config` 函数加载配置，确保配置加载流程的一致性。

## 可视化与结果目录
- `results/models/`：保存周期性与最终模型权重。
- `results/plots/`：训练曲线、reward 曲线等图形。
- `results/videos/`：由 `src/visualization/animation` 输出的 2D/3D 以及 ISAC 对比动画。
- `src/training/logger.py` 生成的日志可用于复现训练现场。

## 兼容与已废弃接口
- `src/envs/channel_model.py`、`src/envs/sensing_model.py` 暂时保留为 shim，内部直接调用 `src/envs/models/communication_model.py` 与 `src/envs/models/sensing_model.py`，以免训练脚本在迁移过程中失败。
- `src/envs/user_model.py` 依旧暴露旧接口，后续将逐步整合到 `src/envs/models/user_model.py`。

## 依赖与配置追踪
- `requirements.txt` 明确列出 `torch`、`gym`、`numpy` 等库。
- 训练、评估和动画流程均基于 `src/configs` 中定义的 YAML 模板，可通过 CLI `--config` 参数覆盖。
- 所有路径都以项目根 `UAV_ISAC_MADDPG/` 为参考，CI/调试脚本应先创建 `results/`、`src/visualization/` 等目录后再写入。
