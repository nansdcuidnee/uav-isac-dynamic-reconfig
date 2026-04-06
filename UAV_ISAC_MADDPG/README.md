# UAV_ISAC_MADDPG

UAV_ISAC_MADDPG is a multi-agent ISAC research project built with PyTorch and Gym-style environments.

## Week-1 Dynamic Member Scope

This branch includes the week-1 minimum dynamic member mechanism:

- scripted `leave` / `join` events from config
- active member masking (`active_mask`)
- inactive UAV exclusion from action and metrics
- reproducible demo artifacts (log + timeline + mask plot)

Current validation scenario is `3 -> 2 -> 4` via scripted events.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Main training entry:

```bash
python main.py --mode train --num_drones 3 --algorithm maddpg
```

Run dynamic demo:

```bash
python scripts/run_patent_dynamic.py
```

Backward-compatible wrapper:

```bash
python validate_dynamic_events.py
```

## Structure

- `main.py`: unified CLI entry for train/eval/animation.
- `src/configs/`: YAML configs (`config.yaml`, `env_config.yaml`, `algo_config.yaml`, `train_config.yaml`, `default.yaml`, `patent_dynamic.yaml`).
- `src/envs/`: environment core (`uav_env.py`, dynamics, models, legacy shim modules).
- `src/reconfiguration/`: dynamic reconfiguration modules.
  - `event_manager.py`: step-triggered scripted event resolution.
  - `member_manager.py`: `active_mask` state and activate/deactivate API.
- `src/agents/`: MADDPG/DDPG/DQN and related components.
- `src/training/`: training loop, trainer, logger, checkpoint, evaluator.
- `src/evaluation/`: standalone evaluation scripts.
- `src/visualization/`: animation and plotting utilities.
  - `plot_event_timeline.py`
  - `plot_active_mask.py`
- `scripts/`: operational scripts.
  - `run_patent_dynamic.py`: one-command dynamic scenario run.
  - `scripts/export/test_data_exporter.py`: exporter demo.
- `tests/`: minimal test coverage for dynamic mechanism.

## Dynamic Flow

```text
src/configs/patent_dynamic.yaml
  -> src/envs/uav_env.py
    -> src/reconfiguration/event_manager.py
    -> src/reconfiguration/member_manager.py
  -> scripts/run_patent_dynamic.py
    -> logs/dynamic_env_demo.log
    -> logs/event_timeline.txt
    -> logs/event_timeline.png
    -> logs/active_mask_change.png
```

## Implemented vs Pending

Implemented now:

- scripted join/leave events
- active_mask updates and exposure in `info`
- week-1 demo artifacts and tests

Not implemented yet:

- auto-trigger events (battery/link degradation)
- role reassignment and advanced recovery logic
- full autonomous reconfiguration policy
