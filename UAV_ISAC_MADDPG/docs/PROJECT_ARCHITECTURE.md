# Project Architecture

## Overview

`main.py` is the unified entry point. It parses CLI arguments and routes execution to training, evaluation, or visualization paths.

Core runtime stack:

1. load config from `src/configs/*`
2. build `UAVEnv`
3. build agent (MADDPG/DDPG/DQN)
4. run train/eval loop
5. output artifacts to `results/` and `logs/`

## Dynamic Reconfiguration (Week-1)

Week-1 adds a minimal, testable dynamic member mechanism.

- fixed scripted schedule only
- event types: `leave`, `join`
- active state tracked by `active_mask`
- inactive UAVs excluded from action and metric contribution

Current canonical scenario: `3 -> 2 -> 4`.

## Directory Boundaries

- `src/envs/`
  - `uav_env.py`: environment logic (`reset`, `step`, rewards, done, info).
  - `models/`, `dynamics/`: domain model internals.
  - legacy compatibility shims remain for migration safety.
- `src/reconfiguration/`
  - `event_manager.py`: schedule reader and per-step normalized event payload.
  - `member_manager.py`: mask ownership and activation safety checks.
- `src/training/`: trainer/logger/checkpoint/evaluator.
- `src/agents/`: policy implementations.
- `src/visualization/`
  - `plot_event_timeline.py`: event timeline text/figure outputs.
  - `plot_active_mask.py`: active mask heatmap output.
- `scripts/run_patent_dynamic.py`: one-command scenario runner.
- `validate_dynamic_events.py`: backward-compatible wrapper to the script above.
- `tests/`: dynamic behavior tests.

## Data and Control Flow

Training path:

1. `main.py --mode train` loads config.
2. `src/training/train.py` builds env + agent.
3. `Trainer` collects transitions and updates policy.
4. checkpoints/curves are saved under `results/`.

Dynamic demo path:

1. `python scripts/run_patent_dynamic.py`
2. load `src/configs/patent_dynamic.yaml`
3. instantiate `UAVEnv`
4. `EventManager` resolves step events
5. `MemberManager` updates active states
6. write demo outputs to `logs/`

## Artifacts

Primary outputs:

- `logs/dynamic_env_demo.log`
- `logs/event_timeline.txt`
- `logs/event_timeline.png`
- `logs/active_mask_change.png`

Training outputs remain in `results/models`, `results/plots`, `results/videos`.

## Current Status

Implemented:

- scripted join/leave
- active_mask propagation in info
- join reset behavior and inactive exclusion
- smoke tests and demo generation script

Deferred to later phases:

- automatic event triggers
- role-vector and advanced recovery/reconfiguration policies
- full patent-grade autonomous strategy layer
