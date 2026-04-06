#!/usr/bin/env python3
"""Run the week-1 patent dynamic scenario and generate demo artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.uav_env import UAVEnv
from src.visualization.plot_active_mask import plot_active_mask
from src.visualization.plot_event_timeline import (
    plot_event_timeline,
    write_event_timeline_text,
)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_runtime_params(config: Dict) -> Dict:
    env_cfg = config.get("env", {})
    event_cfg = config.get("event", {})
    return {
        "max_drones": int(env_cfg.get("max_drones", env_cfg.get("num_drones", 3))),
        "num_users": int(env_cfg.get("num_users", 20)),
        "initial_active_drones": int(env_cfg.get("initial_active_drones", 3)),
        "max_steps": int(env_cfg.get("max_steps", 60)),
        "alpha": float(env_cfg.get("alpha", 1.0)),
        "beta": float(env_cfg.get("beta", 2.0)),
        "gamma": float(env_cfg.get("gamma", 1.0)),
        "event_schedule": event_cfg.get("schedule", []) if event_cfg.get("enabled", False) else [],
    }


def _build_event_lookup(event_schedule: List[Dict]) -> Dict[int, Dict]:
    return {int(item.get("step")): item for item in event_schedule}


def run_dynamic_demo(config_path: Path, logs_dir: Path) -> Dict[str, str]:
    config = load_config(config_path)
    params = _extract_runtime_params(config)
    event_lookup = _build_event_lookup(params["event_schedule"])

    env = UAVEnv(
        num_drones=params["max_drones"],
        num_users=params["num_users"],
        max_steps=params["max_steps"],
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        max_drones=params["max_drones"],
        initial_active_drones=params["initial_active_drones"],
        event_schedule=params["event_schedule"],
    )

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "dynamic_env_demo.log"
    timeline_txt = logs_dir / "event_timeline.txt"
    timeline_png = logs_dir / "event_timeline.png"
    active_mask_png = logs_dir / "active_mask_change.png"

    active_mask_history: List[List[int]] = []
    event_history: List[Dict] = []

    _, info = env.reset()
    active_mask_history.append(info.get("active_mask", []))

    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"[Step 0] active_mask={active_mask_history[-1]}\n")

        for step in range(1, params["max_steps"] + 1):
            # Use deterministic no-op actions for stable, reproducible logs.
            actions = np.zeros(params["max_drones"], dtype=int)
            _, _, done, _, info = env.step(actions)

            current_mask = info.get("active_mask", [])
            active_mask_history.append(current_mask)

            if info.get("trigger_event", False):
                scheduled = event_lookup.get(step, {})
                record = {
                    "step": step,
                    "event_type": info.get("event_type"),
                    "agent_id": scheduled.get("agent_id", -1),
                    "reason": scheduled.get("reason", "unknown"),
                    "active_mask": current_mask,
                }
                event_history.append(record)
                log.write(
                    f"[Step {step}] EVENT: {record['event_type']} agent {record['agent_id']} ({record['reason']})\n"
                )
                log.write(f"[Step {step}] active_mask={current_mask}\n")
            else:
                log.write(f"[Step {step}] active_mask={current_mask}\n")

            if done:
                break

    write_event_timeline_text(event_history, str(timeline_txt))
    plot_event_timeline(event_history, params["max_steps"], str(timeline_png))
    plot_active_mask(active_mask_history, str(active_mask_png))

    return {
        "log": str(log_file),
        "timeline_txt": str(timeline_txt),
        "timeline_png": str(timeline_png),
        "active_mask_png": str(active_mask_png),
    }


def main() -> None:
    config_path = REPO_ROOT / "src" / "configs" / "patent_dynamic.yaml"
    logs_dir = REPO_ROOT / "logs"

    outputs = run_dynamic_demo(config_path=config_path, logs_dir=logs_dir)
    print("Dynamic demo finished.")
    print(f"log: {outputs['log']}")
    print(f"timeline text: {outputs['timeline_txt']}")
    print(f"timeline plot: {outputs['timeline_png']}")
    print(f"active mask plot: {outputs['active_mask_png']}")


if __name__ == "__main__":
    main()
