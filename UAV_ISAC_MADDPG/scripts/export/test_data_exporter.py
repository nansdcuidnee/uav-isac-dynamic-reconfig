#!/usr/bin/env python3
"""Generate a minimal trajectory sample and export Unity-friendly data."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.uav_env import UAVEnv
from scripts.export.data_exporter import export_simulation_data


def generate_test_trajectory_data(num_drones: int = 3, num_steps: int = 100):
    trajectory_data = []
    for episode in range(1):
        for step in range(num_steps):
            for uav_id in range(num_drones):
                x = 100 + uav_id * 50 + step * 0.5
                y = 10 + uav_id * 10
                z = 200 + step * 0.1
                trajectory_data.append(
                    {
                        "episode": episode,
                        "step": step,
                        "uav_id": uav_id,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
    return trajectory_data


def main() -> None:
    env = UAVEnv(num_drones=3, num_users=10, max_steps=100)
    trajectory_data = generate_test_trajectory_data(num_drones=3, num_steps=100)
    export_simulation_data(env, trajectory_data, export_dir="unity_data")
    print("test data export done")


if __name__ == "__main__":
    main()
