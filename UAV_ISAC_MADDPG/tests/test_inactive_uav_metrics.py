#!/usr/bin/env python3
"""Tests that inactive UAVs do not contribute to key metrics."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.uav_env import UAVEnv


def test_inactive_uav_metrics():
    env = UAVEnv(
        num_drones=4,
        num_users=20,
        max_steps=20,
        alpha=1.0,
        beta=2.0,
        gamma=1.0,
        max_drones=4,
        initial_active_drones=3,
        event_schedule=[
            {"step": 10, "type": "leave", "agent_id": 2, "reason": "scripted_leave"}
        ],
    )

    env.reset()

    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(4, dtype=int))

    coverage_before = info.get("coverage_rate", 0)
    rate_before = info.get("total_system_rate", 0)

    _, _, _, _, info = env.step(np.zeros(4, dtype=int))

    coverage_after = info.get("coverage_rate", 0)
    rate_after = info.get("total_system_rate", 0)

    assert coverage_after <= coverage_before + 1e-9
    assert rate_after <= rate_before + 1e-9


if __name__ == "__main__":
    test_inactive_uav_metrics()
    print("test_inactive_uav_metrics.py passed")
