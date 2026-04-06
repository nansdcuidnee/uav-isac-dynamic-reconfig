#!/usr/bin/env python3
"""Tests for active mask transitions and inactive-action handling."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.uav_env import UAVEnv


def _build_env(event_schedule):
    return UAVEnv(
        num_drones=4,
        num_users=20,
        max_steps=60,
        alpha=1.0,
        beta=2.0,
        gamma=1.0,
        max_drones=4,
        initial_active_drones=3,
        event_schedule=event_schedule,
    )


def test_active_mask_update():
    env = _build_env(
        [
            {"step": 20, "type": "leave", "agent_id": 2, "reason": "scripted_leave"},
            {"step": 40, "type": "join", "agent_id": 2, "reason": "scripted_rejoin"},
            {"step": 41, "type": "join", "agent_id": 3, "reason": "scripted_join"},
        ]
    )

    _, info = env.reset()
    assert info.get("active_mask", []) == [1, 1, 1, 0]

    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(4, dtype=int))
    assert info.get("active_mask", []) == [1, 1, 0, 0]

    for _ in range(20, 40):
        _, _, _, _, info = env.step(np.zeros(4, dtype=int))
    assert info.get("active_mask", []) == [1, 1, 1, 0]

    _, _, _, _, info = env.step(np.zeros(4, dtype=int))
    assert info.get("active_mask", []) == [1, 1, 1, 1]


def test_inactive_uav_actions():
    env = _build_env([
        {"step": 10, "type": "leave", "agent_id": 2, "reason": "scripted_leave"}
    ])

    env.reset()
    initial_pos = env.drone_pos[2].copy()

    for _ in range(10):
        env.step(np.zeros(4, dtype=int))

    before_leave_pos = env.drone_pos[2].copy()
    assert not np.array_equal(initial_pos, before_leave_pos)

    env.step(np.zeros(4, dtype=int))
    after_leave_pos = env.drone_pos[2].copy()

    for _ in range(11, 15):
        env.step(np.zeros(4, dtype=int))

    final_pos = env.drone_pos[2].copy()
    assert np.array_equal(after_leave_pos, final_pos)


if __name__ == "__main__":
    test_active_mask_update()
    test_inactive_uav_actions()
    print("test_active_mask.py passed")
