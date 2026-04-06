#!/usr/bin/env python3
"""Tests for scripted event manager behavior."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconfiguration.event_manager import EventManager


def test_event_trigger():
    event_schedule = [
        {"step": 20, "type": "leave", "agent_id": 2, "reason": "scripted_leave"},
        {"step": 40, "type": "join", "agent_id": 2, "reason": "scripted_rejoin"},
        {"step": 41, "type": "join", "agent_id": 3, "reason": "scripted_join"},
    ]
    manager = EventManager(event_schedule)

    assert manager.get_event(19)["trigger_flag"] is False

    event20 = manager.get_event(20)
    assert event20["trigger_flag"] is True
    assert event20["event_type"] == "leave"
    assert event20["affected_agents"] == [2]

    assert manager.get_event(21)["trigger_flag"] is False

    event40 = manager.get_event(40)
    assert event40["trigger_flag"] is True
    assert event40["event_type"] == "join"
    assert event40["affected_agents"] == [2]

    event41 = manager.get_event(41)
    assert event41["trigger_flag"] is True
    assert event41["event_type"] == "join"
    assert event41["affected_agents"] == [3]

    assert manager.get_event(42)["trigger_flag"] is False


if __name__ == "__main__":
    test_event_trigger()
    print("test_event_manager.py passed")
