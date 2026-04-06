"""Step-scripted event manager for week-1 dynamic join/leave scenarios."""

from __future__ import annotations

from typing import Any, Dict, List


class EventManager:
    """Reads a fixed schedule and returns normalized event payloads per step."""

    def __init__(self, event_schedule: List[Dict[str, Any]] | None):
        self.event_schedule = event_schedule or []

    def get_event(self, current_step: int) -> Dict[str, Any]:
        for event in self.event_schedule:
            if event.get("step") == current_step:
                return {
                    "trigger_flag": True,
                    "event_type": event.get("type"),
                    "affected_agents": [event.get("agent_id")],
                    "reason": event.get("reason"),
                    "step": current_step,
                }

        return {
            "trigger_flag": False,
            "event_type": None,
            "affected_agents": [],
            "reason": None,
            "step": current_step,
        }
