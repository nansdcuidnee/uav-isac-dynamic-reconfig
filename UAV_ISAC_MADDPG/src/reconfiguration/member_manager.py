"""Active member state manager for dynamic UAV participation."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


class MemberManager:
    """Owns active_mask and safe activate/deactivate operations."""

    def __init__(
        self,
        max_drones: int | None = None,
        initial_active_drones: int | None = None,
        active_mask: Iterable[int] | None = None,
    ):
        if active_mask is not None:
            self.active_mask = np.array(list(active_mask), dtype=int)
            self.max_drones = len(self.active_mask)
            return

        if max_drones is None:
            raise ValueError("max_drones must be provided if active_mask is not set")

        self.max_drones = int(max_drones)
        self.active_mask = np.zeros(self.max_drones, dtype=int)

        if initial_active_drones is not None:
            active_count = min(int(initial_active_drones), self.max_drones)
            self.active_mask[:active_count] = 1

    def _check_id(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self.max_drones:
            raise ValueError(
                f"agent_id {agent_id} is out of bounds for max_drones {self.max_drones}"
            )

    def activate(self, agent_id: int):
        self._check_id(agent_id)
        if self.active_mask[agent_id] == 1:
            return self.active_mask.copy()

        if self.num_active() >= self.max_drones:
            raise ValueError("cannot activate more members than max_drones")

        self.active_mask[agent_id] = 1
        return self.active_mask.copy()

    def deactivate(self, agent_id: int):
        self._check_id(agent_id)
        if self.active_mask[agent_id] == 0:
            return self.active_mask.copy()

        self.active_mask[agent_id] = 0
        return self.active_mask.copy()

    def get_active_ids(self) -> List[int]:
        return np.where(self.active_mask == 1)[0].tolist()

    def num_active(self) -> int:
        return int(np.sum(self.active_mask))

    def get_active_mask(self):
        return self.active_mask.copy()

    def get_active_mask_list(self) -> List[int]:
        return self.active_mask.tolist()
