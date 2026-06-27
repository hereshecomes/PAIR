from __future__ import annotations

import copy
from typing import Any

from pair.oracles.base_oracle import BaseOracle


class CalendarOracle(BaseOracle):
    def plan(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        return copy.deepcopy(episode["oracle"]["reference_trace"])
