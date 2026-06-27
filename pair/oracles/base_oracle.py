from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOracle(ABC):
    @abstractmethod
    def plan(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError
