from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    episode: dict[str, Any]
    trace: list[dict[str, Any]]

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def available_tools(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def state_hash(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def final_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        raise NotImplementedError
