from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pair.envs.base import BaseEnv


class BaseAgent(ABC):
    name = "base"

    @abstractmethod
    def run(self, env: BaseEnv) -> dict[str, Any]:
        raise NotImplementedError


def finish_row(env: BaseEnv, agent_name: str) -> dict[str, Any]:
    summary = env.summary()
    return {
        "episode_id": env.episode["episode_id"],
        "family_id": env.episode["family_id"],
        "domain": env.episode["domain"],
        "world_type": env.episode["world_type"],
        "agent": agent_name,
        "trace": env.trace,
        "final_state": summary,
        "final_success": bool(summary.get("final_success", False)),
        "cost": {"tool_calls": len(env.trace), "tokens_used": 0, "cost_usd": 0.0},
    }
