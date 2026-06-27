from __future__ import annotations

from typing import Any


def affected_cone(episode: dict[str, Any]) -> list[str]:
    return list(episode.get("affected_cone", []))


def intervention_role(episode: dict[str, Any]) -> str:
    return str(episode.get("intervention", {}).get("type", "none"))
