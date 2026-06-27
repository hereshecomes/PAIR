from __future__ import annotations

from typing import Any


def partial_order_constraints(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("oracle", {}).get("partial_order", {"constraints": []})


def reference_trace(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return episode.get("oracle", {}).get("reference_trace", [])
