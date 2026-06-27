from __future__ import annotations

from typing import Any

from pair.envs.calendar_world import CalendarWorld
from pair.envs.codebase_world import CodebaseWorld
from pair.envs.spreadsheet_world import SpreadsheetWorld

REQUIRED_EPISODE_KEYS = {"episode_id", "family_id", "domain", "task_type", "world_type", "goal", "world", "oracle", "intervention", "affected_cone"}
REQUIRED_WORLDS = {"base", "irrelevant", "causal", "distractor", "conflict"}


def _env_for(episode: dict[str, Any]):
    if episode.get("domain") == "calendar":
        return CalendarWorld(episode)
    if episode.get("domain") == "spreadsheet":
        return SpreadsheetWorld(episode)
    if episode.get("domain") == "codebase":
        return CodebaseWorld(episode)
    raise ValueError(f"unknown domain: {episode.get('domain')}")


def replay_reference(episode: dict[str, Any]) -> bool:
    try:
        env = _env_for(episode)
        for action in episode["oracle"]["reference_trace"]:
            env.step(action)
        return env.final_check()
    except Exception:
        return False


def schema_check(episode: dict[str, Any]) -> dict[str, Any]:
    failures = [f"missing {key}" for key in sorted(REQUIRED_EPISODE_KEYS) if key not in episode]
    domain = episode.get("domain")
    if domain not in {"calendar", "spreadsheet", "codebase"}:
        failures.append("domain must be calendar, spreadsheet, or codebase")
    if episode.get("world_type") not in REQUIRED_WORLDS:
        failures.append("invalid world_type")
    world = episode.get("world", {})
    if domain == "calendar" and not isinstance(world.get("events"), list):
        failures.append("world.events must be a list")
    if domain == "spreadsheet":
        if not isinstance(world.get("cells"), dict):
            failures.append("world.cells must be a dict")
        if not isinstance(world.get("tables"), dict):
            failures.append("world.tables must be a dict")
    if domain == "codebase" and not isinstance(world.get("files"), dict):
        failures.append("world.files must be a dict")
    if not isinstance(episode.get("oracle", {}).get("reference_trace"), list):
        failures.append("oracle.reference_trace must be a list")
    return {"name": "schema", "passed": not failures, "failures": failures}


def no_leakage_check(episode: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    goal = episode.get("goal", {})
    forbidden_goal_keys = {"oracle", "reference_trace", "answer", "target_event", "expected_cells", "expected_views", "affected_cone"}
    leaked = sorted(k for k in forbidden_goal_keys if k in goal)
    if leaked:
        failures.append(f"goal leaks hidden keys: {leaked}")
    if episode.get("world_type") == "irrelevant" and episode.get("affected_cone"):
        failures.append("irrelevant world should have empty affected_cone")
    return {"name": "no_leakage", "passed": not failures, "failures": failures}


def oracle_replay_check(episode: dict[str, Any]) -> dict[str, Any]:
    ok = replay_reference(episode)
    return {"name": "oracle_replay", "passed": ok, "failures": [] if ok else ["oracle replay failed"]}


def validate_episode(episode: dict[str, Any]) -> dict[str, Any]:
    checks = [schema_check(episode), no_leakage_check(episode), oracle_replay_check(episode)]
    failures = [failure for check in checks for failure in check["failures"]]
    return {"episode_id": episode.get("episode_id"), "passed": all(c["passed"] for c in checks), "checks": checks, "failures": failures}


def validate_family(family: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [validate_episode(ep) for ep in episodes]
    world_types = {ep.get("world_type") for ep in episodes}
    failures: list[Any] = [r for r in reports if not r["passed"]]
    if world_types != REQUIRED_WORLDS:
        failures.append({"episode_id": family.get("family_id"), "failures": ["missing paired world types"]})
    task_types = {ep.get("task_type") for ep in episodes}
    if len(task_types) != 1:
        failures.append({"episode_id": family.get("family_id"), "failures": ["mixed task types within family"]})
    return {"family_id": family.get("family_id"), "passed": not failures, "episode_reports": reports, "failures": failures}
