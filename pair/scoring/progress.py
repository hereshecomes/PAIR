from __future__ import annotations

from typing import Any

from pair.scoring.canonicalize import canonicalize_trace


WRITE_EFFECTS = {"write", "notify"}


def has_state_changing_progress(trace: list[dict[str, Any]]) -> bool:
    for op in canonicalize_trace(trace):
        if op.get("effect") in WRITE_EFFECTS and op.get("ok", True):
            return True
    return False


def has_task_relevant_verification(episode: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    domain = episode.get("domain")
    if domain == "calendar":
        return _calendar_relevant_read(episode, trace)
    if domain == "spreadsheet":
        return _spreadsheet_relevant_read(episode, trace)
    if domain == "codebase":
        return _codebase_relevant_read(episode, trace)
    return any(op.get("effect") == "read" for op in canonicalize_trace(trace))


def progress_gate(episode: dict[str, Any], prediction: dict[str, Any]) -> float:
    trace = prediction.get("trace", [])
    if has_state_changing_progress(trace):
        return 1.0
    if prediction.get("final_success") and has_task_relevant_verification(episode, trace):
        return 1.0
    return 0.0


def gated_pair(ts: float, ii: float, cs: float, pl: float, tm: float, gate: float) -> float:
    return 0.25 * ts + 0.20 * ii + 0.20 * cs + float(gate) * (0.20 * pl + 0.15 * tm)


def _calendar_relevant_read(episode: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    goal = episode.get("goal", {})
    oracle = episode.get("oracle", {})
    target = oracle.get("target_event") or {}
    target_id = oracle.get("target_event_id") or oracle.get("cancel_event_id")
    target_people = set(target.get("participants") or goal.get("participants") or [])
    target_title = target.get("title") or goal.get("title")
    for step in trace:
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool == "check_availability":
            people = set(args.get("participants") or [])
            if not target_people or people.intersection(target_people):
                return True
        if tool == "list_events":
            people = set(args.get("participants") or [])
            title = args.get("title")
            if target_title and title and title in target_title:
                return True
            if target_people and people.intersection(target_people):
                return True
            events = (step.get("observation") or {}).get("events") or []
            if target_id and any(ev.get("event_id") == target_id for ev in events):
                return True
    return False


def _spreadsheet_relevant_read(episode: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    goal = episode.get("goal", {})
    oracle = episode.get("oracle", {})
    expected_cells = set(oracle.get("expected_cells", {}))
    expected_views = set(oracle.get("expected_views", {}))
    related_cells = set(goal.get("dependent_cells") or []).union(goal.get("depends_on") or []).union(expected_cells)
    target_table = goal.get("table")
    target_view = goal.get("view")
    for step in trace:
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool in {"read_cell", "inspect_formula", "validate_cell"} and args.get("cell") in related_cells:
            return True
        if tool == "read_range" and set(args.get("cells") or []).intersection(related_cells):
            return True
        if tool == "read_table" and (args.get("table") == target_table or expected_views or target_view):
            return True
    return False


def _codebase_relevant_read(episode: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    oracle = episode.get("oracle", {})
    target_files = set(oracle.get("target_files") or oracle.get("expected_files", {}).keys())
    test_names = set(oracle.get("test_names") or [])
    for step in trace:
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool == "run_tests":
            return True
        if tool == "read_file":
            path = args.get("path")
            if path in target_files or path in test_names or str(path).startswith("tests/"):
                return True
        if tool == "search_code":
            query = str(args.get("query") or "")
            if any(query and query in path for path in target_files.union(test_names)):
                return True
    return False
