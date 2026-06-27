from __future__ import annotations

import json
from typing import Any


def _sorted_people(values: list[str] | None) -> list[str]:
    return sorted(str(v) for v in (values or []))


def _signature(tool: str, args: dict[str, Any]) -> str:
    return json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"))


def canonicalize_step(step: dict[str, Any]) -> dict[str, Any]:
    tool = step.get("tool")
    args = dict(step.get("args") or {})
    ok = bool(step.get("ok", True))
    if tool == "check_availability":
        key_args = {"participants": _sorted_people(args.get("participants")), "start": args.get("start"), "end": args.get("end"), "room": args.get("room"), "exclude_event_id": args.get("exclude_event_id")}
        return {"tool": tool, "intent": "check_conflict", "effect": "read", "key_args": key_args, "touches": ["event.conflict_constraints", "query.availability"], "ok": ok}
    if tool == "list_events":
        key_args = {"participants": _sorted_people(args.get("participants")), "title": args.get("title"), "start": args.get("start"), "end": args.get("end"), "include_canceled": args.get("include_canceled")}
        return {"tool": tool, "intent": "read_events", "effect": "read", "key_args": key_args, "touches": ["calendar.events"], "ok": ok}
    if tool == "create_event":
        key_args = {"title": args.get("title"), "participants": _sorted_people(args.get("participants")), "start": args.get("start"), "end": args.get("end"), "room": args.get("room")}
        return {"tool": tool, "intent": "write_event", "effect": "write", "key_args": key_args, "touches": ["event.participants", "event.start_time", "event.location"], "ok": ok}
    if tool == "update_event":
        changed = sorted((step.get("state_delta") or {}).get("changed_fields", {}).keys())
        key_args = {"event_id": args.get("event_id"), "start": args.get("start"), "end": args.get("end"), "room": args.get("room"), "changed_fields": changed}
        return {"tool": tool, "intent": "update_event", "effect": "write", "key_args": key_args, "touches": ["event.start_time" if f in {"start", "end"} else f"event.{f}" for f in changed], "ok": ok}
    if tool == "cancel_event":
        return {"tool": tool, "intent": "cancel_event", "effect": "write", "key_args": {"event_id": args.get("event_id")}, "touches": ["event.cancel_scope"], "ok": ok}
    if tool == "send_notification":
        return {"tool": tool, "intent": "notify", "effect": "notify", "key_args": {"event_id": args.get("event_id"), "recipients": _sorted_people(args.get("recipients"))}, "touches": ["notification.recipients"], "ok": ok}
    if tool == "read_cell":
        return {"tool": tool, "intent": "read_cell", "effect": "read", "key_args": {"cell": args.get("cell")}, "touches": ["cell.value"], "ok": ok}
    if tool == "read_range":
        return {"tool": tool, "intent": "read_range", "effect": "read", "key_args": {"cells": sorted(str(c) for c in args.get("cells", []))}, "touches": ["cell.value", "formula.dependent_cells"], "ok": ok}
    if tool == "inspect_formula":
        return {"tool": tool, "intent": "inspect_formula", "effect": "read", "key_args": {"cell": args.get("cell")}, "touches": ["cell.formula"], "ok": ok}
    if tool == "validate_cell":
        return {"tool": tool, "intent": "validate_cell", "effect": "read", "key_args": {"cell": args.get("cell"), "value": args.get("value")}, "touches": ["cell.validation"], "ok": ok}
    if tool == "read_table":
        return {"tool": tool, "intent": "read_table", "effect": "read", "key_args": {"table": args.get("table")}, "touches": ["table.rows"], "ok": ok}
    if tool == "update_cell":
        return {"tool": tool, "intent": "update_cell", "effect": "write", "key_args": {"cell": args.get("cell"), "value": args.get("value")}, "touches": ["cell.value", "formula.dependent_cells", "summary.cells"], "ok": ok}
    if tool == "update_formula":
        return {"tool": tool, "intent": "update_formula", "effect": "write", "key_args": {"cell": args.get("cell"), "formula": args.get("formula")}, "touches": ["cell.formula", "formula.dependent_cells", "summary.cells"], "ok": ok}
    if tool == "filter_rows":
        return {"tool": tool, "intent": "filter_rows", "effect": "write", "key_args": {"table": args.get("table"), "column": args.get("column"), "op": args.get("op"), "value": args.get("value"), "view": args.get("view")}, "touches": ["table.rows", "view.membership"], "ok": ok}
    if tool == "sort_rows":
        return {"tool": tool, "intent": "sort_rows", "effect": "write", "key_args": {"table": args.get("table"), "column": args.get("column"), "descending": args.get("descending"), "view": args.get("view")}, "touches": ["table.rows", "view.order"], "ok": ok}
    if tool == "read_file":
        return {"tool": tool, "intent": "read_file", "effect": "read", "key_args": {"path": args.get("path")}, "touches": ["code.file"], "ok": ok}
    if tool == "search_code":
        return {"tool": tool, "intent": "search_code", "effect": "read", "key_args": {"query": args.get("query")}, "touches": ["code.index"], "ok": ok}
    if tool == "run_tests":
        return {"tool": tool, "intent": "run_tests", "effect": "read", "key_args": {}, "touches": ["test.result"], "ok": ok}
    if tool == "edit_file":
        return {"tool": tool, "intent": "edit_file", "effect": "write", "key_args": {"path": args.get("path")}, "touches": ["code.file", "code.target_function"], "ok": ok}
    if tool == "finish":
        return {"tool": tool, "intent": "finish", "effect": "finish", "key_args": {}, "touches": [], "ok": ok}
    return {"tool": tool, "intent": "unknown", "effect": "other", "key_args": args, "touches": [], "ok": ok}


def canonicalize_trace(trace: list[dict[str, Any]], fold_repeated_reads: bool = True) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    seen_reads: set[str] = set()
    for raw in trace:
        step = canonicalize_step(raw)
        if fold_repeated_reads and step["effect"] == "read":
            sig = _signature(step["tool"], step["key_args"])
            if sig in seen_reads:
                continue
            seen_reads.add(sig)
        canonical.append(step)
    return canonical


def count_redundant_reads(trace: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    redundant = 0
    for raw in trace:
        step = canonicalize_step(raw)
        if step["effect"] != "read":
            continue
        sig = _signature(step["tool"], step["key_args"])
        if sig in seen:
            redundant += 1
        seen.add(sig)
    return redundant
