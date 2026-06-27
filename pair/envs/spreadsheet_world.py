from __future__ import annotations

import copy
import re
from typing import Any

from pair.envs.base import BaseEnv
from pair.utils.io import stable_json_hash

CELL_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


def _as_number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except Exception:
        return 0


def _cell_key(cell: str) -> tuple[str, int]:
    m = CELL_RE.match(cell)
    if not m:
        return (cell, 0)
    return (m.group(1), int(m.group(2)))


def _range_cells(start: str, end: str) -> list[str]:
    col_a, row_a = _cell_key(start)
    col_b, row_b = _cell_key(end)
    if col_a != col_b:
        return [start, end]
    lo, hi = sorted([row_a, row_b])
    return [f"{col_a}{i}" for i in range(lo, hi + 1)]


class SpreadsheetWorld(BaseEnv):
    schema_version = "pair.spreadsheet.v0.1"

    def __init__(self, episode: dict[str, Any]):
        self.episode = copy.deepcopy(episode)
        self.reset()

    def reset(self) -> dict[str, Any]:
        world = self.episode["world"]
        self.cells = copy.deepcopy(world.get("cells", {}))
        self.tables = copy.deepcopy(world.get("tables", {}))
        self.validation = copy.deepcopy(world.get("validation", {}))
        self.views: dict[str, list[dict[str, Any]]] = copy.deepcopy(world.get("views", {}))
        self.done = False
        self.trace: list[dict[str, Any]] = []
        self._recalculate_all()
        return {"goal": copy.deepcopy(self.episode["goal"])}

    def available_tools(self) -> list[str]:
        return ["read_cell", "read_range", "inspect_formula", "validate_cell", "update_cell", "update_formula", "read_table", "filter_rows", "sort_rows", "finish"]

    def _record(self, tool: str, args: dict[str, Any], ok: bool, observation: dict[str, Any], state_delta: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {"index": len(self.trace), "tool": tool, "args": copy.deepcopy(args), "ok": bool(ok), "observation": copy.deepcopy(observation), "state_delta": copy.deepcopy(state_delta or {})}
        self.trace.append(row)
        return copy.deepcopy(observation)

    def _cell(self, cell: str) -> dict[str, Any]:
        return self.cells.setdefault(cell, {"value": None, "formula": None, "depends_on": [], "label": cell})

    def _formula_value(self, formula: str) -> float | int | None:
        expr = str(formula or "").strip()
        if expr.startswith("="):
            expr = expr[1:]
        sum_match = re.fullmatch(r"SUM\(([A-Z]+[0-9]+):([A-Z]+[0-9]+)\)", expr)
        if sum_match:
            return sum(_as_number(self._cell(c).get("value")) for c in _range_cells(sum_match.group(1), sum_match.group(2)))
        if "*" in expr:
            parts = [p.strip() for p in expr.split("*")]
            val: float | int = 1
            for part in parts:
                val *= _as_number(self._cell(part).get("value") if CELL_RE.match(part) else part)
            return int(val) if float(val).is_integer() else val
        if "+" in expr:
            val = sum(_as_number(self._cell(p.strip()).get("value") if CELL_RE.match(p.strip()) else p.strip()) for p in expr.split("+"))
            return int(val) if float(val).is_integer() else val
        if CELL_RE.match(expr):
            return _as_number(self._cell(expr).get("value"))
        return None

    def _formula_deps(self, formula: str) -> list[str]:
        expr = str(formula or "")
        deps: list[str] = []
        m = re.search(r"SUM\(([A-Z]+[0-9]+):([A-Z]+[0-9]+)\)", expr)
        if m:
            deps.extend(_range_cells(m.group(1), m.group(2)))
        deps.extend(re.findall(r"[A-Z]+[0-9]+", expr))
        out: list[str] = []
        for dep in deps:
            if dep not in out:
                out.append(dep)
        return out

    def _recalculate_all(self) -> None:
        for _ in range(3):
            for cell, payload in list(self.cells.items()):
                formula = payload.get("formula")
                if formula:
                    payload["depends_on"] = self._formula_deps(formula)
                    payload["value"] = self._formula_value(formula)

    def _validate(self, cell: str, value: Any) -> tuple[bool, list[str]]:
        rules = self.validation.get(cell, {})
        value_n = _as_number(value)
        errors: list[str] = []
        if "min" in rules and value_n < rules["min"]:
            errors.append(f"below_min:{rules['min']}")
        if "max" in rules and value_n > rules["max"]:
            errors.append(f"above_max:{rules['max']}")
        if rules.get("integer") and int(value_n) != value_n:
            errors.append("not_integer")
        return not errors, errors

    def _filter_table(self, table: str, column: str, op: str, value: Any) -> list[dict[str, Any]]:
        rows = copy.deepcopy(self.tables.get(table, []))
        if op == "=":
            op = "=="
        value_n = _as_number(value)
        out = []
        for row in rows:
            left = row.get(column)
            ok = False
            if op == "==":
                ok = left == value
            elif op == ">=":
                ok = _as_number(left) >= value_n
            elif op == "<=":
                ok = _as_number(left) <= value_n
            elif op == ">":
                ok = _as_number(left) > value_n
            elif op == "<":
                ok = _as_number(left) < value_n
            if ok:
                out.append(row)
        return out

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            return self._record("after_done", action, False, {"error": "episode already finished"})
        tool = str(action.get("tool", ""))
        args = copy.deepcopy(action.get("args", {}))
        if tool == "read_cell":
            cell = str(args.get("cell", ""))
            return self._record(tool, args, True, {"cell": cell, "payload": copy.deepcopy(self._cell(cell))})
        if tool == "read_range":
            cells = [str(c) for c in args.get("cells", [])]
            return self._record(tool, {"cells": cells}, True, {"cells": {c: copy.deepcopy(self._cell(c)) for c in cells}})
        if tool == "inspect_formula":
            cell = str(args.get("cell", ""))
            payload = copy.deepcopy(self._cell(cell))
            return self._record(tool, args, True, {"cell": cell, "formula": payload.get("formula"), "depends_on": payload.get("depends_on", [])})
        if tool == "validate_cell":
            cell = str(args.get("cell", ""))
            ok, errors = self._validate(cell, args.get("value"))
            return self._record(tool, args, True, {"valid": ok, "errors": errors, "cell": cell})
        if tool == "update_cell":
            cell = str(args.get("cell", ""))
            value = args.get("value")
            if not CELL_RE.match(cell):
                return self._record(tool, args, False, {"updated": False, "error": f"invalid cell address {cell}", "cell": cell}, {"write_cell": cell, "failed_write": True})
            ok, errors = self._validate(cell, value)
            if not ok:
                return self._record(tool, args, False, {"updated": False, "errors": errors, "cell": cell}, {"write_cell": cell, "failed_write": True})
            before = copy.deepcopy(self._cell(cell))
            self.cells[cell] = {"value": _as_number(value), "formula": None, "depends_on": [], "label": before.get("label", cell)}
            self._recalculate_all()
            return self._record(tool, args, True, {"updated": True, "cell": cell, "payload": copy.deepcopy(self._cell(cell))}, {"write_cell": cell, "changed_fields": {"value": [before.get("value"), _as_number(value)]}})
        if tool == "update_formula":
            cell = str(args.get("cell", ""))
            formula = str(args.get("formula", ""))
            if not CELL_RE.match(cell):
                return self._record(tool, args, False, {"updated": False, "error": f"invalid cell address {cell}", "cell": cell}, {"write_cell": cell, "failed_write": True})
            before = copy.deepcopy(self._cell(cell))
            self.cells[cell] = {"value": before.get("value"), "formula": formula, "depends_on": self._formula_deps(formula), "label": before.get("label", cell)}
            self._recalculate_all()
            return self._record(tool, args, True, {"updated": True, "cell": cell, "payload": copy.deepcopy(self._cell(cell))}, {"write_cell": cell, "changed_fields": {"formula": [before.get("formula"), formula]}})
        if tool == "read_table":
            table = str(args.get("table", ""))
            rows = copy.deepcopy(self.tables.get(table, []))
            return self._record(tool, args, True, {"table": table, "rows": rows, "count": len(rows)})
        if tool == "filter_rows":
            table = str(args.get("table", ""))
            view = str(args.get("view", "filtered_view"))
            column = str(args.get("column", ""))
            if table not in self.tables:
                return self._record(tool, args, False, {"error": f"unknown table {table}"})
            if self.tables[table] and column not in self.tables[table][0]:
                return self._record(tool, args, False, {"error": f"unknown column {column}"})
            op = str(args.get("op", "=="))
            if op == "=":
                args["op"] = "=="
                op = "=="
            rows = self._filter_table(table, column, op, args.get("value"))
            self.views[view] = copy.deepcopy(rows)
            return self._record(tool, args, True, {"view": view, "rows": rows, "count": len(rows)}, {"write_view": view, "row_ids": [r.get("id") for r in rows]})
        if tool == "sort_rows":
            table = str(args.get("table", ""))
            view = str(args.get("view", "sorted_view"))
            column = str(args.get("column", ""))
            descending = bool(args.get("descending", False))
            if table not in self.tables:
                return self._record(tool, args, False, {"error": f"unknown table {table}"})
            if self.tables[table] and column not in self.tables[table][0]:
                return self._record(tool, args, False, {"error": f"unknown column {column}"})
            rows = sorted(copy.deepcopy(self.tables[table]), key=lambda r: r.get(column), reverse=descending)
            self.views[view] = rows
            return self._record(tool, args, True, {"view": view, "rows": rows, "count": len(rows)}, {"write_view": view, "row_ids": [r.get("id") for r in rows]})
        if tool == "finish":
            self.done = True
            return self._record(tool, args, True, {"finished": True, "final_success": self.final_check()})
        return self._record(tool, args, False, {"error": f"unknown tool {tool}"})

    def _verification_ok(self) -> bool:
        if not self.episode.get("goal", {}).get("verification_required"):
            return True
        write_tools = {"update_cell", "update_formula", "filter_rows", "sort_rows"}
        read_tools = {"read_cell", "read_range", "inspect_formula", "validate_cell", "read_table"}
        first_write = next((s["index"] for s in self.trace if s.get("tool") in write_tools), None)
        if first_write is None:
            return False
        return any(s.get("tool") in read_tools and s["index"] < first_write for s in self.trace)

    def final_check(self) -> bool:
        oracle = self.episode.get("oracle", {})
        if not self._verification_ok():
            return False
        for cell, expected in oracle.get("expected_cells", {}).items():
            payload = self._cell(cell)
            if "value" in expected and payload.get("value") != expected.get("value"):
                return False
            if "formula" in expected and payload.get("formula") != expected.get("formula"):
                return False
        for view, expected_ids in oracle.get("expected_views", {}).items():
            ids = [r.get("id") for r in self.views.get(view, [])]
            if ids != expected_ids:
                return False
        wrong_cells = set(oracle.get("forbidden_write_cells", []))
        wrong_views = set(oracle.get("forbidden_write_views", []))
        for step in self.trace:
            delta = step.get("state_delta", {})
            if delta.get("write_cell") in wrong_cells and step.get("ok"):
                return False
            if delta.get("write_view") in wrong_views and step.get("ok"):
                return False
        return True

    def state_hash(self) -> str:
        return stable_json_hash({"cells": self.cells, "tables": self.tables, "views": self.views})

    def summary(self) -> dict[str, Any]:
        oracle = self.episode.get("oracle", {})
        return {
            "state_hash": self.state_hash(),
            "cells": copy.deepcopy(self.cells),
            "tables": copy.deepcopy(self.tables),
            "views": copy.deepcopy(self.views),
            "target_cell_states": {c: copy.deepcopy(self._cell(c)) for c in oracle.get("expected_cells", {})},
            "target_view_states": {v: [r.get("id") for r in self.views.get(v, [])] for v in oracle.get("expected_views", {})},
            "verification_ok": self._verification_ok(),
            "final_success": self.final_check(),
        }
