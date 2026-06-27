from __future__ import annotations

import copy
import random
from typing import Any

from pair.utils.io import stable_json_hash
from pair.utils.seed import family_seed

WORLD_TYPES = ["base", "irrelevant", "causal", "distractor", "conflict"]
TASK_TYPES = ["update_cell_with_dependency", "repair_formula", "filter_or_sort_table", "verify_before_edit"]


def n(value: Any) -> int | float:
    if isinstance(value, (int, float)):
        return value
    f = float(value)
    return int(f) if f.is_integer() else f


def eval_formula(cells: dict[str, dict[str, Any]], formula: str) -> int | float | None:
    expr = str(formula or "")
    if expr.startswith("="):
        expr = expr[1:]
    if expr.startswith("SUM(") and expr.endswith(")"):
        body = expr[4:-1]
        start, end = body.split(":")
        col = start[0]
        a, b = int(start[1:]), int(end[1:])
        return sum(n(cells[f"{col}{i}"]["value"]) for i in range(min(a, b), max(a, b) + 1))
    if "*" in expr:
        val: int | float = 1
        for part in expr.split("*"):
            part = part.strip()
            val *= n(cells[part]["value"] if part in cells else part)
        return int(val) if float(val).is_integer() else val
    if "+" in expr:
        val = sum(n(cells[p.strip()]["value"] if p.strip() in cells else p.strip()) for p in expr.split("+"))
        return int(val) if float(val).is_integer() else val
    return None


def deps(formula: str) -> list[str]:
    import re
    out = []
    for cell in re.findall(r"[A-Z]+[0-9]+", formula or ""):
        if cell not in out:
            out.append(cell)
    if "SUM(" in (formula or "") and ":" in formula:
        body = formula.split("SUM(", 1)[1].split(")", 1)[0]
        start, end = body.split(":")
        if start[0] == end[0]:
            for i in range(int(start[1:]), int(end[1:]) + 1):
                cell = f"{start[0]}{i}"
                if cell not in out:
                    out.append(cell)
    return out


def recalc(cells: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cells = copy.deepcopy(cells)
    for _ in range(3):
        for cell, payload in cells.items():
            if payload.get("formula"):
                payload["depends_on"] = deps(payload["formula"])
                payload["value"] = eval_formula(cells, payload["formula"])
    return cells


def make_cells(idx: int, rng: random.Random) -> dict[str, dict[str, Any]]:
    price = 10 + idx % 8
    qty = 2 + idx % 5
    cells = {
        "A1": {"value": "Product", "formula": None, "depends_on": [], "label": "Product"},
        "B1": {"value": "Unit Price", "formula": None, "depends_on": [], "label": "Unit Price"},
        "C1": {"value": "Qty", "formula": None, "depends_on": [], "label": "Qty"},
        "D1": {"value": "Line Total", "formula": None, "depends_on": [], "label": "Line Total"},
    }
    for row in range(2, 6):
        row_price = price + row - 2
        row_qty = qty + (row % 2)
        cells[f"A{row}"] = {"value": f"SKU-{idx:03d}-{row}", "formula": None, "depends_on": [], "label": "Product"}
        cells[f"B{row}"] = {"value": row_price, "formula": None, "depends_on": [], "label": "Unit Price"}
        cells[f"C{row}"] = {"value": row_qty, "formula": None, "depends_on": [], "label": "Qty"}
        cells[f"D{row}"] = {"value": None, "formula": f"=B{row}*C{row}", "depends_on": [f"B{row}", f"C{row}"], "label": "Line Total"}
    cells["D6"] = {"value": None, "formula": "=SUM(D2:D5)", "depends_on": ["D2", "D3", "D4", "D5"], "label": "Grand Total"}
    return recalc(cells)


def make_table(idx: int, rng: random.Random) -> list[dict[str, Any]]:
    regions = ["East", "West", "East", "South", "West", "North"]
    rows = []
    for i, region in enumerate(regions, start=1):
        rows.append({"id": f"r{idx:03d}_{i}", "product": f"SKU-{idx:03d}-{i}", "region": region, "amount": 50 + idx * 3 + i * 7, "owner": ["Ava", "Ben", "Cy", "Dee", "Eli", "Fay"][i - 1]})
    return rows


def base_world(idx: int, rng: random.Random, hard_mode: bool = False) -> dict[str, Any]:
    cells = make_cells(idx, rng)
    table = make_table(idx, rng)
    validation = {"B2": {"min": 1, "max": 99, "integer": True}, "C2": {"min": 1, "max": 50, "integer": True}}
    if hard_mode:
        cells["E1"] = {"value": "Audit Total", "formula": None, "depends_on": [], "label": "Audit Total"}
        cells["E6"] = {"value": None, "formula": "=D6+0", "depends_on": ["D6"], "label": "Audit Total"}
        cells = recalc(cells)
        table.extend(
            [
                {"id": f"r{idx:03d}_d1", "product": f"SKU-{idx:03d}-D1", "region": "East", "amount": 50 + idx * 3, "owner": "Distractor"},
                {"id": f"r{idx:03d}_d2", "product": f"SKU-{idx:03d}-D2", "region": "West", "amount": 500 + idx, "owner": "Distractor"},
            ]
        )
        validation["B3"] = {"min": 1, "max": 99, "integer": True}
    return {"sheets": ["Sheet1"], "cells": cells, "tables": {"sales": table}, "validation": validation, "views": {}}


def view_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in rows]


def filter_ids(world: dict[str, Any], column: str, value: Any) -> list[str]:
    return view_ids([r for r in world["tables"]["sales"] if r.get(column) == value])


def sort_ids(world: dict[str, Any], column: str, descending: bool) -> list[str]:
    return view_ids(sorted(world["tables"]["sales"], key=lambda r: r.get(column), reverse=descending))


def build_episode(family_id: str, world_type: str, task_type: str, seed: int, goal: dict[str, Any], world: dict[str, Any], intervention: dict[str, Any], cone: list[str], oracle: dict[str, Any], difficulty: str) -> dict[str, Any]:
    episode_id = f"{family_id}_{world_type}"
    return {
        "schema_version": oracle.get("schema_version", "pair.spreadsheet.v0.1"),
        "domain": "spreadsheet",
        "family_id": family_id,
        "episode_id": episode_id,
        "task_type": task_type,
        "world_type": world_type,
        "difficulty": difficulty,
        "intervention_role": intervention.get("type", "none"),
        "seed": seed,
        "goal": copy.deepcopy(goal),
        "world": copy.deepcopy(world),
        "intervention": copy.deepcopy(intervention),
        "affected_cone": cone,
        "oracle": copy.deepcopy(oracle),
        "state_hash": stable_json_hash({"world": world, "goal": goal, "intervention": intervention, "task_type": task_type}),
    }


def family_record(family_id: str, task_type: str, seed: int, index: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"family_id": family_id, "domain": "spreadsheet", "task_type": task_type, "seed": family_seed(seed, index), "episode_ids": {ep["world_type"]: ep["episode_id"] for ep in episodes}}


def update_or_verify_family(index: int, seed: int, cfg: dict[str, Any], task_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(family_seed(seed, index))
    family_id = f"spreadsheet_{index + 1:06d}"
    hard_mode = bool(cfg.get("hard_mode"))
    base = base_world(index + 1, rng, hard_mode=hard_mode)
    current = base["cells"]["B2"]["value"]
    preferred = current + 5
    fallback = current + 2
    worlds = []
    for wt in WORLD_TYPES:
        world = copy.deepcopy(base)
        intervention = {"type": "none"}
        cone: list[str] = []
        claim: dict[str, Any] = {}
        target_value = preferred
        difficulty = "medium"
        if wt == "irrelevant":
            world["cells"]["Z9"] = {"value": "cosmetic note", "formula": None, "depends_on": [], "label": "Note"}
            intervention = {"type": "irrelevant_note", "entity": "Z9"}
            difficulty = "easy"
        elif wt == "causal":
            world["cells"]["C2"]["value"] += 1
            world["cells"] = recalc(world["cells"])
            intervention = {"type": "dependency_value_changed", "entity": "C2"}
            cone = ["cell.value", "formula.dependent_cells", "summary.cells"]
            difficulty = "hard"
        elif wt == "distractor":
            claim = {"claimed_cell": "B3", "claimed_value": preferred, "must_verify": True}
            intervention = {"type": "distractor_cell", "suggested_cell": "B3"}
            cone = ["instruction.distractor_cell"]
            difficulty = "hard"
        elif wt == "conflict":
            world["validation"]["B2"] = {"min": 1, "max": fallback, "integer": True}
            target_value = fallback
            claim = {"claimed_value": preferred, "must_verify": True}
            intervention = {"type": "validation_conflict", "entity": "B2", "invalid_value": preferred}
            cone = ["cell.validation", "cell.value", "formula.dependent_cells"]
            difficulty = "hard"
        dependent_cells = ["D2", "D6", "E6"] if hard_mode else ["D2", "D6"]
        goal = {"type": task_type, "sheet": "Sheet1", "target_cell": "B2", "preferred_value": preferred, "candidate_values": [preferred, fallback], "stale_value": current, "dependent_cells": dependent_cells, "user_claim": claim, "verification_required": task_type == "verify_before_edit" or bool(claim.get("must_verify"))}
        after = copy.deepcopy(world["cells"])
        after["B2"] = {"value": target_value, "formula": None, "depends_on": [], "label": after["B2"].get("label", "Unit Price")}
        after = recalc(after)
        expected = {"B2": {"value": target_value}, "D2": {"value": after["D2"]["value"]}, "D6": {"value": after["D6"]["value"]}}
        if hard_mode:
            expected["E6"] = {"value": after["E6"]["value"]}
        trace = [{"tool": "read_cell", "args": {"cell": "B2"}}, {"tool": "validate_cell", "args": {"cell": "B2", "value": preferred}}]
        if target_value != preferred:
            trace.append({"tool": "validate_cell", "args": {"cell": "B2", "value": target_value}})
        trace += [{"tool": "update_cell", "args": {"cell": "B2", "value": target_value}}, {"tool": "read_range", "args": {"cells": dependent_cells}}, {"tool": "finish", "args": {}}]
        oracle = {"schema_version": "pair.spreadsheet.v0.2" if hard_mode else "pair.spreadsheet.v0.1", "task_type": task_type, "expected_cells": expected, "expected_views": {}, "target_cell": "B2", "dependent_cells": dependent_cells, "reference_trace": trace, "partial_order": {"constraints": ["verify-before-write", "write-before-finish"]}, "minimal_semantic_calls": len(trace) - 1, "final_checker": "spreadsheet_cell_update_checker", "forbidden_write_cells": [claim.get("claimed_cell")] if claim.get("claimed_cell") else []}
        worlds.append(build_episode(family_id, wt, task_type, family_seed(seed, index), goal, world, intervention, cone, oracle, difficulty))
    return family_record(family_id, task_type, seed, index, worlds), worlds


def repair_formula_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(family_seed(seed, index))
    family_id = f"spreadsheet_{index + 1:06d}"
    hard_mode = bool(cfg.get("hard_mode"))
    base = base_world(index + 1, rng, hard_mode=hard_mode)
    worlds = []
    target_formula = "=B2*C2"
    for wt in WORLD_TYPES:
        world = copy.deepcopy(base)
        world["cells"]["D2"]["formula"] = "=B2+C2"
        world["cells"] = recalc(world["cells"])
        intervention = {"type": "broken_formula"}
        cone = ["cell.formula", "formula.dependent_cells", "summary.cells"]
        claim: dict[str, Any] = {}
        difficulty = "medium"
        if wt == "irrelevant":
            world["cells"]["Z9"] = {"value": "style change", "formula": None, "depends_on": [], "label": "Note"}
            intervention = {"type": "irrelevant_note", "entity": "Z9"}
            cone = []
            difficulty = "easy"
        elif wt == "causal":
            world["cells"]["C2"]["value"] += 2
            world["cells"] = recalc(world["cells"])
            intervention = {"type": "dependency_value_changed", "entity": "C2"}
            difficulty = "hard"
        elif wt == "distractor":
            claim = {"claimed_formula_cell": "D3", "must_verify": True}
            intervention = {"type": "distractor_formula_cell", "suggested_cell": "D3"}
            cone = ["instruction.distractor_cell"]
            difficulty = "hard"
        elif wt == "conflict":
            world["cells"]["B2"]["value"] += 1
            world["cells"] = recalc(world["cells"])
            intervention = {"type": "source_cell_changed", "entity": "B2"}
            difficulty = "hard"
        after = copy.deepcopy(world["cells"])
        after["D2"]["formula"] = target_formula
        after = recalc(after)
        dependent_cells = ["D6", "E6"] if hard_mode else ["D6"]
        goal = {"type": "repair_formula", "sheet": "Sheet1", "formula_cell": "D2", "target_formula": target_formula, "stale_formula": "=B2+C2", "depends_on": ["B2", "C2"], "dependent_cells": dependent_cells, "user_claim": claim, "verification_required": bool(claim.get("must_verify"))}
        trace = [{"tool": "inspect_formula", "args": {"cell": "D2"}}, {"tool": "read_range", "args": {"cells": ["B2", "C2"]}}, {"tool": "update_formula", "args": {"cell": "D2", "formula": target_formula}}, {"tool": "read_range", "args": {"cells": dependent_cells}}, {"tool": "finish", "args": {}}]
        expected_cells = {"D2": {"formula": target_formula, "value": after["D2"]["value"]}, "D6": {"value": after["D6"]["value"]}}
        if hard_mode:
            expected_cells["E6"] = {"value": after["E6"]["value"]}
        oracle = {"schema_version": "pair.spreadsheet.v0.2" if hard_mode else "pair.spreadsheet.v0.1", "task_type": "repair_formula", "expected_cells": expected_cells, "expected_views": {}, "target_cell": "D2", "dependent_cells": dependent_cells, "reference_trace": trace, "partial_order": {"constraints": ["inspect-before-write", "write-before-finish"]}, "minimal_semantic_calls": len(trace) - 1, "final_checker": "spreadsheet_formula_repair_checker", "forbidden_write_cells": [claim.get("claimed_formula_cell")] if claim.get("claimed_formula_cell") else []}
        worlds.append(build_episode(family_id, wt, "repair_formula", family_seed(seed, index), goal, world, intervention, cone, oracle, difficulty))
    return family_record(family_id, "repair_formula", seed, index, worlds), worlds


def table_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(family_seed(seed, index))
    family_id = f"spreadsheet_{index + 1:06d}"
    hard_mode = bool(cfg.get("hard_mode"))
    base = base_world(index + 1, rng, hard_mode=hard_mode)
    operation = "filter" if index % 2 == 0 else "sort"
    worlds = []
    for wt in WORLD_TYPES:
        world = copy.deepcopy(base)
        intervention = {"type": "none"}
        cone: list[str] = []
        claim: dict[str, Any] = {}
        difficulty = "medium"
        filter_value = "East"
        fallback = "West"
        target_filter = filter_value
        sort_column = "amount"
        descending = True
        if wt == "irrelevant":
            world["cells"]["Z9"] = {"value": "comment", "formula": None, "depends_on": [], "label": "Note"}
            intervention = {"type": "irrelevant_note", "entity": "Z9"}
            difficulty = "easy"
        elif wt == "causal":
            world["tables"]["sales"][1]["region"] = "East"
            world["tables"]["sales"][1]["amount"] += 99
            intervention = {"type": "row_value_changed", "entity": world["tables"]["sales"][1]["id"]}
            cone = ["table.rows", "view.membership", "view.order"]
            difficulty = "hard"
        elif wt == "distractor":
            claim = {"claimed_filter_value": "North", "claimed_sort_column": "product", "must_verify": True}
            intervention = {"type": "distractor_table_instruction"}
            cone = ["instruction.distractor_filter"]
            difficulty = "hard"
        elif wt == "conflict":
            target_filter = fallback
            claim = {"claimed_filter_value": "Central", "must_verify": True}
            intervention = {"type": "empty_filter_conflict", "invalid_filter_value": "Central"}
            cone = ["table.rows", "view.membership"]
            difficulty = "hard"
        view = "analysis_view"
        if operation == "filter":
            expected_ids = filter_ids(world, "region", target_filter)
            requested_filter = target_filter if wt != "conflict" else "Central"
            candidate_filter_values = [requested_filter, fallback, "East"]
            fallback_policy = {
                "condition": "if_filter_value_has_no_matching_rows_after_read_table",
                "candidate_order": candidate_filter_values,
                "selection_rule": "use_first_candidate_with_matching_rows",
                "strict_order": True,
            }
            goal = {"type": "filter_or_sort_table", "operation": "filter", "table": "sales", "view": view, "filter_column": "region", "filter_value": requested_filter, "candidate_filter_values": candidate_filter_values, "fallback_policy": fallback_policy, "user_claim": claim, "verification_required": bool(claim.get("must_verify"))}
            trace = [{"tool": "read_table", "args": {"table": "sales"}}, {"tool": "filter_rows", "args": {"table": "sales", "column": "region", "op": "==", "value": target_filter, "view": view}}, {"tool": "finish", "args": {}}]
        else:
            expected_ids = sort_ids(world, sort_column, descending)
            goal = {"type": "filter_or_sort_table", "operation": "sort", "table": "sales", "view": view, "sort_column": sort_column, "descending": descending, "stale_sort_column": "product", "user_claim": claim, "verification_required": bool(claim.get("must_verify"))}
            trace = [{"tool": "read_table", "args": {"table": "sales"}}, {"tool": "sort_rows", "args": {"table": "sales", "column": sort_column, "descending": descending, "view": view}}, {"tool": "finish", "args": {}}]
        oracle = {"schema_version": "pair.spreadsheet.v0.2" if hard_mode else "pair.spreadsheet.v0.1", "task_type": "filter_or_sort_table", "expected_cells": {}, "expected_views": {view: expected_ids}, "target_view": view, "reference_trace": trace, "partial_order": {"constraints": ["read-table-before-view-write"]}, "minimal_semantic_calls": len(trace) - 1, "final_checker": "spreadsheet_table_view_checker", "forbidden_write_views": []}
        worlds.append(build_episode(family_id, wt, "filter_or_sort_table", family_seed(seed, index), goal, world, intervention, cone, oracle, difficulty))
    return family_record(family_id, "filter_or_sort_table", seed, index, worlds), worlds


def generate_spreadsheet_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = list(cfg.get("task_types") or TASK_TYPES)
    task = tasks[index % len(tasks)]
    if task == "update_cell_with_dependency":
        return update_or_verify_family(index, seed, cfg, task)
    if task == "verify_before_edit":
        return update_or_verify_family(index, seed, cfg, task)
    if task == "repair_formula":
        return repair_formula_family(index, seed, cfg)
    if task == "filter_or_sort_table":
        return table_family(index, seed, cfg)
    raise ValueError(f"unknown spreadsheet task: {task}")
