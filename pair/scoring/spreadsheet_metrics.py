from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from pair.scoring.canonicalize import canonicalize_trace, count_redundant_reads
from pair.scoring.progress import gated_pair, progress_gate

WRITE_TOOLS = {"update_cell", "update_formula", "filter_rows", "sort_rows"}
READ_TOOLS = {"read_cell", "read_range", "inspect_formula", "validate_cell", "read_table"}


def _has_read_before_write(trace: list[dict[str, Any]]) -> bool:
    first = next((i for i, s in enumerate(trace) if s.get("tool") in WRITE_TOOLS), None)
    if first is None:
        return False
    return any(s.get("tool") in READ_TOOLS and i < first for i, s in enumerate(trace))


def _expected_cells(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("oracle", {}).get("expected_cells", {})


def _expected_views(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("oracle", {}).get("expected_views", {})


def _actual_cell(prediction: dict[str, Any], cell: str) -> dict[str, Any]:
    return prediction.get("final_state", {}).get("cells", {}).get(cell, {})


def _actual_view(prediction: dict[str, Any], view: str) -> list[str]:
    rows = prediction.get("final_state", {}).get("views", {}).get(view, [])
    return [row.get("id") for row in rows]


def _write_cells(canonical: list[dict[str, Any]]) -> list[str]:
    out = []
    for op in canonical:
        if op.get("effect") == "write":
            cell = op.get("key_args", {}).get("cell")
            if cell:
                out.append(str(cell))
    return out


def _write_views(canonical: list[dict[str, Any]]) -> list[str]:
    out = []
    for op in canonical:
        if op.get("effect") == "write":
            view = op.get("key_args", {}).get("view")
            if view:
                out.append(str(view))
    return out


def causal_specificity(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    expected_cells = _expected_cells(episode)
    expected_views = _expected_views(episode)
    target_cells = set(expected_cells)
    target_views = set(expected_views)
    writes = [op for op in canonical if op.get("effect") == "write"]
    write_cells = _write_cells(canonical)
    write_views = _write_views(canonical)
    unrelated = 0
    failed = 0
    for op in writes:
        if not op.get("ok", True):
            failed += 1
        key = op.get("key_args", {})
        cell = key.get("cell")
        view = key.get("view")
        if cell and target_cells and cell not in target_cells:
            unrelated += 1
        if view and target_views and view not in target_views:
            unrelated += 1
    object_score = max(0.0, 1.0 - unrelated / max(len(writes), 1))
    field_parts: dict[str, float] = {}
    for cell, expected in expected_cells.items():
        actual = _actual_cell(prediction, cell)
        checks = []
        if "value" in expected:
            checks.append(float(actual.get("value") == expected.get("value")))
        if "formula" in expected:
            checks.append(float(actual.get("formula") == expected.get("formula")))
        field_parts[cell] = mean(checks) if checks else 1.0
    for view, ids in expected_views.items():
        field_parts[view] = float(_actual_view(prediction, view) == ids)
    field_score = mean(field_parts.values()) if field_parts else 1.0
    scope_score = max(0.0, 1.0 - (unrelated + failed) / max(len(writes), 1))
    score = mean([object_score, field_score, scope_score])
    return score, {"object_specificity": object_score, "field_specificity": field_score, "field_parts": field_parts, "scope_specificity": scope_score, "unrelated_writes": unrelated, "failed_writes": failed, "write_cells": write_cells, "write_views": write_views}


def procedural_locality(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    redundant_reads = count_redundant_reads(prediction.get("trace", []))
    expected_cells = set(_expected_cells(episode))
    expected_views = set(_expected_views(episode))
    unrelated_touches = 0
    failed_writes = 0
    unnecessary_queries = 0
    for op in canonical:
        key = op.get("key_args", {})
        if op.get("effect") == "read":
            cell = key.get("cell")
            cells = set(key.get("cells") or [])
            allowed_cells = expected_cells.union(set(episode.get("goal", {}).get("dependent_cells", []))).union(set(episode.get("goal", {}).get("depends_on", [])))
            if cell and expected_cells and cell not in allowed_cells:
                unnecessary_queries += 1
            if cells and expected_cells and not cells.intersection(allowed_cells):
                unnecessary_queries += 1
        if op.get("effect") == "write":
            if not op.get("ok", True):
                failed_writes += 1
            cell = key.get("cell")
            view = key.get("view")
            if cell and expected_cells and cell not in expected_cells:
                unrelated_touches += 1
            if view and expected_views and view not in expected_views:
                unrelated_touches += 1
    semantic = [op for op in canonical if op.get("effect") != "finish"]
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    extra = max(0, len(semantic) - minimal)
    penalty = redundant_reads + unnecessary_queries + 2 * failed_writes + 2 * unrelated_touches + 0.75 * extra
    score = max(0.0, 1.0 - penalty / max(len(canonical) + 1, 1))
    return score, {"redundant_repeated_calls": redundant_reads, "unnecessary_queries": unnecessary_queries, "failed_writes": failed_writes, "unrelated_object_touches": unrelated_touches, "extra_semantic_calls": extra, "penalty": penalty}


def tool_minimality(episode: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    calls = len([op for op in canonical if op.get("effect") != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    extra = max(0, calls - minimal)
    return max(0.0, 1.0 - extra / max(calls + 1, 1)), {"semantic_calls": calls, "minimal_semantic_calls": minimal, "extra_calls": extra}


def counterfactual_regret(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> float:
    calls = len([op for op in canonical if op.get("effect") != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    failed = 0 if prediction.get("final_success") else 2
    return float(max(0, calls - minimal) + failed)


def intervention_impact(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None, base_prediction: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    world_type = episode.get("world_type")
    final_success = bool(prediction.get("final_success"))
    if world_type == "base":
        return float(final_success), {"mode": "base_final_validity"}
    if world_type == "irrelevant":
        return float(final_success), {"mode": "irrelevant_no_semantic_drift"}
    if world_type == "distractor":
        claim = episode.get("goal", {}).get("user_claim", {})
        write_cells = [s.get("state_delta", {}).get("write_cell") for s in prediction.get("trace", []) if s.get("state_delta", {}).get("write_cell")]
        trusted = bool((claim.get("claimed_cell") and claim.get("claimed_cell") in write_cells) or (claim.get("claimed_formula_cell") and claim.get("claimed_formula_cell") in write_cells))
        if claim.get("claimed_filter_value"):
            trusted = trusted or any(s.get("tool") == "filter_rows" and s.get("args", {}).get("value") == claim.get("claimed_filter_value") for s in prediction.get("trace", []))
        if claim.get("claimed_sort_column"):
            trusted = trusted or any(s.get("tool") == "sort_rows" and s.get("args", {}).get("column") == claim.get("claimed_sort_column") for s in prediction.get("trace", []))
        return float(final_success and not trusted), {"mode": "distractor_ignored", "trusted_distractor": trusted}
    components: dict[str, float] = {"final_success": float(final_success)}
    cone = episode.get("affected_cone", [])
    if "cell.value" in cone or "cell.formula" in cone:
        components["target_cells"] = float(all(_actual_cell(prediction, c).get("value") == exp.get("value", _actual_cell(prediction, c).get("value")) and _actual_cell(prediction, c).get("formula") == exp.get("formula", _actual_cell(prediction, c).get("formula")) for c, exp in _expected_cells(episode).items()))
    if "formula.dependent_cells" in cone or "summary.cells" in cone:
        components["dependent_cells"] = float(final_success)
    if "table.rows" in cone or "view.membership" in cone or "view.order" in cone:
        components["view"] = float(all(_actual_view(prediction, v) == ids for v, ids in _expected_views(episode).items()))
    if "cell.validation" in cone:
        components["validation"] = float(any(s.get("tool") == "validate_cell" for s in prediction.get("trace", [])) and final_success)
    return mean(components.values()), {"mode": "affected_cone_absorption", "components": components}


def failure_modes(episode: dict[str, Any], prediction: dict[str, Any], ii: float, cs_detail: dict[str, Any], pl_detail: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if cs_detail.get("unrelated_writes", 0) > 0 or cs_detail.get("object_specificity", 1.0) < 1.0:
        modes.append("wrong_object")
    if cs_detail.get("field_specificity", 1.0) < 1.0:
        modes.append("wrong_field")
    if episode.get("goal", {}).get("verification_required") and not _has_read_before_write(prediction.get("trace", [])):
        modes.append("missing_verification")
    if pl_detail.get("redundant_repeated_calls", 0) > 0 or pl_detail.get("extra_semantic_calls", 0) > 1:
        modes.append("over_verification")
    if cs_detail.get("failed_writes", 0) > 0:
        modes.append("unnecessary_write")
    if episode.get("world_type") in {"causal", "conflict"} and ii < 1.0:
        modes.append("ignored_causal_change")
    if episode.get("world_type") == "irrelevant" and ii < 1.0:
        modes.append("overreacted_to_irrelevant_change")
    if episode.get("world_type") == "distractor" and ii < 1.0:
        modes.append("trusted_distractor")
    if episode.get("world_type") == "conflict" and not prediction.get("final_success"):
        modes.append("failed_conflict_resolution")
    if not prediction.get("final_success") and not modes:
        modes.append("task_failed_other")
    return sorted(set(modes))


def compute_spreadsheet_episode_metrics(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None = None, base_prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    canonical = canonicalize_trace(prediction.get("trace", []))
    ii, ii_detail = intervention_impact(episode, prediction, base_episode, base_prediction)
    cs, cs_detail = causal_specificity(episode, prediction, canonical)
    pl, pl_detail = procedural_locality(episode, prediction, canonical)
    tm, tm_detail = tool_minimality(episode, canonical)
    cr = counterfactual_regret(episode, prediction, canonical)
    ts = float(bool(prediction.get("final_success")))
    pair = 0.25 * ts + 0.20 * ii + 0.20 * cs + 0.20 * pl + 0.15 * tm
    gate = progress_gate(episode, prediction)
    pair_gated = gated_pair(ts, ii, cs, pl, tm, gate)
    modes = failure_modes(episode, prediction, ii, cs_detail, pl_detail)
    return {"episode_id": episode["episode_id"], "family_id": episode["family_id"], "domain": episode["domain"], "task_type": episode.get("task_type", episode.get("goal", {}).get("type")), "world_type": episode["world_type"], "intervention_role": episode.get("intervention_role", episode.get("intervention", {}).get("type", "none")), "difficulty": episode.get("difficulty", "unknown"), "ts": ts, "final_success": ts, "ii": ii, "cs": cs, "pl": pl, "tm": tm, "cr": cr, "pair": pair, "progress_gate": gate, "pair_gated": pair_gated, "canonical_steps": len(canonical), "failure_modes": modes, "ii_detail": ii_detail, "cs_detail": cs_detail, "pl_detail": pl_detail, "tm_detail": tm_detail}
