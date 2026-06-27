from __future__ import annotations

from statistics import mean
from typing import Any

from pair.scoring.canonicalize import canonicalize_step, canonicalize_trace
from pair.scoring.progress import gated_pair, progress_gate

READ_TOOLS = {"read_file", "search_code", "run_tests"}
WRITE_TOOLS = {"edit_file"}


def _has_read_before_write(trace: list[dict[str, Any]]) -> bool:
    first_write = next((i for i, step in enumerate(trace) if step.get("tool") in WRITE_TOOLS), None)
    if first_write is None:
        return False
    return any(step.get("tool") in READ_TOOLS and i < first_write for i, step in enumerate(trace))


def _has_test_run(trace: list[dict[str, Any]]) -> bool:
    return any(step.get("tool") == "run_tests" for step in trace)


def _write_files(canonical: list[dict[str, Any]]) -> list[str]:
    out = []
    for op in canonical:
        if op.get("effect") == "write":
            path = op.get("key_args", {}).get("path")
            if path:
                out.append(str(path))
    return out


def _redundant_code_reads(trace: list[dict[str, Any]]) -> int:
                                                                                 
    seen: set[str] = set()
    redundant = 0
    for raw in trace:
        step = canonicalize_step(raw)
        if step.get("tool") not in {"read_file", "search_code"}:
            continue
        sig = f"{step.get('tool')}:{step.get('key_args')}"
        if sig in seen:
            redundant += 1
        seen.add(sig)
    return redundant


def causal_specificity(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    expected_files = episode.get("oracle", {}).get("expected_files", {})
    target_files = set(episode.get("oracle", {}).get("target_files", list(expected_files)))
    forbidden_files = set(episode.get("oracle", {}).get("forbidden_files", []))
    writes = [op for op in canonical if op.get("effect") == "write"]
    write_files = _write_files(canonical)
    unrelated = 0
    failed = 0
    for op in writes:
        path = op.get("key_args", {}).get("path")
        if not op.get("ok", True):
            failed += 1
        if path and target_files and path not in target_files:
            unrelated += 1
        if path and path in forbidden_files:
            unrelated += 1
    object_score = max(0.0, 1.0 - unrelated / max(len(writes), 1))
    field_parts = {}
    files = prediction.get("final_state", {}).get("files", {})
    for path, expected in expected_files.items():
        field_parts[path] = float(files.get(path) == expected)
    field_score = mean(field_parts.values()) if field_parts else 1.0
    scope_score = max(0.0, 1.0 - (unrelated + failed) / max(len(writes), 1))
    score = mean([object_score, field_score, scope_score])
    return score, {
        "object_specificity": object_score,
        "field_specificity": field_score,
        "field_parts": field_parts,
        "scope_specificity": scope_score,
        "unrelated_writes": unrelated,
        "failed_writes": failed,
        "write_files": write_files,
    }


def procedural_locality(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    target_files = set(episode.get("oracle", {}).get("target_files", []))
    forbidden_files = set(episode.get("oracle", {}).get("forbidden_files", []))
    redundant_reads = _redundant_code_reads(prediction.get("trace", []))
    unnecessary_queries = 0
    failed_writes = 0
    unrelated_touches = 0
    for op in canonical:
        key = op.get("key_args", {})
        path = key.get("path")
        if op.get("effect") == "read":
            query = str(key.get("query", ""))
            if path and target_files and path not in target_files and path not in {f"tests/test_{p.split('/')[-1].split('.')[0]}.py" for p in target_files}:
                unnecessary_queries += 1
            if query and target_files and not any(query in f for f in target_files):
                if query.lower() in {"readme", "docs", "unrelated"}:
                    unnecessary_queries += 1
        if op.get("effect") == "write":
            if not op.get("ok", True):
                failed_writes += 1
            if path and (path not in target_files or path in forbidden_files):
                unrelated_touches += 1
    semantic = [op for op in canonical if op.get("effect") != "finish"]
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 5))
    extra = max(0, len(semantic) - minimal)
    penalty = redundant_reads + unnecessary_queries + 2 * failed_writes + 2 * unrelated_touches + 0.75 * extra
    score = max(0.0, 1.0 - penalty / max(len(canonical) + 1, 1))
    return score, {
        "redundant_repeated_calls": redundant_reads,
        "unnecessary_queries": unnecessary_queries,
        "failed_writes": failed_writes,
        "unrelated_object_touches": unrelated_touches,
        "extra_semantic_calls": extra,
        "penalty": penalty,
    }


def tool_minimality(episode: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    calls = len([op for op in canonical if op.get("effect") != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 5))
    extra = max(0, calls - minimal)
    return max(0.0, 1.0 - extra / max(calls + 1, 1)), {"semantic_calls": calls, "minimal_semantic_calls": minimal, "extra_calls": extra}


def counterfactual_regret(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> float:
    calls = len([op for op in canonical if op.get("effect") != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 5))
    failed = 0 if prediction.get("final_success") else 2
    return float(max(0, calls - minimal) + failed)


def intervention_impact(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None, base_prediction: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    world_type = episode.get("world_type")
    final_success = bool(prediction.get("final_success"))
    if world_type == "base":
        return float(final_success), {"mode": "base_final_validity"}
    if world_type == "irrelevant":
        touched = set(prediction.get("final_state", {}).get("touched_files", []))
        irrelevant = {f for f in touched if f.startswith("docs/") or f == "README.md"}
        return float(final_success and not irrelevant), {"mode": "irrelevant_no_file_drift", "irrelevant_touches": sorted(irrelevant)}
    if world_type == "distractor":
        claimed = episode.get("oracle", {}).get("claimed_file")
        writes = [s.get("state_delta", {}).get("write_file") for s in prediction.get("trace", [])]
        trusted = bool(claimed and claimed in writes)
        return float(final_success and not trusted), {"mode": "distractor_ignored", "trusted_distractor": trusted}
    components = {"final_success": float(final_success)}
    cone = set(episode.get("affected_cone", []))
    if "tests.expected_behavior" in cone or "test.result" in cone:
        components["run_tests"] = float(_has_test_run(prediction.get("trace", [])) and final_success)
    if "code.target_function" in cone:
        components["target_file"] = float(all(path in prediction.get("final_state", {}).get("touched_files", []) for path in episode.get("oracle", {}).get("target_files", [])) and final_success)
    return mean(components.values()), {"mode": "affected_cone_absorption", "components": components}


def failure_modes(episode: dict[str, Any], prediction: dict[str, Any], ii: float, cs_detail: dict[str, Any], pl_detail: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if cs_detail.get("unrelated_writes", 0) > 0 or cs_detail.get("object_specificity", 1.0) < 1.0:
        modes.append("wrong_object")
    if cs_detail.get("field_specificity", 1.0) < 1.0:
        modes.append("wrong_field")
    if episode.get("goal", {}).get("verification_required") and not _has_read_before_write(prediction.get("trace", [])):
        modes.append("missing_verification")
    if not _has_test_run(prediction.get("trace", [])):
        modes.append("missing_test")
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
    if not prediction.get("final_success") and not modes:
        modes.append("task_failed_other")
    return sorted(set(modes))


def compute_codebase_episode_metrics(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None = None, base_prediction: dict[str, Any] | None = None) -> dict[str, Any]:
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
    return {
        "episode_id": episode["episode_id"],
        "family_id": episode["family_id"],
        "domain": episode["domain"],
        "task_type": episode.get("task_type", episode.get("goal", {}).get("type")),
        "world_type": episode["world_type"],
        "intervention_role": episode.get("intervention_role", episode.get("intervention", {}).get("type", "none")),
        "difficulty": episode.get("difficulty", "unknown"),
        "ts": ts,
        "final_success": ts,
        "ii": ii,
        "cs": cs,
        "pl": pl,
        "tm": tm,
        "cr": cr,
        "pair": pair,
        "progress_gate": gate,
        "pair_gated": pair_gated,
        "canonical_steps": len(canonical),
        "failure_modes": modes,
        "ii_detail": ii_detail,
        "cs_detail": cs_detail,
        "pl_detail": pl_detail,
        "tm_detail": tm_detail,
    }
