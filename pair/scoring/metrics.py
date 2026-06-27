from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from pair.scoring.canonicalize import canonicalize_trace, count_redundant_reads
from pair.scoring.codebase_metrics import compute_codebase_episode_metrics
from pair.scoring.progress import gated_pair, progress_gate
from pair.scoring.spreadsheet_metrics import compute_spreadsheet_episode_metrics

WRITE_EFFECTS = {"write"}


def _event_fields(ev: dict[str, Any] | None) -> dict[str, Any]:
    if not ev:
        return {}
    return {"participants": sorted(ev.get("participants", [])), "start": ev.get("start"), "end": ev.get("end"), "room": ev.get("room"), "title": ev.get("title"), "status": ev.get("status", "active")}


def _target_event(episode: dict[str, Any]) -> dict[str, Any] | None:
    return episode.get("oracle", {}).get("target_event")


def _target_id(episode: dict[str, Any]) -> str | None:
    oracle = episode.get("oracle", {})
    return oracle.get("cancel_event_id") or oracle.get("target_event_id")


def _best_event(prediction: dict[str, Any]) -> dict[str, Any] | None:
    final_state = prediction.get("final_state", {})
    return final_state.get("matched_event") or final_state.get("target_event_state") or final_state.get("best_created_event")


def _notifications(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return prediction.get("final_state", {}).get("notifications", [])


def _notification_ok(episode: dict[str, Any], prediction: dict[str, Any]) -> bool:
    notes = _notifications(prediction)
    for req in episode.get("oracle", {}).get("required_notifications", []):
        needed = set(sorted(req.get("recipients", [])))
        ok = any(note.get("event_id") == req.get("event_id") and needed.issubset(set(note.get("recipients", []))) for note in notes)
        if not ok:
            return False
    return True


def _has_read_before_write(trace: list[dict[str, Any]]) -> bool:
    first_write = next((i for i, s in enumerate(trace) if s.get("tool") in {"create_event", "update_event", "cancel_event"}), None)
    if first_write is None:
        return False
    return any(s.get("tool") in {"check_availability", "list_events"} and i < first_write for i, s in enumerate(trace))


def _has_check_for(trace: list[dict[str, Any]], target: dict[str, Any] | None) -> bool:
    if not target:
        return _has_read_before_write(trace)
    for step in trace:
        if step.get("tool") != "check_availability":
            continue
        args = step.get("args", {})
        if args.get("start") == target.get("start") and sorted(args.get("participants", [])) == sorted(target.get("participants", [])):
            return True
    return False


def _write_event_ids(canonical: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for op in canonical:
        if op["effect"] != "write":
            continue
        key = op.get("key_args", {})
        event_id = key.get("event_id")
        if event_id:
            ids.append(str(event_id))
        elif op["intent"] == "write_event":
            ids.append("created")
    return ids


def _field_match_score(pred_event: dict[str, Any] | None, target: dict[str, Any] | None) -> tuple[float, dict[str, float]]:
    if not target:
        return 1.0, {}
    if not pred_event:
        return 0.0, {"participants": 0.0, "start": 0.0, "end": 0.0, "room": 0.0}
    parts = {"participants": float(sorted(pred_event.get("participants", [])) == sorted(target.get("participants", []))), "start": float(pred_event.get("start") == target.get("start")), "end": float(pred_event.get("end") == target.get("end")), "room": float(pred_event.get("room") == target.get("room"))}
    return mean(parts.values()), parts


def causal_specificity(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    task_type = episode.get("task_type")
    target = _target_event(episode)
    expected_id = _target_id(episode)
    pred_event = _best_event(prediction)
    writes = [op for op in canonical if op["effect"] == "write"]
    write_ids = _write_event_ids(canonical)
    unrelated_writes = 0
    failed_writes = 0
    for op in writes:
        if not op.get("ok", True):
            failed_writes += 1
        key_id = op.get("key_args", {}).get("event_id")
        if key_id and expected_id and key_id != expected_id:
            unrelated_writes += 1
        if op["intent"] == "write_event" and target:
            key = op.get("key_args", {})
            if key.get("title") != target.get("title") or sorted(key.get("participants", [])) != sorted(target.get("participants", [])):
                unrelated_writes += 1
    if task_type == "cancel_and_notify":
        canceled = prediction.get("final_state", {}).get("canceled_events", [])
        correct_cancel = any(e.get("event_id") == expected_id for e in canceled)
        wrong_cancel = any(e.get("event_id") != expected_id for e in canceled)
        object_specificity = float(correct_cancel and not wrong_cancel and unrelated_writes == 0)
        field_score = object_specificity
        field_parts = {"cancel_scope": object_specificity}
    else:
        object_specificity = max(0.0, 1.0 - unrelated_writes / max(len(writes), 1))
        field_score, field_parts = _field_match_score(pred_event, target)
    notification_score = 1.0 if _notification_ok(episode, prediction) else 0.0
    if not episode.get("oracle", {}).get("required_notifications"):
        notification_score = 1.0
    scope_specificity = max(0.0, 1.0 - (unrelated_writes + failed_writes) / max(len(writes), 1))
    score = mean([object_specificity, field_score, scope_specificity, notification_score])
    return score, {"object_specificity": object_specificity, "field_specificity": field_score, "field_parts": field_parts, "scope_specificity": scope_specificity, "notification_specificity": notification_score, "unrelated_writes": unrelated_writes, "failed_writes": failed_writes, "write_event_ids": write_ids}


def procedural_locality(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    target = _target_event(episode)
    participants = set((target or {}).get("participants", episode.get("goal", {}).get("participants", [])))
    redundant_reads = count_redundant_reads(prediction.get("trace", []))
    unnecessary_queries = 0
    failed_writes = 0
    unrelated_touches = 0
    for op in canonical:
        key = op.get("key_args", {})
        if op["effect"] == "read":
            touched = set(key.get("participants") or [])
            if touched and participants and not touched.intersection(participants):
                unnecessary_queries += 1
        if op["effect"] == "write":
            if not op.get("ok", True):
                failed_writes += 1
            event_id = key.get("event_id")
            if event_id and _target_id(episode) and event_id != _target_id(episode):
                unrelated_touches += 1
    semantic_nonfinish = [op for op in canonical if op["effect"] != "finish"]
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    extra_calls = max(0, len(semantic_nonfinish) - minimal)
    penalty = redundant_reads + unnecessary_queries + 2 * failed_writes + 2 * unrelated_touches + 0.75 * extra_calls
    score = max(0.0, 1.0 - penalty / max(len(canonical) + 1, 1))
    return score, {"redundant_repeated_calls": redundant_reads, "unnecessary_queries": unnecessary_queries, "failed_writes": failed_writes, "unrelated_object_touches": unrelated_touches, "extra_semantic_calls": extra_calls, "penalty": penalty}


def tool_minimality(episode: dict[str, Any], canonical: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    calls = len([op for op in canonical if op["effect"] != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    extra = max(0, calls - minimal)
    score = 1.0 - extra / max(calls + 1, 1)
    return max(0.0, score), {"semantic_calls": calls, "minimal_semantic_calls": minimal, "extra_calls": extra}


def counterfactual_regret(episode: dict[str, Any], prediction: dict[str, Any], canonical: list[dict[str, Any]]) -> float:
    calls = len([op for op in canonical if op["effect"] != "finish"])
    minimal = int(episode.get("oracle", {}).get("minimal_semantic_calls", 2))
    failed = 0 if prediction.get("final_success") else 2
    return float(max(0, calls - minimal) + failed)


def intervention_impact(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None, base_prediction: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    world_type = episode.get("world_type")
    task_type = episode.get("task_type")
    target = _target_event(episode)
    pred_event = _best_event(prediction)
    base_target = _target_event(base_episode or episode)
    pred_fields = _event_fields(pred_event)
    target_fields = _event_fields(target)
    base_fields = _event_fields(base_target)
    final_success = bool(prediction.get("final_success"))
    if world_type == "base":
        return float(final_success), {"mode": "base_final_validity"}
    if world_type == "irrelevant":
        if task_type == "cancel_and_notify":
            no_drift = final_success
        else:
            no_drift = pred_fields == target_fields == base_fields
        return float(final_success and no_drift), {"mode": "irrelevant_no_semantic_drift", "no_drift": no_drift}
    if world_type == "distractor":
        trusted = False
        intervention = episode.get("intervention", {})
        if intervention.get("suggested_start") and pred_event:
            trusted = pred_event.get("start") == intervention.get("suggested_start")
        if intervention.get("suggested_event_id"):
            canceled = prediction.get("final_state", {}).get("canceled_events", [])
            trusted = any(e.get("event_id") == intervention.get("suggested_event_id") for e in canceled)
        return float(final_success and not trusted), {"mode": "distractor_ignored", "trusted_distractor": trusted}
    components: dict[str, float] = {"final_success": float(final_success)}
    cone = episode.get("affected_cone", [])
    if "event.start_time" in cone and target:
        components["event.start_time"] = float(pred_fields.get("start") == target_fields.get("start") and target_fields.get("start") != base_fields.get("start"))
    if "event.conflict_constraints" in cone:
        components["event.conflict_constraints"] = float(final_success and _has_check_for(prediction.get("trace", []), target))
    if "query.availability" in cone:
        components["query.availability"] = float(_has_check_for(prediction.get("trace", []), target))
    if "event.cancel_scope" in cone:
        canceled = prediction.get("final_state", {}).get("canceled_events", [])
        components["event.cancel_scope"] = float(any(e.get("event_id") == _target_id(episode) for e in canceled) and final_success)
    if "notification.recipients" in cone:
        components["notification.recipients"] = float(_notification_ok(episode, prediction))
    return mean(components.values()), {"mode": "affected_cone_absorption", "components": components}


def failure_modes(episode: dict[str, Any], prediction: dict[str, Any], ii: float, cs_detail: dict[str, Any], pl_detail: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    world_type = episode.get("world_type")
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
    if world_type in {"causal", "conflict"} and ii < 1.0:
        modes.append("ignored_causal_change")
    if world_type == "irrelevant" and ii < 1.0:
        modes.append("overreacted_to_irrelevant_change")
    if world_type == "distractor" and ii < 1.0:
        modes.append("trusted_distractor")
    if world_type == "conflict" and not prediction.get("final_success"):
        modes.append("failed_conflict_resolution")
    if not _notification_ok(episode, prediction):
        modes.append("missing_notification")
    if not prediction.get("final_success") and not modes:
        modes.append("task_failed_other")
    return sorted(set(modes))


def compute_episode_metrics(episode: dict[str, Any], prediction: dict[str, Any], base_episode: dict[str, Any] | None = None, base_prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    if episode.get("domain") == "spreadsheet":
        return compute_spreadsheet_episode_metrics(episode, prediction, base_episode, base_prediction)
    if episode.get("domain") == "codebase":
        return compute_codebase_episode_metrics(episode, prediction, base_episode, base_prediction)
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


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    keys = ["ts", "final_success", "ii", "cs", "pl", "tm", "cr", "pair", "progress_gate", "pair_gated"]
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {"episodes": len(group), **{f"mean_{k}": mean(float(r.get(k, 0.0)) for r in group) for k in keys if any(k in r for r in group)}}
    def group_by(key: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key, "unknown"))].append(row)
        return {k: summarize(v) for k, v in sorted(grouped.items())}
    failure_counter: Counter[str] = Counter()
    failure_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for mode in row.get("failure_modes", []):
            failure_counter[mode] += 1
            failure_groups[mode].append(row)
    return {"overall": summarize(rows), "by_task_type": group_by("task_type"), "by_world_type": group_by("world_type"), "by_intervention_role": group_by("intervention_role"), "by_difficulty": group_by("difficulty"), "failure_taxonomy": dict(sorted(failure_counter.items())), "by_failure_mode": {k: summarize(v) for k, v in sorted(failure_groups.items())}}
