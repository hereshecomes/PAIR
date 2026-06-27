from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta
from typing import Any

from pair.utils.io import stable_json_hash
from pair.utils.seed import family_seed

ISO = "%Y-%m-%dT%H:%M:%S"
PEOPLE = ["Alice", "Bob", "Chen", "Diego", "Eve", "Fatima", "Grace", "Hiro", "Iris", "Jon"]
ROOMS = ["Room-A", "Room-B", "Room-C"]
COLORS = ["blue", "green", "orange", "purple", "gray"]
WORLD_TYPES = ["base", "irrelevant", "causal", "distractor", "conflict"]
TASK_TYPES = ["schedule", "reschedule", "cancel_and_notify", "verify_before_act"]


def iso_at(date: str, hour: int) -> str:
    return f"{date}T{hour:02d}:00:00"


def add_minutes(value: str, minutes: int) -> str:
    return (datetime.strptime(value, ISO) + timedelta(minutes=minutes)).strftime(ISO)


def make_event(event_id: str, title: str, participants: list[str], start: str, duration_minutes: int, room: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "title": title,
        "participants": sorted(participants),
        "start": start,
        "end": add_minutes(start, duration_minutes),
        "room": room,
        "status": status,
        "metadata": metadata or {},
    }


def target_event(event_id: str, title: str, participants: list[str], start: str, duration_minutes: int, room: str) -> dict[str, Any]:
    return make_event(event_id, title, participants, start, duration_minutes, room, {"kind": "target"})


def check_action(target: dict[str, Any]) -> dict[str, Any]:
    return {"tool": "check_availability", "args": {"participants": target["participants"], "start": target["start"], "end": target["end"], "room": target["room"]}}


def reference_schedule_trace(goal: dict[str, Any], target: dict[str, Any], include_blocked_probe: bool) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    if include_blocked_probe and goal["preferred_start"] != target["start"]:
        trace.append({"tool": "check_availability", "args": {"participants": target["participants"], "start": goal["preferred_start"], "end": add_minutes(goal["preferred_start"], goal["duration_minutes"]), "room": target["room"]}})
    trace.extend([
        check_action(target),
        {"tool": "create_event", "args": {"title": target["title"], "participants": target["participants"], "start": target["start"], "end": target["end"], "room": target["room"]}},
        {"tool": "finish", "args": {}},
    ])
    return trace


def reference_reschedule_trace(goal: dict[str, Any], target: dict[str, Any], include_blocked_probe: bool) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = [{"tool": "list_events", "args": {"title": goal["title"], "participants": goal["participants"]}}]
    if include_blocked_probe and goal["preferred_start"] != target["start"]:
        trace.append({"tool": "check_availability", "args": {"participants": target["participants"], "start": goal["preferred_start"], "end": add_minutes(goal["preferred_start"], goal["duration_minutes"]), "room": target["room"]}})
    trace.extend([
        check_action(target),
        {"tool": "update_event", "args": {"event_id": target["event_id"], "start": target["start"], "end": target["end"], "room": target["room"]}},
        {"tool": "send_notification", "args": {"event_id": target["event_id"], "recipients": target["participants"], "message": "Meeting rescheduled"}},
        {"tool": "finish", "args": {}},
    ])
    return trace


def reference_cancel_trace(goal: dict[str, Any], cancel_event_id: str, recipients: list[str]) -> list[dict[str, Any]]:
    return [
        {"tool": "list_events", "args": {"title": goal["title"], "participants": goal["participants"]}},
        {"tool": "cancel_event", "args": {"event_id": cancel_event_id, "reason": "Requested cancellation"}},
        {"tool": "send_notification", "args": {"event_id": cancel_event_id, "recipients": recipients, "message": "Meeting canceled"}},
        {"tool": "finish", "args": {}},
    ]


def build_episode(
    family_id: str,
    world_type: str,
    task_type: str,
    seed: int,
    date: str,
    goal: dict[str, Any],
    events: list[dict[str, Any]],
    intervention: dict[str, Any],
    affected_cone: list[str],
    target: dict[str, Any] | None,
    reference_trace: list[dict[str, Any]],
    minimal_semantic_calls: int,
    difficulty: str,
    cancel_event_id: str | None = None,
    required_notifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    episode_id = f"{family_id}_{world_type}"
    oracle = {
        "task_type": task_type,
        "target_event": target,
        "target_event_id": target["event_id"] if target else cancel_event_id,
        "cancel_event_id": cancel_event_id,
        "required_notifications": required_notifications or [],
        "reference_trace": reference_trace,
        "partial_order": {"constraints": ["verify-before-write", "write-before-notify", "notify-before-finish"]},
        "minimal_semantic_calls": minimal_semantic_calls,
        "final_checker": f"calendar_{task_type}_checker",
    }
    return {
        "schema_version": "pair.calendar.v0.2",
        "domain": "calendar",
        "family_id": family_id,
        "episode_id": episode_id,
        "task_type": task_type,
        "world_type": world_type,
        "difficulty": difficulty,
        "intervention_role": intervention.get("type", "none"),
        "seed": seed,
        "goal": copy.deepcopy(goal),
        "world": {"date": date, "people": PEOPLE, "rooms": ROOMS, "events": copy.deepcopy(events)},
        "intervention": intervention,
        "affected_cone": affected_cone,
        "oracle": oracle,
        "state_hash": stable_json_hash({"events": events, "goal": goal, "intervention": intervention, "task_type": task_type}),
    }


def base_context(index: int, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    date = str(cfg.get("start_date", "2026-07-01"))
    duration = int(cfg.get("duration_minutes", 60))
    start_hour = int(cfg.get("workday_start_hour", 9))
    end_hour = int(cfg.get("workday_end_hour", 17))
    rng = random.Random(family_seed(seed, index))
    participants = sorted(rng.sample(PEOPLE, 2))
    outsider = next(p for p in PEOPLE if p not in participants)
    room = rng.choice(ROOMS)
    other_room = rng.choice([r for r in ROOMS if r != room] or ROOMS)
    slots = [iso_at(date, h) for h in range(start_hour, end_hour)]
    old_start = slots[0]
    base_start = rng.choice(slots[1:5])
    alt_start = next(s for s in slots[1:7] if s != base_start)
    distractor_start = next(s for s in reversed(slots[1:7]) if s not in {base_start, alt_start})
    unrelated = make_event(f"existing_{index + 1:06d}_unrelated", "Unrelated 1:1", [outsider], rng.choice([s for s in slots[0:6] if s != base_start]), duration, other_room, {"color": rng.choice(COLORS), "note": "metadata only"})
    return {"date": date, "duration": duration, "rng": rng, "participants": participants, "outsider": outsider, "room": room, "other_room": other_room, "slots": slots, "old_start": old_start, "base_start": base_start, "alt_start": alt_start, "distractor_start": distractor_start, "unrelated": unrelated}


def common_schedule_goal(title: str, participants: list[str], room: str, duration: int, preferred_start: str, candidates: list[str], task_type: str, stale_start: str, user_claim: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": task_type,
        "title": title,
        "participants": sorted(participants),
        "duration_minutes": duration,
        "room": room,
        "preferred_start": preferred_start,
        "candidate_starts": candidates,
        "stale_start": stale_start,
        "window_start": candidates[0],
        "window_end": add_minutes(candidates[-1], duration),
        "user_claim": user_claim or {},
        "verification_required": task_type == "verify_before_act" or bool((user_claim or {}).get("must_verify")),
    }


def schedule_family(index: int, seed: int, cfg: dict[str, Any], task_type: str = "schedule") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ctx = base_context(index, seed, cfg)
    family_id = f"calendar_{index + 1:06d}"
    title = f"Planning Sync {index + 1:04d}"
    p, room, duration = ctx["participants"], ctx["room"], ctx["duration"]
    base, alt, dist = ctx["base_start"], ctx["alt_start"], ctx["distractor_start"]
    candidates = [base] + [s for s in ctx["slots"][1:7] if s != base]
    base_events = [ctx["unrelated"]]
    episodes: list[dict[str, Any]] = []
    specs = [
        ("base", base_events, base, base, {"type": "none"}, [], {}, "easy"),
        ("irrelevant", [dict(ctx["unrelated"], metadata={"color": "teal", "note": "renamed note"})], base, base, {"type": "metadata_change", "entity": ctx["unrelated"]["event_id"], "field": "metadata.color"}, [], {}, "easy"),
        ("causal", base_events + [make_event(f"existing_{index + 1:06d}_causal_block", "Focus Block", [p[0]], base, duration, ctx["other_room"], {"priority": "high"})], base, alt, {"type": "attendee_unavailable", "entity": p[0], "time_window": [base, add_minutes(base, duration)]}, ["event.start_time", "event.conflict_constraints", "query.availability"], {}, "hard"),
        ("distractor", base_events + [make_event(f"existing_{index + 1:06d}_distractor_block", "Private Hold", [p[1]], dist, duration, room, {"priority": "private"})], dist, base, {"type": "distractor_slot", "suggested_start": dist}, ["instruction.distractor_slot"], {"claimed_available_start": dist, "must_verify": task_type == "verify_before_act"}, "medium"),
        ("conflict", base_events + [make_event(f"existing_{index + 1:06d}_conflict_block", "Existing Customer Call", [p[0]], base, duration, room, {"priority": "high"})], base, alt, {"type": "conflict_user_claim", "entity": p[0], "claimed_available": base}, ["event.start_time", "event.conflict_constraints", "query.availability"], {"claimed_available_start": base, "must_verify": True}, "hard"),
    ]
    for world_type, events, preferred, target_start, intervention, cone, claim, difficulty in specs:
        goal = common_schedule_goal(title, p, room, duration, preferred, [preferred] + [s for s in candidates if s != preferred], task_type, base, claim)
        target = target_event("created_target", title, p, target_start, duration, room)
        include_probe = preferred != target_start or goal["verification_required"]
        trace = reference_schedule_trace(goal, target, include_probe)
        minimal = len([a for a in trace if a["tool"] != "finish"])
        episodes.append(build_episode(family_id, world_type, task_type, family_seed(seed, index), ctx["date"], goal, events, intervention, cone, target, trace, minimal, difficulty))
    return family_record(family_id, task_type, seed, index, episodes), episodes


def reschedule_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ctx = base_context(index, seed, cfg)
    family_id = f"calendar_{index + 1:06d}"
    title = f"Roadmap Review {index + 1:04d}"
    p, room, duration = ctx["participants"], ctx["room"], ctx["duration"]
    old, base, alt, dist = ctx["old_start"], ctx["base_start"], ctx["alt_start"], ctx["distractor_start"]
    original_id = f"resched_{index + 1:06d}_target"
    original = make_event(original_id, title, p, old, duration, room, {"priority": "normal"})
    similar = make_event(f"resched_{index + 1:06d}_similar", title + " Prep", p, old, duration, ctx["other_room"], {"priority": "low"})
    base_events = [ctx["unrelated"], original, similar]
    candidates = [base] + [s for s in ctx["slots"][1:7] if s != base]
    episodes: list[dict[str, Any]] = []
    specs = [
        ("base", base_events, base, base, {"type": "none"}, [], {}, "medium"),
        ("irrelevant", [dict(e, metadata={**e.get("metadata", {}), "color": "teal"}) if e["event_id"] == ctx["unrelated"]["event_id"] else e for e in base_events], base, base, {"type": "metadata_change", "entity": ctx["unrelated"]["event_id"], "field": "metadata.color"}, [], {}, "medium"),
        ("causal", base_events + [make_event(f"resched_{index + 1:06d}_causal_block", "Customer Escalation", [p[0]], base, duration, ctx["other_room"], {"priority": "high"})], base, alt, {"type": "attendee_unavailable", "entity": p[0], "time_window": [base, add_minutes(base, duration)]}, ["event.start_time", "event.conflict_constraints", "query.availability"], {}, "hard"),
        ("distractor", base_events + [make_event(f"resched_{index + 1:06d}_distractor_block", "Room Hold", [ctx["outsider"]], dist, duration, room, {"priority": "medium"})], dist, base, {"type": "distractor_slot", "suggested_start": dist}, ["instruction.distractor_slot"], {"claimed_available_start": dist, "must_verify": True}, "hard"),
        ("conflict", base_events + [make_event(f"resched_{index + 1:06d}_conflict_block", "Exec Review", [p[1]], base, duration, room, {"priority": "high"})], base, alt, {"type": "conflict_user_claim", "entity": p[1], "claimed_available": base}, ["event.start_time", "event.conflict_constraints", "query.availability"], {"claimed_available_start": base, "must_verify": True}, "hard"),
    ]
    for world_type, events, preferred, target_start, intervention, cone, claim, difficulty in specs:
        goal = common_schedule_goal(title, p, room, duration, preferred, [preferred] + [s for s in candidates if s != preferred], "reschedule", base, claim)
        goal.update({"target_event_id": original_id, "stale_target_event_id": original_id, "requires_notification": True})
        target = target_event(original_id, title, p, target_start, duration, room)
        trace = reference_reschedule_trace(goal, target, preferred != target_start or bool(claim.get("must_verify")))
        minimal = len([a for a in trace if a["tool"] != "finish"])
        notifications = [{"event_id": original_id, "recipients": p}]
        episodes.append(build_episode(family_id, world_type, "reschedule", family_seed(seed, index), ctx["date"], goal, events, intervention, cone, target, trace, minimal, difficulty, required_notifications=notifications))
    return family_record(family_id, "reschedule", seed, index, episodes), episodes


def cancel_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ctx = base_context(index, seed, cfg)
    family_id = f"calendar_{index + 1:06d}"
    title = f"Budget Check {index + 1:04d}"
    p, room, duration = ctx["participants"], ctx["room"], ctx["duration"]
    base = ctx["base_start"]
    target_id = f"cancel_{index + 1:06d}_target"
    similar_id = f"cancel_{index + 1:06d}_similar"
    target_ev = make_event(target_id, title, p, base, duration, room, {"priority": "normal"})
    similar_ev = make_event(similar_id, title + " Followup", p, ctx["alt_start"], duration, room, {"priority": "low"})
    base_events = [ctx["unrelated"], target_ev, similar_ev]
    episodes: list[dict[str, Any]] = []
    specs = [
        ("base", base_events, target_id, {}, {"type": "none"}, [], "medium"),
        ("irrelevant", [dict(e, metadata={**e.get("metadata", {}), "color": "teal"}) if e["event_id"] == ctx["unrelated"]["event_id"] else e for e in base_events], target_id, {}, {"type": "metadata_change", "entity": ctx["unrelated"]["event_id"], "field": "metadata.color"}, [], "medium"),
        ("causal", base_events, similar_id, {}, {"type": "cancel_scope_changed", "entity": similar_id}, ["event.cancel_scope", "notification.recipients"], "hard"),
        ("distractor", base_events, target_id, {"claimed_event_id": similar_id, "must_verify": True}, {"type": "distractor_event", "suggested_event_id": similar_id}, ["instruction.distractor_event"], "hard"),
        ("conflict", base_events, target_id, {"claimed_event_id": similar_id, "must_verify": True}, {"type": "conflict_user_claim", "claimed_event_id": similar_id}, ["event.cancel_scope", "notification.recipients"], "hard"),
    ]
    for world_type, events, cancel_id, claim, intervention, cone, difficulty in specs:
        target_for_goal = next(e for e in events if e["event_id"] == cancel_id)
        goal = {
            "type": "cancel_and_notify",
            "title": target_for_goal["title"],
            "participants": target_for_goal["participants"],
            "target_event_id": cancel_id,
            "stale_target_event_id": target_id,
            "claimed_event_id": claim.get("claimed_event_id"),
            "requires_notification": True,
            "user_claim": claim,
            "verification_required": bool(claim.get("must_verify")),
        }
        trace = reference_cancel_trace(goal, cancel_id, target_for_goal["participants"])
        notifications = [{"event_id": cancel_id, "recipients": target_for_goal["participants"]}]
        episodes.append(build_episode(family_id, world_type, "cancel_and_notify", family_seed(seed, index), ctx["date"], goal, events, intervention, cone, None, trace, len([a for a in trace if a["tool"] != "finish"]), difficulty, cancel_event_id=cancel_id, required_notifications=notifications))
    return family_record(family_id, "cancel_and_notify", seed, index, episodes), episodes


def family_record(family_id: str, task_type: str, seed: int, index: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "domain": "calendar",
        "task_type": task_type,
        "seed": family_seed(seed, index),
        "episode_ids": {ep["world_type"]: ep["episode_id"] for ep in episodes},
        "world_types": WORLD_TYPES,
        "interventions": {ep["world_type"]: ep["intervention"] for ep in episodes},
    }


def generate_calendar_family(index: int, seed: int, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = config or {}
    task_types = list(cfg.get("task_types") or cfg.get("release_task_types") or ["schedule"])
    task_type = task_types[index % len(task_types)]
    if task_type == "schedule":
        return schedule_family(index, seed, cfg, "schedule")
    if task_type == "verify_before_act":
        return schedule_family(index, seed, cfg, "verify_before_act")
    if task_type == "reschedule":
        return reschedule_family(index, seed, cfg)
    if task_type == "cancel_and_notify":
        return cancel_family(index, seed, cfg)
    raise ValueError(f"unknown calendar task_type: {task_type}")
