from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from pair.envs.base import BaseEnv
from pair.utils.io import stable_json_hash

ISO = "%Y-%m-%dT%H:%M:%S"


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, ISO)


def try_parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_time(value)
    except ValueError:
        return None


def valid_time_window(start: Any, end: Any) -> bool:
    start_dt = try_parse_time(start)
    end_dt = try_parse_time(end)
    return bool(start_dt and end_dt and start_dt < end_dt)


def overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    start_a_dt = try_parse_time(start_a)
    end_a_dt = try_parse_time(end_a)
    start_b_dt = try_parse_time(start_b)
    end_b_dt = try_parse_time(end_b)
    if not (start_a_dt and end_a_dt and start_b_dt and end_b_dt):
        return False
    return start_a_dt < end_b_dt and start_b_dt < end_a_dt


def canonical_people(values: list[str] | None) -> list[str]:
    return sorted(str(v) for v in (values or []))


class CalendarWorld(BaseEnv):
    def __init__(self, episode: dict[str, Any]):
        self.episode = copy.deepcopy(episode)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.events = copy.deepcopy(self.episode["world"]["events"])
        for ev in self.events:
            ev.setdefault("status", "active")
        self.notifications: list[dict[str, Any]] = []
        self.done = False
        self.trace: list[dict[str, Any]] = []
        return {"events": copy.deepcopy(self.events), "goal": copy.deepcopy(self.episode["goal"])}

    def available_tools(self) -> list[str]:
        return ["list_events", "check_availability", "create_event", "update_event", "cancel_event", "send_notification", "finish"]

    def _find_index(self, event_id: str) -> int | None:
        return next((i for i, ev in enumerate(self.events) if ev["event_id"] == event_id), None)

    def _active_events(self) -> list[dict[str, Any]]:
        return [ev for ev in self.events if ev.get("status", "active") != "canceled"]

    def _conflicts(self, participants: list[str], start: str, end: str, room: str | None = None, exclude_event_id: str | None = None) -> list[dict[str, Any]]:
        people = set(participants)
        found: list[dict[str, Any]] = []
        for ev in self._active_events():
            if exclude_event_id and ev["event_id"] == exclude_event_id:
                continue
            if not overlaps(start, end, ev["start"], ev["end"]):
                continue
            participant_hit = bool(people.intersection(ev.get("participants", [])))
            room_hit = bool(room and ev.get("room") == room)
            if participant_hit or room_hit:
                found.append(copy.deepcopy(ev))
        return found

    def _record(self, tool: str, args: dict[str, Any], ok: bool, observation: dict[str, Any], state_delta: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {"index": len(self.trace), "tool": tool, "args": copy.deepcopy(args), "ok": bool(ok), "observation": copy.deepcopy(observation), "state_delta": copy.deepcopy(state_delta or {})}
        self.trace.append(row)
        return copy.deepcopy(observation)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            return self._record("after_done", action, False, {"error": "episode already finished"})
        tool = str(action.get("tool", ""))
        args = copy.deepcopy(action.get("args", {}))
        if tool == "list_events":
            participants = set(args.get("participants") or [])
            title = args.get("title")
            start = args.get("start")
            end = args.get("end")
            include_canceled = bool(args.get("include_canceled", False))
            rows = []
            source = self.events if include_canceled else self._active_events()
            for ev in source:
                if participants and not participants.intersection(ev.get("participants", [])):
                    continue
                if title and title not in ev.get("title", ""):
                    continue
                if start is not None or end is not None:
                    if not valid_time_window(start, end):
                        return self._record(tool, args, False, {"error": "invalid time window", "events": [], "count": 0})
                    if not overlaps(start, end, ev["start"], ev["end"]):
                        continue
                rows.append(copy.deepcopy(ev))
            return self._record(tool, args, True, {"events": rows, "count": len(rows)})
        if tool == "check_availability":
            participants = canonical_people(args.get("participants", []))
            raw_start = args.get("start")
            raw_end = args.get("end")
            if not valid_time_window(raw_start, raw_end):
                return self._record(tool, args, False, {"available": False, "conflicts": [], "error": "invalid time window"})
            start = str(raw_start)
            end = str(raw_end)
            room = args.get("room")
            conflicts = self._conflicts(participants, start, end, room=room, exclude_event_id=args.get("exclude_event_id"))
            return self._record(tool, {"participants": participants, "start": start, "end": end, "room": room, "exclude_event_id": args.get("exclude_event_id")}, True, {"available": not conflicts, "conflicts": conflicts})
        if tool == "create_event":
            participants = canonical_people(args.get("participants", []))
            raw_start = args.get("start")
            raw_end = args.get("end")
            if not valid_time_window(raw_start, raw_end):
                return self._record(tool, args, False, {"created": False, "conflicts": [], "error": "invalid time window"}, {"failed_write": True})
            start = str(raw_start)
            end = str(raw_end)
            room = str(args.get("room", ""))
            conflicts = self._conflicts(participants, start, end, room=room)
            normalized = {"event_id": f"created_{len([e for e in self.events if str(e.get('event_id', '')).startswith('created_')]) + 1}", "title": str(args.get("title", "Untitled")), "participants": participants, "start": start, "end": end, "room": room, "status": "active", "metadata": copy.deepcopy(args.get("metadata", {}))}
            if conflicts:
                return self._record(tool, normalized, False, {"created": False, "conflicts": conflicts})
            self.events.append(copy.deepcopy(normalized))
            return self._record(tool, normalized, True, {"created": True, "event": normalized}, {"created_event_id": normalized["event_id"], "write_event_id": normalized["event_id"]})
        if tool == "update_event":
            event_id = str(args.get("event_id", ""))
            idx = self._find_index(event_id)
            if idx is None:
                return self._record(tool, args, False, {"updated": False, "error": "event not found"})
            before = copy.deepcopy(self.events[idx])
            if before.get("status") == "canceled":
                return self._record(tool, args, False, {"updated": False, "error": "event canceled"})
            candidate = copy.deepcopy(before)
            for field in ["title", "participants", "start", "end", "room", "metadata"]:
                if field in args:
                    candidate[field] = canonical_people(args[field]) if field == "participants" else copy.deepcopy(args[field])
            if not valid_time_window(candidate.get("start"), candidate.get("end")):
                return self._record(tool, args, False, {"updated": False, "error": "invalid time window"}, {"write_event_id": event_id, "failed_write": True})
            conflicts = self._conflicts(candidate["participants"], candidate["start"], candidate["end"], room=candidate.get("room"), exclude_event_id=event_id)
            if conflicts:
                return self._record(tool, args, False, {"updated": False, "conflicts": conflicts}, {"write_event_id": event_id, "failed_write": True})
            self.events[idx] = candidate
            changed = {k: [before.get(k), candidate.get(k)] for k in candidate if before.get(k) != candidate.get(k)}
            return self._record(tool, args, True, {"updated": True, "event": candidate}, {"updated_event_id": event_id, "write_event_id": event_id, "changed_fields": changed})
        if tool == "cancel_event":
            event_id = str(args.get("event_id", ""))
            idx = self._find_index(event_id)
            if idx is None:
                return self._record(tool, args, False, {"canceled": False, "error": "event not found"})
            before = copy.deepcopy(self.events[idx])
            if before.get("status") == "canceled":
                return self._record(tool, args, False, {"canceled": False, "error": "already canceled"}, {"write_event_id": event_id, "failed_write": True})
            self.events[idx]["status"] = "canceled"
            self.events[idx].setdefault("metadata", {})["cancel_reason"] = args.get("reason", "")
            return self._record(tool, args, True, {"canceled": True, "event_id": event_id}, {"canceled_event_id": event_id, "write_event_id": event_id, "changed_fields": {"status": [before.get("status", "active"), "canceled"]}})
        if tool == "send_notification":
            event_id = str(args.get("event_id", ""))
            recipients = canonical_people(args.get("recipients", []))
            note = {"event_id": event_id, "recipients": recipients, "message": str(args.get("message", ""))}
            self.notifications.append(note)
            return self._record(tool, args, True, {"sent": True, "notification": note}, {"notification_event_id": event_id, "notification_recipients": recipients})
        if tool == "finish":
            self.done = True
            return self._record(tool, args, True, {"finished": True, "final_success": self.final_check()})
        return self._record(tool, args, False, {"error": f"unknown tool {tool}"})

    def _event_by_id(self, event_id: str | None) -> dict[str, Any] | None:
        if not event_id:
            return None
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return copy.deepcopy(ev)
        return None

    def _event_matches_target(self, ev: dict[str, Any] | None, target: dict[str, Any] | None) -> bool:
        if not ev or not target:
            return False
        if ev.get("status", "active") == "canceled":
            return False
        same = ev.get("title") == target.get("title") and canonical_people(ev.get("participants")) == canonical_people(target.get("participants")) and ev.get("start") == target.get("start") and ev.get("end") == target.get("end") and ev.get("room") == target.get("room")
        if not same:
            return False
        return not self._conflicts(ev["participants"], ev["start"], ev["end"], room=ev.get("room"), exclude_event_id=ev.get("event_id"))

    def _matching_event(self) -> dict[str, Any] | None:
        target = self.episode["oracle"].get("target_event")
        if not target:
            return None
        target_id = target.get("event_id")
        if target_id and target_id != "created_target":
            ev = self._event_by_id(target_id)
            return ev if self._event_matches_target(ev, target) else None
        for ev in self.events:
            if self._event_matches_target(ev, target):
                return copy.deepcopy(ev)
        return None

    def _best_event(self) -> dict[str, Any] | None:
        matched = self._matching_event()
        if matched:
            return matched
        target = self.episode["oracle"].get("target_event") or {}
        if target.get("event_id") and target.get("event_id") != "created_target":
            return self._event_by_id(target.get("event_id"))
        created = [ev for ev in self.events if str(ev.get("event_id", "")).startswith("created_") and ev.get("title") == target.get("title")]
        return copy.deepcopy(created[-1]) if created else None

    def _notification_ok(self, event_id: str, recipients: list[str]) -> bool:
        needed = set(canonical_people(recipients))
        for note in self.notifications:
            if note.get("event_id") == event_id and needed.issubset(set(canonical_people(note.get("recipients")))):
                return True
        return False

    def _verification_ok(self) -> bool:
        if not self.episode.get("goal", {}).get("verification_required"):
            return True
        first_write = next((s["index"] for s in self.trace if s.get("tool") in {"create_event", "update_event", "cancel_event"}), None)
        if first_write is None:
            return False
        return any(s.get("tool") in {"check_availability", "list_events"} and s["index"] < first_write for s in self.trace)

    def _cancel_success(self) -> bool:
        cancel_id = self.episode["oracle"].get("cancel_event_id")
        ev = self._event_by_id(cancel_id)
        if not ev or ev.get("status") != "canceled":
            return False
        unrelated_canceled = [e for e in self.events if e.get("status") == "canceled" and e.get("event_id") != cancel_id]
        if unrelated_canceled:
            return False
        for req in self.episode["oracle"].get("required_notifications", []):
            if not self._notification_ok(req["event_id"], req.get("recipients", [])):
                return False
        return self._verification_ok()

    def final_check(self) -> bool:
        task_type = self.episode.get("task_type") or self.episode.get("goal", {}).get("type")
        if task_type == "cancel_and_notify":
            return self._cancel_success()
        matched = self._matching_event()
        if not matched:
            return False
        if not self._verification_ok():
            return False
        for req in self.episode["oracle"].get("required_notifications", []):
            if not self._notification_ok(req["event_id"], req.get("recipients", [])):
                return False
        return True

    def state_hash(self) -> str:
        return stable_json_hash({"events": self.events, "notifications": self.notifications})

    def summary(self) -> dict[str, Any]:
        target_id = self.episode["oracle"].get("target_event_id")
        cancel_id = self.episode["oracle"].get("cancel_event_id")
        return {
            "state_hash": self.state_hash(),
            "events": copy.deepcopy(self.events),
            "notifications": copy.deepcopy(self.notifications),
            "matched_event": self._matching_event(),
            "best_created_event": self._best_event(),
            "target_event_state": self._event_by_id(target_id),
            "cancel_event_state": self._event_by_id(cancel_id),
            "canceled_events": [copy.deepcopy(e) for e in self.events if e.get("status") == "canceled"],
            "verification_ok": self._verification_ok(),
            "final_success": self.final_check(),
        }
