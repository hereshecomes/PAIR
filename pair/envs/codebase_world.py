from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from pair.envs.base import BaseEnv
from pair.utils.io import stable_json_hash


@dataclass(frozen=True)
class CodebaseSandboxPolicy:
    network_disabled: bool = True
    temp_workspace_only: bool = True
    allow_shell: bool = False
    allow_package_install: bool = False
    subprocess_default_enabled: bool = False
    pytest_timeout_seconds: int = 10
    max_file_bytes: int = 200_000
    max_processes: int = 4


DEFAULT_POLICY = CodebaseSandboxPolicy()


class CodebaseWorld(BaseEnv):
       

    def __init__(self, episode: dict[str, Any], policy: CodebaseSandboxPolicy = DEFAULT_POLICY):
        self.episode = episode
        self.policy = policy
        self.files = copy.deepcopy(episode.get("world", {}).get("files", {}))
        self.initial_files = copy.deepcopy(self.files)
        self.trace: list[dict[str, Any]] = []
        self.done = False
        self.test_runs = 0
        self.last_test_result: dict[str, Any] | None = None

    def reset(self) -> dict[str, Any]:
        self.files = copy.deepcopy(self.initial_files)
        self.trace = []
        self.done = False
        self.test_runs = 0
        self.last_test_result = None
        return {"files": sorted(self.files), "policy": self.policy.__dict__}

    def _safe_path(self, path: str | None) -> tuple[bool, str]:
        if not path or not isinstance(path, str):
            return False, "missing path"
        if path.startswith("/") or ".." in path.split("/"):
            return False, "path outside virtual workspace"
        if path not in self.files and path not in self.episode.get("oracle", {}).get("allowed_new_files", []):
            return False, "unknown file"
        return True, ""

    def _record(self, tool: str, args: dict[str, Any], ok: bool, observation: dict[str, Any], state_delta: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "index": len(self.trace),
            "tool": tool,
            "args": copy.deepcopy(args),
            "ok": bool(ok),
            "observation": copy.deepcopy(observation),
            "state_delta": copy.deepcopy(state_delta or {}),
        }
        self.trace.append(row)
        return observation

    def _run_tests(self) -> dict[str, Any]:
        expected = self.episode.get("oracle", {}).get("expected_files", {})
        forbidden = set(self.episode.get("oracle", {}).get("forbidden_files", []))
        failures = []
        for path, expected_content in expected.items():
            if self.files.get(path) != expected_content:
                failures.append({"file": path, "reason": "content_mismatch", "hint": self.episode.get("oracle", {}).get("test_failure_hint", "target source behavior does not match tests")})
        for path in forbidden:
            if self.files.get(path) != self.initial_files.get(path):
                failures.append({"file": path, "reason": "forbidden_file_modified"})
        passed = not failures
        return {
            "passed": passed,
            "failures": failures[:8],
            "summary": f"{'passed' if passed else 'failed'} {len(failures)} checks",
            "tests": self.episode.get("oracle", {}).get("test_names", []),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action.get("tool")
        args = dict(action.get("args") or {})
        if self.done and tool != "finish":
            return self._record(str(tool), args, False, {"error": "episode already finished"})
        if tool == "read_file":
            path = args.get("path")
            ok, error = self._safe_path(path)
            if not ok:
                return self._record(tool, args, False, {"error": error})
            return self._record(tool, args, True, {"path": path, "content": self.files[path]})
        if tool == "search_code":
            query = str(args.get("query", ""))
            matches = []
            for path, content in self.files.items():
                if query.lower() in content.lower() or query.lower() in path.lower():
                    matches.append({"path": path, "line": 1, "preview": content[:160]})
            return self._record(tool, args, True, {"query": query, "matches": matches[:20]})
        if tool == "edit_file":
            path = args.get("path")
            content = args.get("content")
            ok, error = self._safe_path(path)
            if not ok:
                return self._record(tool, args, False, {"error": error})
            if not isinstance(content, str):
                return self._record(tool, args, False, {"error": "content must be a string"})
            if len(content.encode("utf-8")) > self.policy.max_file_bytes:
                return self._record(tool, args, False, {"error": "file too large"})
            before = self.files.get(path)
            self.files[path] = content
            return self._record(tool, args, True, {"path": path, "updated": True}, {"write_file": path, "changed": before != content})
        if tool == "run_tests":
            self.test_runs += 1
            self.last_test_result = self._run_tests()
            return self._record(tool, args, True, self.last_test_result, {"test_run": self.test_runs})
        if tool == "finish":
            self.done = True
            result = self._run_tests()
            self.last_test_result = result
            return self._record(tool, args, True, {"final_success": result["passed"], "test_summary": result["summary"]})
        return self._record(str(tool), args, False, {"error": f"unknown tool: {tool}"})

    def available_tools(self) -> list[str]:
        return ["read_file", "search_code", "edit_file", "run_tests", "finish"]

    def state_hash(self) -> str:
        return stable_json_hash({"files": self.files})

    def final_check(self) -> bool:
        return bool(self._run_tests()["passed"])

    def summary(self) -> dict[str, Any]:
        result = self.last_test_result or self._run_tests()
        touched = []
        for path, content in self.files.items():
            if self.initial_files.get(path) != content:
                touched.append(path)
        return {
            "final_success": bool(result["passed"]),
            "tests_passed": bool(result["passed"]),
            "test_failures": result.get("failures", []),
            "test_runs": self.test_runs,
            "files": copy.deepcopy(self.files),
            "touched_files": sorted(touched),
            "state_hash": self.state_hash(),
            "policy": self.policy.__dict__,
        }
