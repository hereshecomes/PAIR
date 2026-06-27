from __future__ import annotations

import copy
import random
from typing import Any

from pair.utils.io import stable_json_hash
from pair.utils.seed import family_seed

WORLD_TYPES = ["base", "irrelevant", "causal", "distractor", "conflict"]
TASK_TYPES = ["fix_failing_test", "update_api_contract", "repair_edge_case", "verify_before_edit"]


def _calc_files(idx: int) -> dict[str, str]:
    return {
        "src/calc.py": "def safe_divide(a, b):\n    return a / b\n",
        "tests/test_calc.py": "from src.calc import safe_divide\n\n\ndef test_safe_divide_zero():\n    assert safe_divide(6, 0) is None\n",
        "README.md": f"# Fixture {idx}\n\nSmall arithmetic package.\n",
    }


def _api_files(idx: int) -> dict[str, str]:
    return {
        "src/api.py": "def format_user(name):\n    return name.title()\n",
        "tests/test_api.py": "from src.api import format_user\n\n\ndef test_format_user():\n    assert format_user('ada') == 'Ada'\n",
        "README.md": f"# Fixture {idx}\n\nSmall API package.\n",
    }


def _text_files(idx: int) -> dict[str, str]:
    return {
        "src/text.py": "def slugify(text):\n    return text.lower().replace(' ', '-')\n",
        "tests/test_text.py": "from src.text import slugify\n\n\ndef test_slugify_punctuation():\n    assert slugify('Hello, World!') == 'hello-world'\n",
        "README.md": f"# Fixture {idx}\n\nSmall text package.\n",
    }


def _verify_files(idx: int) -> dict[str, str]:
    return {
        "src/normalize.py": "def normalize_email(value):\n    return value.strip()\n",
        "tests/test_normalize.py": "from src.normalize import normalize_email\n\n\ndef test_normalize_email():\n    assert normalize_email(' Ada@Example.COM ') == 'ada@example.com'\n",
        "README.md": f"# Fixture {idx}\n\nSmall normalization package.\n",
    }


def _replace(files: dict[str, str], path: str, content: str) -> dict[str, str]:
    out = copy.deepcopy(files)
    out[path] = content
    return out


def _episode(
    family_id: str,
    world_type: str,
    task_type: str,
    seed: int,
    world: dict[str, Any],
    goal: dict[str, Any],
    intervention: dict[str, Any],
    cone: list[str],
    oracle: dict[str, Any],
    difficulty: str,
) -> dict[str, Any]:
    return {
        "schema_version": "pair.codebase.v0.1",
        "domain": "codebase",
        "family_id": family_id,
        "episode_id": f"{family_id}_{world_type}",
        "task_type": task_type,
        "world_type": world_type,
        "difficulty": difficulty,
        "intervention_role": intervention.get("type", "none"),
        "seed": seed,
        "goal": copy.deepcopy(goal),
        "world": copy.deepcopy(world),
        "intervention": copy.deepcopy(intervention),
        "affected_cone": list(cone),
        "oracle": copy.deepcopy(oracle),
        "state_hash": stable_json_hash({"world": world, "goal": goal, "intervention": intervention, "task_type": task_type}),
    }


def _family_record(family_id: str, task_type: str, seed: int, index: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"family_id": family_id, "domain": "codebase", "task_type": task_type, "seed": family_seed(seed, index), "episode_ids": {ep["world_type"]: ep["episode_id"] for ep in episodes}}


def _reference_trace(target_file: str, expected_content: str) -> list[dict[str, Any]]:
    return [
        {"tool": "search_code", "args": {"query": target_file.split("/")[-1].split(".")[0]}},
        {"tool": "read_file", "args": {"path": target_file}},
        {"tool": "run_tests", "args": {}},
        {"tool": "edit_file", "args": {"path": target_file, "content": expected_content}},
        {"tool": "run_tests", "args": {}},
        {"tool": "finish", "args": {}},
    ]


def _world(files: dict[str, str]) -> dict[str, Any]:
    return {"files": copy.deepcopy(files), "language": "python", "test_runner": "builtin_pytest_checker"}


def _build_worlds(
    family_id: str,
    task_type: str,
    seed: int,
    base_files: dict[str, str],
    target_file: str,
    base_expected: str,
    causal_expected: str,
    conflict_expected: str,
    base_goal: dict[str, Any],
    bug_file: str,
    distractor_file: str,
    failure_hint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    stale_expected = base_expected
    for wt in WORLD_TYPES:
        files = copy.deepcopy(base_files)
        goal = copy.deepcopy(base_goal)
        intervention = {"type": "none"}
        cone: list[str] = []
        expected = base_expected
        difficulty = "medium"
        forbidden_files = [distractor_file, "README.md"]
        if wt == "irrelevant":
            files["docs/notes.md"] = "Cosmetic release note. Do not edit for this task.\n"
            intervention = {"type": "irrelevant_doc", "entity": "docs/notes.md"}
            forbidden_files.append("docs/notes.md")
            difficulty = "easy"
        elif wt == "causal":
            expected = causal_expected
            intervention = {"type": "test_expectation_changed", "entity": bug_file}
            cone = ["tests.expected_behavior", "code.target_function", "test.result"]
            goal["user_claim"] = {"claim": "The failure may be in the original target function; verify tests before editing.", "must_verify": True}
            difficulty = "hard"
        elif wt == "distractor":
            goal["user_claim"] = {"claimed_file": distractor_file, "claim": "A previous note says the bug is probably here.", "must_verify": True}
            intervention = {"type": "distractor_file_claim", "suggested_file": distractor_file}
            cone = ["instruction.distractor_file"]
            difficulty = "hard"
        elif wt == "conflict":
            expected = conflict_expected
            goal["user_claim"] = {"claimed_fix": "Use the old behavior from the stale issue description.", "stale_expected_file": target_file, "must_verify": True}
            intervention = {"type": "stale_issue_conflict", "entity": bug_file}
            cone = ["tests.expected_behavior", "code.target_function", "test.result"]
            difficulty = "hard"
        oracle = {
            "task_type": task_type,
            "expected_files": {target_file: expected},
            "target_files": [target_file],
            "forbidden_files": forbidden_files,
            "stale_expected_files": {target_file: stale_expected},
            "claimed_file": goal.get("user_claim", {}).get("claimed_file"),
            "reference_trace": _reference_trace(target_file, expected),
            "partial_order": {"constraints": ["read-before-edit", "test-before-finish"]},
            "minimal_semantic_calls": 5,
            "final_checker": "codebase_builtin_checker",
            "test_names": [bug_file],
            "test_failure_hint": failure_hint,
        }
        episodes.append(_episode(family_id, wt, task_type, seed, _world(files), goal, intervention, cone, oracle, difficulty))
    return _family_record(family_id, task_type, seed, int(family_id.rsplit("_", 1)[1]) - 1, episodes), episodes


def fix_failing_test_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_id = f"codebase_{index + 1:06d}"
    files = _calc_files(index + 1)
    return _build_worlds(
        family_id,
        "fix_failing_test",
        family_seed(seed, index),
        files,
        "src/calc.py",
        "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\n",
        "def safe_divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n",
        {"type": "fix_failing_test", "repo": family_id, "target_file_hint": "src/calc.py", "test_command": "run_tests", "verification_required": True, "description": "Fix the failing safe_divide behavior with a local code change."},
        "tests/test_calc.py",
        "src/api.py",
        "safe_divide return behavior does not match the failing zero-division assertion",
    )


def update_api_contract_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_id = f"codebase_{index + 1:06d}"
    files = _api_files(index + 1)
    return _build_worlds(
        family_id,
        "update_api_contract",
        family_seed(seed, index),
        files,
        "src/api.py",
        "def format_user(name, uppercase=False):\n    value = name.strip().title()\n    return value.upper() if uppercase else value\n",
        "def format_user(name, uppercase=False, prefix=''):\n    value = name.strip().title()\n    if prefix:\n        value = f'{prefix} {value}'\n    return value.upper() if uppercase else value\n",
        "def format_user(name, uppercase=True):\n    value = name.strip().title()\n    return value.upper() if uppercase else value\n",
        {"type": "update_api_contract", "repo": family_id, "target_file_hint": "src/api.py", "test_command": "run_tests", "verification_required": True, "description": "Update the public format_user API to match the tests while keeping the change local."},
        "tests/test_api.py",
        "src/calc.py",
        "format_user contract does not match the failing API assertion",
    )


def repair_edge_case_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_id = f"codebase_{index + 1:06d}"
    files = _text_files(index + 1)
    return _build_worlds(
        family_id,
        "repair_edge_case",
        family_seed(seed, index),
        files,
        "src/text.py",
        "import re\n\n\ndef slugify(text):\n    value = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n    return value\n",
        "import re\n\n\ndef slugify(text):\n    value = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n    return value or 'empty'\n",
        "def slugify(text):\n    return text.strip().lower().replace(' ', '-')\n",
        {"type": "repair_edge_case", "repo": family_id, "target_file_hint": "src/text.py", "test_command": "run_tests", "verification_required": True, "description": "Repair slugify edge-case handling without changing unrelated modules."},
        "tests/test_text.py",
        "src/api.py",
        "slugify output does not match the failing edge-case assertion",
    )


def verify_before_edit_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_id = f"codebase_{index + 1:06d}"
    files = _verify_files(index + 1)
    return _build_worlds(
        family_id,
        "verify_before_edit",
        family_seed(seed, index),
        files,
        "src/normalize.py",
        "def normalize_email(value):\n    return value.strip().lower()\n",
        "def normalize_email(value):\n    return value.strip().casefold()\n",
        "def normalize_email(value):\n    local, _, domain = value.strip().partition('@')\n    return local + '@' + domain.lower()\n",
        {"type": "verify_before_edit", "repo": family_id, "target_file_hint": "src/normalize.py", "test_command": "run_tests", "verification_required": True, "description": "Verify the failing behavior before editing normalize_email."},
        "tests/test_normalize.py",
        "src/text.py",
        "normalize_email output does not match the failing normalization assertion",
    )


def generate_codebase_family(index: int, seed: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random.Random(family_seed(seed, index))
    tasks = list(cfg.get("task_types") or TASK_TYPES)
    task = tasks[index % len(tasks)]
    if task == "fix_failing_test":
        return fix_failing_test_family(index, seed, cfg)
    if task == "update_api_contract":
        return update_api_contract_family(index, seed, cfg)
    if task == "repair_edge_case":
        return repair_edge_case_family(index, seed, cfg)
    if task == "verify_before_edit":
        return verify_before_edit_family(index, seed, cfg)
    raise ValueError(f"unknown codebase task type: {task}")
