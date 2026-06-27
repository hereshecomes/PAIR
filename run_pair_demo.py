from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def add_pair_package() -> Path:
    here = Path(__file__).resolve().parent
    for root in [here / "pair_anonymous_release", here]:
        if (root / "pair").is_dir():
            sys.path.insert(0, str(root))
            return root
    archive = here / "pair_anonymous_release.tar.gz"
    if not archive.exists():
        raise FileNotFoundError("Expected pair_anonymous_release/ or pair_anonymous_release.tar.gz next to this file.")
    target = Path(tempfile.gettempdir()) / "pair_demo_package"
    package_root = target / "pair_anonymous_release"
    if not (package_root / "pair").is_dir():
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                destination = (target / member.name).resolve()
                if root != destination and root not in destination.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
            handle.extractall(target, members=members)
    sys.path.insert(0, str(package_root))
    return package_root


def replay_episode(episode: dict[str, Any], mode: str) -> dict[str, Any]:
    from pair.agents.base_agent import finish_row
    from pair.envs.calendar_world import CalendarWorld

    env = CalendarWorld(episode)
    if mode == "reference":
        actions = episode["oracle"]["reference_trace"]
    elif mode == "stop_immediately":
        actions = [{"tool": "finish", "args": {}}]
    else:
        raise ValueError(f"Unknown replay mode: {mode}")
    for action in actions:
        env.step(action)
    return finish_row(env, mode)


def rounded_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "episodes",
        "mean_ts",
        "mean_ii",
        "mean_cs",
        "mean_pl",
        "mean_tm",
        "mean_pair",
        "mean_progress_gate",
        "mean_pair_gated",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        value = summary.get(key)
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


def main() -> None:
    add_pair_package()
    from pair.generators.calendar_generator import generate_calendar_family
    from pair.generators.validate_family import validate_family
    from pair.scoring.metrics import aggregate_metrics, compute_episode_metrics

    family, episodes = generate_calendar_family(0, 7, {"task_types": ["schedule"]})
    base_episode = next(episode for episode in episodes if episode["world_type"] == "base")
    output: dict[str, Any] = {
        "family_id": family["family_id"],
        "domain": family["domain"],
        "task_type": family["task_type"],
        "world_types": family["world_types"],
        "validation_passed": bool(validate_family(family, episodes)["passed"]),
        "runs": {},
    }
    for mode in ["reference", "stop_immediately"]:
        base_prediction = replay_episode(base_episode, mode)
        rows = [
            compute_episode_metrics(episode, replay_episode(episode, mode), base_episode, base_prediction)
            for episode in episodes
        ]
        output["runs"][mode] = {
            "overall": rounded_summary(aggregate_metrics(rows)["overall"]),
            "by_world_type": {
                world_type: rounded_summary(values)
                for world_type, values in aggregate_metrics(rows)["by_world_type"].items()
            },
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
