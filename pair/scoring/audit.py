from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from pair.scoring.metrics import aggregate_metrics

METRIC_KEYS = ["ts", "ii", "cs", "pl", "tm"]
DEFAULT_WEIGHTS = {"ts": 0.25, "ii": 0.20, "cs": 0.20, "pl": 0.20, "tm": 0.15}
WEIGHT_SCHEMES = {
    "default": DEFAULT_WEIGHTS,
    "no_tm": {"ts": 0.30, "ii": 0.25, "cs": 0.25, "pl": 0.20, "tm": 0.00},
    "no_pl": {"ts": 0.30, "ii": 0.25, "cs": 0.25, "pl": 0.00, "tm": 0.20},
    "ts_only": {"ts": 1.00, "ii": 0.00, "cs": 0.00, "pl": 0.00, "tm": 0.00},
    "process_heavy": {"ts": 0.15, "ii": 0.20, "cs": 0.20, "pl": 0.25, "tm": 0.20},
    "causal_heavy": {"ts": 0.15, "ii": 0.30, "cs": 0.30, "pl": 0.15, "tm": 0.10},
}


def aggregate_agent_scores(agent_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {agent: aggregate_metrics(rows) for agent, rows in sorted(agent_rows.items())}


def weighted_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(float(row.get(k, 0.0)) * float(weights.get(k, 0.0)) for k in METRIC_KEYS)


def weight_sensitivity(agent_rows: dict[str, list[dict[str, Any]]], schemes: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    schemes = schemes or WEIGHT_SCHEMES
    out: dict[str, Any] = {}
    for name, weights in schemes.items():
        scores = {agent: mean(weighted_score(row, weights) for row in rows) for agent, rows in agent_rows.items()}
        ranking = sorted(scores, key=lambda a: (-scores[a], a))
        out[name] = {"weights": weights, "scores": scores, "ranking": ranking}
    default_rank = out.get("default", {}).get("ranking", [])
    for name, payload in out.items():
        ranking = payload["ranking"]
        payload["rank_shift_from_default"] = {agent: ranking.index(agent) - default_rank.index(agent) for agent in ranking if agent in default_rank}
    return out


def _family_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family_id"]].append(row)
    return grouped


def bootstrap_ci(agent_rows: dict[str, list[dict[str, Any]]], metric: str = "pair", n: int = 1000, seed: int = 13) -> dict[str, Any]:
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for agent, rows in agent_rows.items():
        grouped = _family_groups(rows)
        family_ids = sorted(grouped)
        samples: list[float] = []
        for _ in range(n):
            picked = [rng.choice(family_ids) for _ in family_ids]
            values = [float(row[metric]) for fid in picked for row in grouped[fid]]
            samples.append(mean(values))
        samples.sort()
        lo = samples[int(0.025 * (n - 1))]
        hi = samples[int(0.975 * (n - 1))]
        obs = mean(float(row[metric]) for row in rows)
        out[agent] = {"mean": obs, "ci95": [lo, hi], "metric": metric, "families": len(family_ids), "bootstrap_samples": n}
    return out


def pairwise_deltas(agent_rows: dict[str, list[dict[str, Any]]], pairs: list[tuple[str, str]], metric: str = "pair", n: int = 1000, seed: int = 17) -> dict[str, Any]:
    rng = random.Random(seed)
    grouped_by_agent = {agent: _family_groups(rows) for agent, rows in agent_rows.items()}
    out: dict[str, Any] = {}
    for left, right in pairs:
        common = sorted(set(grouped_by_agent[left]).intersection(grouped_by_agent[right]))
        samples: list[float] = []
        for _ in range(n):
            picked = [rng.choice(common) for _ in common]
            left_values = [float(row[metric]) for fid in picked for row in grouped_by_agent[left][fid]]
            right_values = [float(row[metric]) for fid in picked for row in grouped_by_agent[right][fid]]
            samples.append(mean(left_values) - mean(right_values))
        samples.sort()
        obs = mean(float(row[metric]) for row in agent_rows[left]) - mean(float(row[metric]) for row in agent_rows[right])
        out[f"{left}_minus_{right}"] = {"delta": obs, "ci95": [samples[int(0.025 * (n - 1))], samples[int(0.975 * (n - 1))]], "metric": metric, "families": len(common), "bootstrap_samples": n}
    return out


def failure_taxonomy_table(agent_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_modes = sorted({mode for metrics in agent_metrics.values() for mode in metrics.get("failure_taxonomy", {})})
    table = {}
    for agent, metrics in sorted(agent_metrics.items()):
        taxonomy = metrics.get("failure_taxonomy", {})
        table[agent] = {mode: int(taxonomy.get(mode, 0)) for mode in all_modes}
    return {"modes": all_modes, "table": table}
