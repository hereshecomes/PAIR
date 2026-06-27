# PAIR

PAIR is an anonymized paired-intervention evaluation library for tool-using agents. It provides core components for constructing controlled twin worlds, replaying tool trajectories, canonicalizing actions, and computing programmatic process metrics.

This release is intentionally minimal. It keeps the reusable benchmark skeleton and removes experiment launchers, model-specific runners, raw result files, paper assets, and nonessential baseline scripts.

## Contents

- `pair/agents`: base agent interface and trace finalization helper.
- `pair/envs`: executable environments for calendar, spreadsheet, and codebase-style tasks.
- `pair/generators`: paired-world family generators and validation utilities.
- `pair/oracles`: reference-trace and partial-order accessors.
- `pair/scoring`: canonicalization, episode metrics, progress-gated summary score, bootstrap and ranking utilities.
- `pair/reporting`: lightweight trace rendering utilities.
- `pair/models`: placeholder interface for downstream diagnostic models.

The local archive is expected to be named:

```bash
pair_anonymous_release.tar.gz
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
tar -xzf pair_anonymous_release.tar.gz
pip install -e pair_anonymous_release
```

The demo runner can also load the archive directly if `pair_anonymous_release/` has not been extracted.

## Quick Run

```bash
python3 run_pair_demo.py
```

Expected behavior:

- generate one Calendar paired-world family with five worlds;
- validate the family schema, leakage checks, and reference replay;
- replay the reference trajectory;
- replay a stop-immediately sanity control;
- print aggregate `TS`, `II`, `CS`, `PL`, `TM`, `PAIR`, progress-gate rate, and `PAIR_gated`.

The reference trajectory should score near 1.0 across all metrics. The stop-immediately control should have low terminal success and a low gated score, showing that the gated summary does not reward lazy traces.

## Core Metrics

PAIR reports a metric vector rather than relying only on terminal success.

- `TS`: terminal success under the final checker.
- `II`: intervention impact, measuring whether the trace absorbs causal changes and remains stable under irrelevant changes.
- `CS`: causal specificity, measuring whether actions touch the correct object, field, and scope.
- `PL`: procedural locality, measuring whether the repair is local and avoids unrelated or redundant operations.
- `TM`: tool minimality, measuring tool-use economy after canonicalization.
- `PAIR`: compact weighted summary of the metric vector.
- `PAIR_gated`: progress-gated summary that withholds locality and tool-economy credit when a trace makes no task-relevant verification or state-changing progress.

Scalar summaries are computed per episode and then averaged. Because the progress gate is episode-specific, `PAIR_gated` should not be reconstructed by applying the formula to aggregate metric means.

## Minimal API Example

```python
from pair.generators.calendar_generator import generate_calendar_family
from pair.envs.calendar_world import CalendarWorld
from pair.scoring.metrics import compute_episode_metrics

family, episodes = generate_calendar_family(0, 7, {"task_types": ["schedule"]})
episode = episodes[0]
env = CalendarWorld(episode)

for action in episode["oracle"]["reference_trace"]:
    env.step(action)

prediction = {
    "episode_id": episode["episode_id"],
    "family_id": episode["family_id"],
    "domain": episode["domain"],
    "world_type": episode["world_type"],
    "agent": "reference",
    "trace": env.trace,
    "final_state": env.summary(),
    "final_success": env.summary()["final_success"],
}

metrics = compute_episode_metrics(episode, prediction)
```

## Anonymity

This release avoids author names, institution names, server paths, local filesystem paths, model-provider credentials, and project history. It is intended for anonymous review as a compact code artifact rather than a full experiment archive.
