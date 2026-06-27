from __future__ import annotations

from pair.scoring.metrics import aggregate_metrics


def score_report(rows):
    return aggregate_metrics(rows)
