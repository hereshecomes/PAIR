from __future__ import annotations

from typing import Iterable


def split_family_ids(family_ids: Iterable[str], train: float = 0.6, valid: float = 0.2) -> dict[str, list[str]]:
    ids = sorted(family_ids)
    n = len(ids)
    train_n = int(n * train)
    valid_n = int(n * valid)
    return {"train": ids[:train_n], "valid": ids[train_n:train_n + valid_n], "test": ids[train_n + valid_n:]}
