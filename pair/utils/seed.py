from __future__ import annotations

import random


def family_seed(seed: int, index: int, stride: int = 1009) -> int:
    return int(seed) + int(index) * stride


def rng_for(seed: int, index: int = 0) -> random.Random:
    return random.Random(family_seed(seed, index))
