from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CPDConfig:
    model: str
    lambda_inv: float = 0.5
    lambda_sens: float = 0.5
    lambda_rank: float = 0.2


class CPDTrainerInterface:
    def train(self, config: CPDConfig):
        raise NotImplementedError("CPD training is reserved for Phase 2.")
