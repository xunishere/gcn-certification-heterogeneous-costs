"""Shared mode configuration and threat-model semantics."""
from dataclasses import asdict, dataclass

MODES = ("legacy", "heterogeneous-cost", "risk-budget", "combined")
PAPER_MODES = ("legacy", "heterogeneous-cost")


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "heterogeneous-cost"
    alpha: float = 1.0
    eta: float = 0.5
    beta: float = 10.0
    q0: int = 14
    q: int = 3
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown mode: " + self.mode)
        if self.alpha < 0 or self.beta < 0 or self.q0 < 0:
            raise ValueError("cost and budget parameters must be non-negative")
        if self.q < 0 or int(self.q) != self.q:
            raise ValueError("q must be a non-negative integer flip count")
        if not 0 <= self.eta <= 1:
            raise ValueError("eta must be in [0,1]")

    @property
    def use_heterogeneous_cost(self):
        return self.mode in ("heterogeneous-cost", "combined")

    @property
    def use_risk_budget(self):
        return self.mode in ("risk-budget", "combined")

    def as_dict(self):
        return asdict(self)
