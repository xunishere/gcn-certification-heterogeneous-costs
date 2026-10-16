"""Target risk and fixed-mean, auditable risk-budget allocation.

The experiment-level allocator is the only risk-budget policy used by the real
pipeline.  It allocates all targets in a ``(dataset, seed, target_scope)`` set
together, so the integer budgets sum exactly to ``N * Q0``.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

RISK_BUDGET_POLICY = "fixed-mean-simplex-largest-remainder-v1"
COST_BUDGET_POLICY = "fixed-resource-no-mean-scaling-v1"


def target_risk(importance_t, confidence_t, eta=0.5):
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0,1]")
    if not 0.0 <= confidence_t <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    risk = eta * float(importance_t) + (1.0 - eta) * (1.0 - float(confidence_t))
    if not math.isfinite(risk):
        raise ValueError("risk must be finite")
    return risk


def certification_budget(q0, importance_t, confidence_t, eta=0.5, beta=1.0):
    """Return the deprecated independent-target budget used before schema v3.

    This helper remains only for regression tests and old callers.  It cannot
    conserve a target-set budget and is never called by the real experiment
    pipeline.  New code must use :func:`allocate_fixed_mean_budgets`.
    """
    if q0 < 0 or int(q0) != q0 or beta < 0:
        raise ValueError("q0 must be an integer and q0/beta non-negative")
    risk = target_risk(importance_t, confidence_t, eta)
    return int(q0 + math.floor(beta * risk))


def _project_to_nonnegative_simplex(scores, total):
    """Euclidean projection onto ``x >= 0, sum(x) = total``."""
    if total < 0:
        raise ValueError("simplex total must be non-negative")
    if not scores:
        if total:
            raise ValueError("a nonzero total requires at least one score")
        return []
    if total == 0:
        return [0.0] * len(scores)

    ordered = sorted((float(value) for value in scores), reverse=True)
    cumulative = 0.0
    active_count = 0
    active_sum = 0.0
    for rank, value in enumerate(ordered, 1):
        cumulative += value
        threshold = (cumulative - float(total)) / rank
        if value - threshold > 0.0:
            active_count = rank
            active_sum = cumulative
    if active_count == 0:  # total > 0 makes this mathematically unreachable.
        raise RuntimeError("simplex projection has no active coordinate")
    threshold = (active_sum - float(total)) / active_count
    projected = [max(float(value) - threshold, 0.0) for value in scores]
    if any(not math.isfinite(value) for value in projected):
        raise ValueError("projected budgets must be finite")
    return projected


def _canonical_target_risks(target_risks):
    items = (target_risks.items() if isinstance(target_risks, Mapping)
             else target_risks)
    canonical = []
    seen = set()
    for target, risk in items:
        if isinstance(target, bool) or int(target) != target:
            raise ValueError("target ids must be integers")
        target = int(target)
        risk = float(risk)
        if target in seen:
            raise ValueError(f"duplicate target id: {target}")
        if not math.isfinite(risk):
            raise ValueError("risks must be finite")
        seen.add(target)
        canonical.append((target, risk))
    canonical.sort(key=lambda item: item[0])
    return canonical


@dataclass(frozen=True)
class FixedMeanBudgetAllocation:
    policy: str
    q0: int
    beta: float
    target_risks: dict
    continuous_scores: dict
    projected_scores: dict
    budgets: dict
    checksum: str

    @property
    def target_count(self):
        return len(self.budgets)

    @property
    def total(self):
        return sum(self.budgets.values())

    @property
    def mean(self):
        return float(self.q0) if self.target_count else float("nan")

    @property
    def minimum(self):
        return min(self.budgets.values(), default=None)

    @property
    def maximum(self):
        return max(self.budgets.values(), default=None)

    @property
    def distribution(self):
        return dict(sorted(Counter(self.budgets.values()).items()))

    def as_manifest(self):
        records = []
        for target in sorted(self.budgets):
            records.append({
                "target": target,
                "risk": self.target_risks[target],
                "continuous_score": self.continuous_scores[target],
                "projected_score": self.projected_scores[target],
                "Q_t": self.budgets[target],
            })
        return {
            "policy": self.policy,
            "Q0": self.q0,
            "beta": self.beta,
            "target_count": self.target_count,
            "budget_total": self.total,
            "budget_mean": self.mean,
            "budget_min": self.minimum,
            "budget_max": self.maximum,
            "budget_distribution": {str(key): value
                                    for key, value in self.distribution.items()},
            "mapping_checksum_sha256": self.checksum,
            "records": records,
        }


def allocate_fixed_mean_budgets(target_risks, q0, beta=1.0):
    """Allocate monotone non-negative integer budgets with mean exactly ``Q0``.

    ``target_risks`` is a mapping or iterable of ``(target_id, risk)`` pairs.
    The continuous starting score is ``Q0 + beta * (r_t - mean(r))``.  Scores
    are projected onto the non-negative simplex with total ``N*Q0`` and then
    deterministically integerized by largest remainder.  Fractional ties are
    broken by increasing target id, making the result independent of input
    order.
    """
    if isinstance(q0, bool) or q0 < 0 or int(q0) != q0:
        raise ValueError("Q0 must be a non-negative integer")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    q0, beta = int(q0), float(beta)
    canonical = _canonical_target_risks(target_risks)
    target_count = len(canonical)
    if not canonical:
        payload = {
            "policy": RISK_BUDGET_POLICY, "Q0": q0, "beta": beta,
            "mapping": [],
        }
        checksum = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        return FixedMeanBudgetAllocation(
            RISK_BUDGET_POLICY, q0, beta, {}, {}, {}, {}, checksum
        )

    targets = [target for target, _ in canonical]
    risks = [risk for _, risk in canonical]
    risk_mean = math.fsum(risks) / target_count
    continuous = [q0 + beta * (risk - risk_mean) for risk in risks]
    total = target_count * q0

    # Handle exact degeneracies explicitly rather than letting floating-point
    # tie behavior distribute artificial +/-1 differences.
    if beta == 0.0 or all(risk == risks[0] for risk in risks):
        projected = [float(q0)] * target_count
        integer = [q0] * target_count
    else:
        projected = _project_to_nonnegative_simplex(continuous, total)
        floors = [int(math.floor(max(value, 0.0))) for value in projected]
        remaining = total - sum(floors)
        if not 0 <= remaining <= target_count:
            raise RuntimeError("invalid largest-remainder residual")
        order = sorted(
            range(target_count),
            key=lambda index: (-(projected[index] - floors[index]), targets[index]),
        )
        integer = floors[:]
        for index in order[:remaining]:
            integer[index] += 1

    budgets = dict(zip(targets, integer))
    if sum(integer) != total:
        raise RuntimeError("fixed-mean budget conservation failed")
    if any(value < 0 or int(value) != value for value in integer):
        raise RuntimeError("risk budgets must be non-negative integers")

    # Check the required weak monotonicity across strictly increasing risk.
    by_risk = sorted(zip(risks, targets, integer), key=lambda item: (item[0], item[1]))
    maximum_at_lower_risk = None
    previous_risk = None
    current_risk_maximum = None
    for risk, _, value in by_risk:
        if previous_risk is None or risk != previous_risk:
            if current_risk_maximum is not None:
                maximum_at_lower_risk = (current_risk_maximum if maximum_at_lower_risk is None
                                         else max(maximum_at_lower_risk,
                                                  current_risk_maximum))
            previous_risk = risk
            current_risk_maximum = value
        else:
            current_risk_maximum = max(current_risk_maximum, value)
        if maximum_at_lower_risk is not None and value < maximum_at_lower_risk:
            raise RuntimeError("integerized budgets are not risk-monotone")

    checksum_payload = {
        "policy": RISK_BUDGET_POLICY,
        "Q0": q0,
        "beta": beta,
        "mapping": [[target, budgets[target]] for target in targets],
    }
    checksum = hashlib.sha256(json.dumps(
        checksum_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return FixedMeanBudgetAllocation(
        policy=RISK_BUDGET_POLICY,
        q0=q0,
        beta=beta,
        target_risks=dict(canonical),
        continuous_scores=dict(zip(targets, continuous)),
        projected_scores=dict(zip(targets, projected)),
        budgets=budgets,
        checksum=checksum,
    )


def fixed_resource_budget(q0, *, use_risk, use_cost, allocated_q_t=None):
    """Build one target's fixed-resource count/cost budget record.

    Risk modes require a ``Q_t`` preallocated by
    :func:`allocate_fixed_mean_budgets`.  Cost modes use that same numeric
    resource limit directly: heterogeneous-cost uses ``B=Q0`` and combined
    uses ``B=Q_t``.  Candidate costs are deliberately absent from this API, so
    they cannot increase the attacker's budget.  Local ``q`` remains a count
    elsewhere in the pipeline.
    """
    if isinstance(q0, bool) or q0 < 0 or int(q0) != q0:
        raise ValueError("Q0 must be a non-negative integer")
    if use_risk:
        if (allocated_q_t is None or isinstance(allocated_q_t, bool)
                or allocated_q_t < 0 or int(allocated_q_t) != allocated_q_t):
            raise ValueError("risk modes require a preallocated non-negative integer Q_t")
        q_t = int(allocated_q_t)
    else:
        q_t = int(q0)
    q_equiv = q_t
    q_cost = float(q_t)
    return {
        "Q0": int(q0),
        "Q_t": q_t,
        "Q_equiv": q_equiv,
        "Q_cost": q_cost,
        "cost_budget_policy": COST_BUDGET_POLICY,
        "global_budget": q_cost if use_cost else float(q_equiv),
        "global_budget_unit": "cost" if use_cost else "count",
    }


def budgets_for_targets(q0, importance, confidence, eta=0.5, beta=1.0,
                        target_ids=None):
    """Convenience sequence API backed by the fixed-mean set allocator."""
    if len(importance) != len(confidence):
        raise ValueError("importance and confidence lengths differ")
    ids = list(range(len(importance))) if target_ids is None else list(target_ids)
    if len(ids) != len(importance):
        raise ValueError("target id and risk-input lengths differ")
    risks = [(target, target_risk(i, p, eta))
             for target, i, p in zip(ids, importance, confidence)]
    allocation = allocate_fixed_mean_budgets(risks, q0, beta)
    return [allocation.budgets[int(target)] for target in ids]
