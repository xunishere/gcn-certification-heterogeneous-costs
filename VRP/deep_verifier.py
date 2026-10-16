"""Sound LP verifier for depth-2/depth-3 sigmoid GCNs.

Small LPs may retain the continuous feature-flip variables explicitly. Real
LPs may instead use exact per-neuron grouped-knapsack first-layer bounds and a
sound Cartesian-product budget box. Sigmoids are enclosed by intersections of
sound affine lines. Dual adds slopes proposed by feasible Monte Carlo attacks,
but every added line is recalibrated over the complete outer interval and all
baseline constraints remain present.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import math
import time
from typing import Sequence

import numpy as np
import torch

from .deep_gcn import DeepSigmoidGCN, NORMALIZATION_VERSION
from .sigmoid_envelope import (
    EnvelopeSelection,
    calibrate_intercepts,
    outer_candidate_slopes,
    select_envelope,
    under_candidate_slopes,
    sigmoid,
)

VERIFIER_SCHEMA_VERSION = "deep-sigmoid-lp-v9"
CBC_RANDOM_SEED = 1
CBC_PRIMARY_PROFILE = "primary-scaled-presolve"
CBC_AUDIT_RETRY_PROFILE = "audit-retry-unscaled-no-presolve"
COUNT_MODES = frozenset(("legacy", "risk", "risk-budget"))
WEIGHTED_MODES = frozenset(("heterogeneous", "heterogeneous-cost", "combined"))


def _cbc_solver(pulp, *, time_limit: float | None, profile: str):
    """Build one deterministic CBC numerical profile for the same LP."""
    common = {
        "msg": False,
        "timeLimit": time_limit,
        "threads": 1,
    }
    if profile == CBC_PRIMARY_PROFILE:
        return pulp.PULP_CBC_CMD(
            **common,
            options=[
                f"randomSeed {CBC_RANDOM_SEED}",
                "primalTolerance 1e-9",
                "dualTolerance 1e-9",
            ],
        )
    if profile == CBC_AUDIT_RETRY_PROFILE:
        return pulp.PULP_CBC_CMD(
            **common,
            mip=False,
            presolve=False,
            options=[
                f"randomSeed {CBC_RANDOM_SEED}",
                "primalTolerance 1e-10",
                "dualTolerance 1e-10",
                "scaling off",
            ],
        )
    raise ValueError(f"unknown CBC numerical profile: {profile}")


def canonical_threat_mode(mode: str) -> str:
    mode = str(mode)
    aliases = {
        "risk": "risk-budget",
        "heterogeneous": "heterogeneous-cost",
    }
    mode = aliases.get(mode, mode)
    if mode not in COUNT_MODES | WEIGHTED_MODES:
        raise ValueError(f"unknown threat mode: {mode}")
    return mode


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    clipped = np.clip(flat, -745.0, 709.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    exponential = np.exp(clipped[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _adjacency_rows(adjacency: torch.Tensor) -> list[dict[int, float]]:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    size = int(adjacency.shape[0])
    rows: list[dict[int, float]] = [dict() for _ in range(size)]
    if adjacency.is_sparse:
        coalesced = adjacency.detach().cpu().coalesce()
        indices = coalesced.indices().numpy()
        values = coalesced.values().double().numpy()
        entries = zip(indices[0], indices[1], values)
    else:
        dense = adjacency.detach().cpu().double().numpy()
        entries = ((row, column, dense[row, column])
                   for row in range(size) for column in range(size)
                   if dense[row, column] != 0.0)
    for row, column, value in entries:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("normalized GCN adjacency must be finite and non-negative")
        if value > 0.0:
            rows[int(row)][int(column)] = value
    if any(not row for row in rows):
        raise ValueError("every normalized-adjacency row must include a self loop")
    return rows


@dataclass(frozen=True)
class ComputationGraph:
    target: int
    depth: int
    layer_nodes: tuple[tuple[int, ...], ...]
    adjacency_blocks: tuple[np.ndarray, ...]

    @property
    def candidate_sources(self) -> tuple[int, ...]:
        return self.layer_nodes[0]

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "depth": self.depth,
            "layer_nodes": [list(nodes) for nodes in self.layer_nodes],
            "layer_sizes": [len(nodes) for nodes in self.layer_nodes],
            "candidate_sources": list(self.candidate_sources),
            "normalization_version": NORMALIZATION_VERSION,
        }


def build_computation_graph(
    adjacency: torch.Tensor,
    target: int,
    depth: int,
) -> ComputationGraph:
    """Build the exact depth-hop target computation graph, including self."""
    if depth not in (2, 3):
        raise ValueError("depth must be 2 or 3")
    rows = _adjacency_rows(adjacency)
    if not 0 <= int(target) < len(rows):
        raise ValueError("target is outside the graph")
    layer_nodes: list[tuple[int, ...] | None] = [None] * (depth + 1)
    layer_nodes[depth] = (int(target),)
    for layer in range(depth - 1, -1, -1):
        sources = set()
        for destination in layer_nodes[layer + 1] or ():
            sources.update(rows[destination])
        layer_nodes[layer] = tuple(sorted(sources))
    concrete_nodes = tuple(nodes or () for nodes in layer_nodes)
    blocks = []
    for layer in range(1, depth + 1):
        previous = concrete_nodes[layer - 1]
        previous_index = {node: index for index, node in enumerate(previous)}
        block = np.zeros((len(concrete_nodes[layer]), len(previous)), dtype=np.float64)
        for row_index, destination in enumerate(concrete_nodes[layer]):
            for source, coefficient in rows[destination].items():
                if source in previous_index:
                    block[row_index, previous_index[source]] = coefficient
        if np.any(block < 0.0) or np.any(block.sum(axis=1) <= 0.0):
            raise RuntimeError("invalid computation-graph adjacency block")
        blocks.append(block)
    return ComputationGraph(
        target=int(target), depth=depth, layer_nodes=concrete_nodes,
        adjacency_blocks=tuple(blocks),
    )


def _validate_parameters(
    model: DeepSigmoidGCN,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if model.depth not in (2, 3):
        raise ValueError("verifier supports only depth 2 or 3")
    weights, biases = model.numpy_parameters()
    if len(weights) != model.depth or len(biases) != model.depth:
        raise RuntimeError("model parameter count does not match depth")
    for weight, bias in zip(weights, biases):
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError("model parameters must be finite")
        if weight.shape[1] != bias.shape[0]:
            raise RuntimeError("weight/bias dimensions disagree")
    return weights, biases


@dataclass(frozen=True)
class IntervalBounds:
    pre_lower: tuple[np.ndarray, ...]
    pre_upper: tuple[np.ndarray, ...]
    activation_lower: tuple[np.ndarray, ...]
    activation_upper: tuple[np.ndarray, ...]
    output_lower: np.ndarray
    output_upper: np.ndarray
    input_attack_relaxed: bool
    input_bound_method: str
    first_layer_bound_seconds: float

    def summary(self) -> dict:
        digest = hashlib.sha256()
        arrays = (self.pre_lower + self.pre_upper
                  + self.activation_lower + self.activation_upper
                  + (self.output_lower, self.output_upper))
        for array in arrays:
            contiguous = np.asarray(array, dtype=np.float64).copy(order="C")
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return {
            "bounds_sha256": digest.hexdigest(),
            "input_attack_relaxed": self.input_attack_relaxed,
            "input_bound_method": self.input_bound_method,
            "hidden_layers": [
                {
                    "shape": list(lower.shape),
                    "pre_min": float(lower.min()),
                    "pre_max": float(upper.max()),
                    "max_width": float((upper - lower).max()),
                }
                for lower, upper in zip(self.pre_lower, self.pre_upper)
            ],
            "output_lower": self.output_lower.tolist(),
            "output_upper": self.output_upper.tolist(),
        }


def _roundoff_padding(magnitude: np.ndarray, operations: int) -> np.ndarray:
    """Conservative binary64 dot-product/addition roundoff allowance."""
    epsilon = np.finfo(np.float64).eps
    scaled = min(0.25, 16.0 * max(1, operations) * epsilon)
    return scaled * np.maximum(1.0, np.asarray(magnitude, dtype=np.float64))


def _nonnegative_interval_product(
    matrix: np.ndarray,
    value_lower: np.ndarray,
    value_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = matrix @ value_lower
    upper = matrix @ value_upper
    maximum_absolute = np.maximum(np.abs(value_lower), np.abs(value_upper))
    magnitude = np.abs(matrix) @ maximum_absolute
    padding = _roundoff_padding(magnitude, 2 * matrix.shape[1] + 1)
    return (np.nextafter(lower - padding, -np.inf),
            np.nextafter(upper + padding, np.inf))


def _affine_interval(
    value_lower: np.ndarray,
    value_upper: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    lower = value_lower @ positive + value_upper @ negative + bias
    upper = value_upper @ positive + value_lower @ negative + bias
    maximum_absolute = np.maximum(np.abs(value_lower), np.abs(value_upper))
    magnitude = maximum_absolute @ np.abs(weight) + np.abs(bias)
    padding = _roundoff_padding(magnitude, 2 * weight.shape[0] + 2)
    # Explicit error padding plus nextafter protects bounds from BLAS summation
    # and final-addition roundoff, including exact clean/degenerate domains.
    lower = np.nextafter(lower - padding, -np.inf)
    upper = np.nextafter(upper + padding, np.inf)
    return lower, upper


def _largest_positive(values: np.ndarray, limit: int) -> np.ndarray:
    """Return up to ``limit`` positive values in descending order."""
    values = np.asarray(values, dtype=np.float64)
    limit = min(max(0, int(limit)), int(values.size))
    if limit == 0:
        return np.empty(0, dtype=np.float64)
    positive = values[values > 0.0]
    if positive.size > limit:
        positive = positive[np.argpartition(positive, -limit)[-limit:]]
    return np.sort(positive)[::-1]


def grouped_fractional_knapsack_max(
    per_source_benefits: Sequence[Sequence[float]],
    source_costs: Sequence[float],
    *,
    q: int,
    global_budget: float,
    weighted: bool,
) -> float:
    """Exact maximum for a grouped continuous count/weighted knapsack.

    Each source owns unit-capacity items, has the local cardinality cap ``q``,
    and all items of one source have the same positive cost. Non-positive
    benefits can be discarded. Therefore only that source's largest ``q``
    benefits can be active, after which ordinary benefit/cost-sorted
    fractional knapsack is exact for the LP attack domain.
    """
    if isinstance(q, bool) or int(q) != q or q < 0:
        raise ValueError("q must be a non-negative integer")
    q = int(q)
    budget = float(global_budget)
    costs = np.asarray(source_costs, dtype=np.float64)
    if (not math.isfinite(budget) or budget < 0.0
            or costs.shape != (len(per_source_benefits),)
            or np.any(~np.isfinite(costs)) or np.any(costs <= 0.0)):
        raise ValueError("invalid grouped-knapsack budget/costs")
    if q == 0 or budget == 0.0:
        return 0.0
    items = []
    for source, benefits in enumerate(per_source_benefits):
        selected = _largest_positive(np.asarray(benefits, dtype=np.float64), q)
        cost = float(costs[source]) if weighted else 1.0
        for benefit in selected:
            items.append((float(benefit) / cost, float(benefit), cost))
    items.sort(key=lambda item: (-item[0], -item[1], item[2]))
    remaining = budget
    objective = 0.0
    for _, benefit, cost in items:
        if remaining <= 0.0:
            break
        amount = min(1.0, remaining / cost)
        objective += amount * benefit
        remaining -= amount * cost
    # The exact real-arithmetic value is padded outward by several ULPs so the
    # quantity remains an upper bound after sorting/multiplication/summation.
    padding = float(_roundoff_padding(
        np.asarray(objective), max(1, 4 * len(items) + 2)
    ))
    return float(np.nextafter(objective + padding, math.inf))


def budget_aware_first_layer_bounds(
    graph: ComputationGraph,
    clean_x: np.ndarray,
    first_weight: np.ndarray,
    first_bias: np.ndarray,
    *,
    q: int,
    global_budget: float,
    source_costs: Sequence[float],
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sound tight per-neuron first-layer bounds under both attack budgets.

    For each first-layer preactivation, the feature-flip expression is affine
    in ``z``. Its maximum increase and decrease are independent grouped
    continuous knapsacks with a local count cap and either a global count or a
    weighted global cap. Solving each scalar objective exactly retains every
    attack; taking the Cartesian product across neurons only drops
    correlations and is therefore a sound outer abstraction.
    """
    started = time.perf_counter()
    q = int(q)
    budget = float(global_budget)
    source_nodes = np.asarray(graph.candidate_sources, dtype=int)
    local_x = np.asarray(clean_x, dtype=np.float64)[source_nodes]
    weight = np.asarray(first_weight, dtype=np.float64)
    bias = np.asarray(first_bias, dtype=np.float64)
    costs = np.asarray(source_costs, dtype=np.float64)
    if local_x.shape[1] != weight.shape[0]:
        raise ValueError("first-layer feature dimensions disagree")
    if costs.shape != (len(source_nodes),):
        raise ValueError("source costs do not match computation graph")
    block = np.asarray(graph.adjacency_blocks[0], dtype=np.float64)
    clean = block @ local_x @ weight + bias
    lower = clean.copy()
    upper = clean.copy()
    attack_possible = q > 0 and budget > 0.0
    if attack_possible:
        hidden = weight.shape[1]
        positive: list[list[np.ndarray]] = []
        negative: list[list[np.ndarray]] = []
        for source_position in range(len(source_nodes)):
            base = ((1.0 - 2.0 * local_x[source_position])[:, None] * weight)
            positive.append([
                _largest_positive(base[:, unit], q) for unit in range(hidden)
            ])
            negative.append([
                _largest_positive(-base[:, unit], q) for unit in range(hidden)
            ])
        for destination in range(block.shape[0]):
            neighbors = np.flatnonzero(block[destination] > 0.0)
            neighbor_costs = costs[neighbors]
            for unit in range(hidden):
                gains = [positive[source][unit] * block[destination, source]
                         for source in neighbors]
                losses = [negative[source][unit] * block[destination, source]
                          for source in neighbors]
                increase = grouped_fractional_knapsack_max(
                    gains, neighbor_costs, q=q, global_budget=budget,
                    weighted=weighted,
                )
                decrease = grouped_fractional_knapsack_max(
                    losses, neighbor_costs, q=q, global_budget=budget,
                    weighted=weighted,
                )
                magnitude = abs(clean[destination, unit]) + increase + decrease
                padding = float(_roundoff_padding(
                    np.asarray(magnitude),
                    4 * max(1, len(neighbors) * min(q, weight.shape[0])) + 4,
                ))
                lower[destination, unit] = np.nextafter(
                    clean[destination, unit] - decrease - padding, -math.inf
                )
                upper[destination, unit] = np.nextafter(
                    clean[destination, unit] + increase + padding, math.inf
                )
    else:
        padding = _roundoff_padding(
            np.abs(clean), 2 * max(1, block.shape[1] * weight.shape[0]) + 2
        )
        lower = np.nextafter(clean - padding, -np.inf)
        upper = np.nextafter(clean + padding, np.inf)
    elapsed = time.perf_counter() - started
    summary = {
        "method": "per-neuron-exact-grouped-fractional-knapsack-v1",
        "weighted": bool(weighted),
        "q": q,
        "global_budget": budget,
        "source_count": len(source_nodes),
        "feature_count": int(weight.shape[0]),
        "first_layer_node_count": int(block.shape[0]),
        "first_layer_hidden_count": int(weight.shape[1]),
        "potential_input_variables": int(local_x.size),
        "seconds": elapsed,
        "maximum_width": float((upper - lower).max()),
    }
    return lower, upper, summary


def interval_propagation(
    graph: ComputationGraph,
    clean_x: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    *,
    attack_possible: bool,
    first_layer_override: tuple[np.ndarray, np.ndarray] | None = None,
    input_bound_method: str = "independent-feature-box",
    first_layer_bound_seconds: float = 0.0,
) -> IntervalBounds:
    """Sound interval propagation through non-negative ``Ahat`` and signed W.

    By default every attackable input is independently placed in ``[0,1]``.
    A caller may instead supply a sound first-layer preactivation box, notably
    the exact per-neuron grouped-knapsack bounds from
    :func:`budget_aware_first_layer_bounds`. Every later layer uses ordinary
    sound interval propagation with explicit numerical padding.
    """
    clean_x = np.asarray(clean_x, dtype=np.float64)
    source_x = clean_x[np.asarray(graph.layer_nodes[0], dtype=int)]
    if not np.all((source_x == 0.0) | (source_x == 1.0)):
        raise ValueError("clean node features must be binary")
    previous_lower = np.zeros_like(source_x) if attack_possible else source_x.copy()
    previous_upper = np.ones_like(source_x) if attack_possible else source_x.copy()
    pre_lower, pre_upper = [], []
    activation_lower, activation_upper = [], []
    output_lower = output_upper = None

    for layer in range(1, graph.depth + 1):
        block = graph.adjacency_blocks[layer - 1]
        if np.any(block < 0.0):
            raise ValueError("interval propagation requires non-negative Ahat")
        if layer == 1 and first_layer_override is not None:
            current_lower = np.asarray(first_layer_override[0], dtype=np.float64)
            current_upper = np.asarray(first_layer_override[1], dtype=np.float64)
            expected = (len(graph.layer_nodes[1]), weights[0].shape[1])
            if current_lower.shape != expected or current_upper.shape != expected:
                raise ValueError("first-layer override has the wrong shape")
        else:
            aggregate_lower, aggregate_upper = _nonnegative_interval_product(
                block, previous_lower, previous_upper
            )
            current_lower, current_upper = _affine_interval(
                aggregate_lower, aggregate_upper,
                np.asarray(weights[layer - 1], dtype=np.float64),
                np.asarray(biases[layer - 1], dtype=np.float64),
            )
        if np.any(~np.isfinite(current_lower)) or np.any(~np.isfinite(current_upper)):
            raise RuntimeError("interval propagation produced non-finite bounds")
        if np.any(current_lower > current_upper):
            raise RuntimeError("interval propagation produced inverted bounds")
        if layer < graph.depth:
            pre_lower.append(current_lower)
            pre_upper.append(current_upper)
            previous_lower = np.nextafter(_sigmoid_array(current_lower), -np.inf)
            previous_upper = np.nextafter(_sigmoid_array(current_upper), np.inf)
            previous_lower = np.maximum(previous_lower, 0.0)
            previous_upper = np.minimum(previous_upper, 1.0)
            activation_lower.append(previous_lower)
            activation_upper.append(previous_upper)
        else:
            output_lower, output_upper = current_lower[0], current_upper[0]
    if output_lower is None or output_upper is None:
        raise RuntimeError("output interval was not computed")
    return IntervalBounds(
        pre_lower=tuple(pre_lower), pre_upper=tuple(pre_upper),
        activation_lower=tuple(activation_lower),
        activation_upper=tuple(activation_upper),
        output_lower=output_lower, output_upper=output_upper,
        input_attack_relaxed=bool(attack_possible),
        input_bound_method=str(input_bound_method),
        first_layer_bound_seconds=float(first_layer_bound_seconds),
    )


def evaluate_computation_graph(
    graph: ComputationGraph,
    local_inputs: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Evaluate the exact local graph and return target logits/preactivations."""
    previous = np.asarray(local_inputs, dtype=np.float64)
    expected = (len(graph.layer_nodes[0]), weights[0].shape[0])
    if previous.shape != expected:
        raise ValueError(f"local input shape {previous.shape} != {expected}")
    hidden_pre = []
    for layer in range(1, graph.depth + 1):
        previous = (graph.adjacency_blocks[layer - 1] @ previous
                    @ np.asarray(weights[layer - 1], dtype=np.float64)
                    + np.asarray(biases[layer - 1], dtype=np.float64))
        if layer < graph.depth:
            hidden_pre.append(previous.copy())
            previous = _sigmoid_array(previous)
    return previous[0].copy(), tuple(hidden_pre)


@dataclass(frozen=True)
class MonteCarloDomain:
    preactivation_samples: tuple[np.ndarray, ...]
    requested_attack_samples: int
    generated_attack_samples: int
    unique_attack_samples: int
    maximum_flips: int
    weighted: bool

    @property
    def total_samples_with_clean(self) -> int:
        if not self.preactivation_samples:
            return 0
        return int(self.preactivation_samples[0].shape[0])

    def summary(self) -> dict:
        return {
            "requested_attack_samples": self.requested_attack_samples,
            "generated_attack_samples": self.generated_attack_samples,
            "unique_attack_samples": self.unique_attack_samples,
            "total_samples_with_clean": self.total_samples_with_clean,
            "maximum_flips": self.maximum_flips,
            "weighted": self.weighted,
        }


def _sample_attack_masks(
    source_count: int,
    feature_count: int,
    *,
    q: int,
    global_budget: float,
    source_costs: Sequence[float],
    weighted: bool,
    samples: int,
    seed: int,
) -> list[tuple[tuple[int, int], ...]]:
    """Generate sparse feasible integer attacks; item zero is always clean.

    Real Planetoid receptive fields contain up to millions of potential input
    variables. Enumerating or permuting that Cartesian product merely to draw
    a few attacks is unnecessary. This sampler chooses sources/features
    directly and stores only the selected flips, while checking both budgets.
    """
    if samples < 0:
        raise ValueError("samples must be non-negative")
    attacks: list[tuple[tuple[int, int], ...]] = [tuple()]
    if samples == 0 or q == 0 or global_budget <= 0.0:
        return attacks
    costs = np.asarray(source_costs, dtype=np.float64)
    if costs.shape != (source_count,) or np.any(~np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("one finite positive cost is required per candidate source")
    item_costs = costs if weighted else np.ones(source_count, dtype=np.float64)
    initially_feasible = np.flatnonzero(item_costs <= global_budget + 1e-12)
    if initially_feasible.size == 0:
        return attacks
    generator = np.random.default_rng(int(seed))
    absolute_maximum = min(
        source_count * min(q, feature_count),
        int(math.floor(global_budget / float(item_costs.min()) + 1e-12)),
    )
    for sample_index in range(samples):
        local_counts = np.zeros(source_count, dtype=np.int32)
        selected: list[set[int]] = [set() for _ in range(source_count)]
        spent = 0.0
        fill = 1.0 if sample_index == 0 else float(generator.uniform(0.1, 1.0))
        desired = max(1, int(math.ceil(fill * max(1, absolute_maximum))))
        flips: list[tuple[int, int]] = []
        while len(flips) < desired:
            eligible = [
                source for source in range(source_count)
                if local_counts[source] < min(q, feature_count)
                and spent + float(item_costs[source]) <= global_budget + 1e-12
            ]
            if not eligible:
                break
            source = int(eligible[int(generator.integers(len(eligible)))])
            if len(selected[source]) >= feature_count:
                local_counts[source] = min(q, feature_count)
                continue
            feature = int(generator.integers(feature_count))
            while feature in selected[source]:
                feature = int(generator.integers(feature_count))
            selected[source].add(feature)
            local_counts[source] += 1
            spent += float(item_costs[source])
            flips.append((source, feature))
        attacks.append(tuple(sorted(flips)))
    return attacks


def monte_carlo_under_domain(
    graph: ComputationGraph,
    clean_x: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    *,
    q: int,
    global_budget: float,
    source_costs: Sequence[float],
    weighted: bool,
    samples: int,
    seed: int,
) -> MonteCarloDomain:
    source_nodes = np.asarray(graph.candidate_sources, dtype=int)
    local_clean = np.asarray(clean_x, dtype=np.float64)[source_nodes]
    attacks = _sample_attack_masks(
        len(source_nodes), local_clean.shape[1], q=q,
        global_budget=global_budget, source_costs=source_costs,
        weighted=weighted, samples=samples, seed=seed,
    )
    layer_samples: list[list[np.ndarray]] = [[] for _ in range(graph.depth - 1)]
    maximum_flips = 0
    fingerprints = set()
    first_weight = np.asarray(weights[0], dtype=np.float64)
    clean_first = (graph.adjacency_blocks[0] @ local_clean @ first_weight
                   + np.asarray(biases[0], dtype=np.float64))
    first_block = graph.adjacency_blocks[0]
    for flips in attacks:
        maximum_flips = max(maximum_flips, len(flips))
        fingerprints.add(flips)
        first = clean_first.copy()
        for source, feature in flips:
            delta_x = 1.0 - 2.0 * local_clean[source, feature]
            first += (first_block[:, source, None] * delta_x
                      * first_weight[feature][None, :])
        hidden_pre = [first]
        previous = _sigmoid_array(first)
        for layer in range(2, graph.depth):
            current = (graph.adjacency_blocks[layer - 1] @ previous
                       @ np.asarray(weights[layer - 1], dtype=np.float64)
                       + np.asarray(biases[layer - 1], dtype=np.float64))
            hidden_pre.append(current)
            previous = _sigmoid_array(current)
        for layer, values in enumerate(hidden_pre):
            layer_samples[layer].append(values)
    return MonteCarloDomain(
        preactivation_samples=tuple(np.stack(values, axis=0)
                                    for values in layer_samples),
        requested_attack_samples=int(samples),
        generated_attack_samples=max(0, len(attacks) - 1),
        unique_attack_samples=max(0, len(fingerprints) - 1),
        maximum_flips=maximum_flips,
        weighted=bool(weighted),
    )


def select_network_envelopes(
    bounds: IntervalBounds,
    *,
    method: str,
    under_domain: MonteCarloDomain | None,
) -> tuple[tuple[tuple[EnvelopeSelection, ...], ...], ...]:
    selected_layers = []
    for layer, (lower, upper) in enumerate(zip(bounds.pre_lower, bounds.pre_upper)):
        rows = []
        for node_index in range(lower.shape[0]):
            neurons = []
            for feature_index in range(lower.shape[1]):
                under_values = None
                if under_domain is not None:
                    under_values = under_domain.preactivation_samples[
                        layer
                    ][:, node_index, feature_index].tolist()
                neurons.append(select_envelope(
                    float(lower[node_index, feature_index]),
                    float(upper[node_index, feature_index]),
                    method=method,
                    under_values=under_values,
                ))
            rows.append(tuple(neurons))
        selected_layers.append(tuple(rows))
    return tuple(selected_layers)


def envelope_diagnostics(
    selections: tuple[tuple[tuple[EnvelopeSelection, ...], ...], ...],
    hidden_layer_nodes: Sequence[Sequence[int]] | None = None,
) -> tuple[dict, list]:
    lower_sources, upper_sources = Counter(), Counter()
    lower_gaps, upper_gaps = [], []
    fallback, used_under = Counter(), 0
    details = []
    for layer, rows in enumerate(selections, start=1):
        layer_details = []
        for node_index, neurons in enumerate(rows):
            neuron_details = []
            for feature_index, selection in enumerate(neurons):
                lower_sources[selection.lower.source] += 1
                upper_sources[selection.upper.source] += 1
                lower_gaps.append(selection.lower.selection_mean_gap)
                upper_gaps.append(selection.upper.selection_mean_gap)
                if selection.used_under_domain:
                    used_under += 1
                if selection.fallback_reason:
                    fallback[selection.fallback_reason] += 1
                neuron_details.append({
                    "feature_index": feature_index,
                    **selection.as_dict(),
                })
            layer_details.append({
                "node_index": node_index,
                "graph_node": (None if hidden_layer_nodes is None
                               else int(hidden_layer_nodes[layer - 1][node_index])),
                "neurons": neuron_details,
            })
        details.append({"hidden_layer": layer, "nodes": layer_details})
    count = len(lower_gaps)
    summary = {
        "neuron_count": count,
        "used_under_neurons": used_under,
        "fallback_counts": dict(sorted(fallback.items())),
        "lower_source_counts": dict(sorted(lower_sources.items())),
        "upper_source_counts": dict(sorted(upper_sources.items())),
        "mean_selected_lower_gap": (math.fsum(lower_gaps) / count if count else 0.0),
        "mean_selected_upper_gap": (math.fsum(upper_gaps) / count if count else 0.0),
        "max_selected_lower_gap": max(lower_gaps, default=0.0),
        "max_selected_upper_gap": max(upper_gaps, default=0.0),
    }
    return summary, details


@dataclass(frozen=True)
class ExactEnumerationResult:
    target: int
    prediction: int
    clean_margin: float
    worst_margin: float
    robust: bool
    attacks_evaluated: int
    worst_mask: tuple[tuple[int, ...], ...]
    rival_worst_margins: dict[int, float]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["rival_worst_margins"] = {
            str(key): value for key, value in self.rival_worst_margins.items()
        }
        return payload


def exhaustive_worst_margin(
    model: DeepSigmoidGCN,
    adjacency: torch.Tensor,
    clean_x: torch.Tensor | np.ndarray,
    target: int,
    *,
    mode: str,
    q: int,
    global_budget: float,
    node_costs: Sequence[float] | None = None,
    max_variables: int = 22,
) -> ExactEnumerationResult:
    """Enumerate every budget-feasible binary attack on a tiny target graph."""
    mode = canonical_threat_mode(mode)
    weighted = mode in WEIGHTED_MODES
    if q < 0 or int(q) != q or global_budget < 0:
        raise ValueError("q/global budget must be non-negative")
    q = int(q)
    graph = build_computation_graph(adjacency, target, model.depth)
    weights, biases = _validate_parameters(model)
    x = (clean_x.detach().cpu().double().numpy() if torch.is_tensor(clean_x)
         else np.asarray(clean_x, dtype=np.float64))
    local_clean = x[np.asarray(graph.candidate_sources, dtype=int)]
    variable_count = int(local_clean.size)
    if variable_count > max_variables:
        raise ValueError(
            f"exhaustive enumeration has {variable_count} variables; "
            f"limit is {max_variables}"
        )
    if node_costs is None:
        all_costs = np.ones(x.shape[0], dtype=np.float64)
    else:
        all_costs = np.asarray(node_costs, dtype=np.float64)
        if all_costs.shape != (x.shape[0],):
            raise ValueError("node_costs must cover every graph node")
    source_costs = all_costs[np.asarray(graph.candidate_sources, dtype=int)]
    if np.any(source_costs <= 0.0) or np.any(~np.isfinite(source_costs)):
        raise ValueError("source costs must be finite and positive")

    clean_logits, _ = evaluate_computation_graph(graph, local_clean, weights, biases)
    prediction = int(np.argmax(clean_logits))
    rivals = [index for index in range(len(clean_logits)) if index != prediction]
    clean_margin = min(float(clean_logits[prediction] - clean_logits[rival])
                       for rival in rivals)
    per_source_options = []
    feature_count = local_clean.shape[1]
    for source, source_cost in enumerate(source_costs):
        options = []
        for count in range(min(q, feature_count) + 1):
            for feature_subset in itertools.combinations(range(feature_count), count):
                cost = count * (float(source_cost) if weighted else 1.0)
                if cost <= global_budget + 1e-12:
                    options.append((feature_subset, cost))
        per_source_options.append(options)

    worst_margin = math.inf
    worst_mask = np.zeros_like(local_clean, dtype=np.uint8)
    rival_worst = {rival: math.inf for rival in rivals}
    attacks_evaluated = 0
    for source_choices in itertools.product(*per_source_options):
        spent = math.fsum(choice[1] for choice in source_choices)
        if spent > global_budget + 1e-12:
            continue
        mask = np.zeros_like(local_clean, dtype=np.uint8)
        for source, (features, _) in enumerate(source_choices):
            if features:
                mask[source, list(features)] = 1
        attacked = local_clean + (1.0 - 2.0 * local_clean) * mask
        logits, _ = evaluate_computation_graph(graph, attacked, weights, biases)
        margins = {rival: float(logits[prediction] - logits[rival])
                   for rival in rivals}
        for rival, margin in margins.items():
            rival_worst[rival] = min(rival_worst[rival], margin)
        margin = min(margins.values())
        attacks_evaluated += 1
        if margin < worst_margin:
            worst_margin = margin
            worst_mask = mask.copy()
    if attacks_evaluated == 0:
        raise RuntimeError("clean attack should always be feasible")
    return ExactEnumerationResult(
        target=int(target), prediction=prediction, clean_margin=clean_margin,
        worst_margin=worst_margin, robust=worst_margin > 0.0,
        attacks_evaluated=attacks_evaluated,
        worst_mask=tuple(tuple(int(value) for value in row) for row in worst_mask),
        rival_worst_margins=rival_worst,
    )


@dataclass(frozen=True)
class ThreatSpecification:
    mode: str
    q: int
    global_budget: float
    source_costs: tuple[float, ...]

    @property
    def weighted(self) -> bool:
        return self.mode in WEIGHTED_MODES


@dataclass
class _LPArtifacts:
    model: object
    logits: list
    attack_variables: list[list]
    variable_count: int
    constraint_count: int
    input_abstraction: str
    potential_input_variable_count: int


def _build_lp(
    graph: ComputationGraph,
    clean_x: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    bounds: IntervalBounds,
    selections: tuple[tuple[tuple[EnvelopeSelection, ...], ...], ...],
    threat: ThreatSpecification,
    *,
    input_abstraction: str,
) -> _LPArtifacts:
    try:
        import pulp
    except ImportError as error:  # pragma: no cover - dependency is pinned.
        raise RuntimeError("PuLP/CBC is required for deep certification") from error

    problem = pulp.LpProblem(
        f"deep_sigmoid_target_{graph.target}", pulp.LpMinimize
    )
    source_nodes = graph.candidate_sources
    feature_count = weights[0].shape[0]
    local_clean = clean_x[np.asarray(source_nodes, dtype=int)]
    potential_input_variables = len(source_nodes) * feature_count
    if input_abstraction not in ("explicit", "budget-box"):
        raise ValueError("input_abstraction must be explicit or budget-box")
    if input_abstraction == "explicit":
        attacks = [[
            pulp.LpVariable(f"z_n{node}_f{feature}", lowBound=0.0, upBound=1.0)
            for feature in range(feature_count)
        ] for node in source_nodes]
        for row, node in zip(attacks, source_nodes):
            problem += (pulp.lpSum(row) <= threat.q, f"local_count_n{node}")
        if threat.weighted:
            problem += (
                pulp.lpSum(threat.source_costs[source] * variable
                           for source, row in enumerate(attacks) for variable in row)
                <= threat.global_budget,
                "global_weighted_cost",
            )
        else:
            problem += (
                pulp.lpSum(variable for row in attacks for variable in row)
                <= threat.global_budget,
                "global_count",
            )
    else:
        # Every feasible attack is already enclosed by each first-layer
        # grouped-knapsack interval. Freeing those preactivations within their
        # Cartesian-product box drops correlations but never drops an attack.
        attacks = []

    previous_activations = None
    for layer in range(1, graph.depth):
        nodes = graph.layer_nodes[layer]
        dimension = weights[layer - 1].shape[1]
        preactivation = [[
            pulp.LpVariable(
                f"p_l{layer}_n{node}_h{hidden}",
                lowBound=float(bounds.pre_lower[layer - 1][node_position, hidden]),
                upBound=float(bounds.pre_upper[layer - 1][node_position, hidden]),
            )
            for hidden in range(dimension)
        ] for node_position, node in enumerate(nodes)]
        activation = [[
            pulp.LpVariable(
                f"h_l{layer}_n{node}_h{hidden}",
                lowBound=float(bounds.activation_lower[layer - 1][node_position, hidden]),
                upBound=float(bounds.activation_upper[layer - 1][node_position, hidden]),
            )
            for hidden in range(dimension)
        ] for node_position, node in enumerate(nodes)]

        block = graph.adjacency_blocks[layer - 1]
        if layer == 1 and input_abstraction == "explicit":
            for destination_index, node in enumerate(nodes):
                for hidden in range(dimension):
                    constant = float(biases[0][hidden])
                    terms = []
                    for source_position, source_node in enumerate(source_nodes):
                        coefficient_a = float(block[destination_index, source_position])
                        if coefficient_a == 0.0:
                            continue
                        for feature in range(feature_count):
                            coefficient_w = float(weights[0][feature, hidden])
                            constant += (coefficient_a
                                         * float(local_clean[source_position, feature])
                                         * coefficient_w)
                            delta = (coefficient_a
                                     * (1.0 - 2.0 * float(local_clean[source_position, feature]))
                                     * coefficient_w)
                            if delta != 0.0:
                                terms.append(delta * attacks[source_position][feature])
                    problem += (
                        preactivation[destination_index][hidden]
                        == constant + pulp.lpSum(terms),
                        f"preactivation_l{layer}_n{node}_h{hidden}",
                    )
        elif layer > 1:
            previous_nodes = graph.layer_nodes[layer - 1]
            for destination_index, node in enumerate(nodes):
                for hidden in range(dimension):
                    terms = []
                    for source_position, _ in enumerate(previous_nodes):
                        coefficient_a = float(block[destination_index, source_position])
                        if coefficient_a == 0.0:
                            continue
                        for prior_hidden in range(weights[layer - 1].shape[0]):
                            coefficient = (coefficient_a
                                           * float(weights[layer - 1][prior_hidden, hidden]))
                            if coefficient != 0.0:
                                terms.append(
                                    coefficient
                                    * previous_activations[source_position][prior_hidden]
                                )
                    problem += (
                        preactivation[destination_index][hidden]
                        == float(biases[layer - 1][hidden]) + pulp.lpSum(terms),
                        f"preactivation_l{layer}_n{node}_h{hidden}",
                    )

        for node_position, node in enumerate(nodes):
            for hidden in range(dimension):
                selection = selections[layer - 1][node_position][hidden]
                # Baseline is the intersection of every sound line generated
                # solely from the outer interval. This remains an LP and is
                # strictly stronger than silently choosing only one line.
                lower_lines: list[tuple[float, float]] = []
                upper_lines: list[tuple[float, float]] = []
                outer_slopes = [candidate.slope for candidate in
                                outer_candidate_slopes(
                                    selection.outer_lower,
                                    selection.outer_upper,
                                )]
                # Intersect every analytic outer-only candidate. This compact
                # set avoids the prohibitive constraint count of a dense slope
                # grid on large depth-3 receptive fields.
                outer_slopes = list(dict.fromkeys(outer_slopes))
                for slope in outer_slopes:
                    calibrated = calibrate_intercepts(
                        selection.outer_lower, selection.outer_upper, slope
                    )
                    lower_lines.append((slope, calibrated.lower_intercept))
                    upper_lines.append((slope, calibrated.upper_intercept))
                if selection.method == "dual":
                    # Add every slope recoverable from the sampled under range,
                    # plus the quantile-selected line, to all baseline lines.
                    # Each intercept is independently recalibrated on the same
                    # complete outer interval, never on samples.
                    if (selection.used_under_domain
                            and selection.under_lower is not None
                            and selection.under_upper is not None):
                        for candidate in under_candidate_slopes(
                                selection.under_lower, selection.under_upper,
                                [selection.under_lower, selection.under_upper]):
                            calibrated = calibrate_intercepts(
                                selection.outer_lower, selection.outer_upper,
                                candidate.slope,
                            )
                            lower_lines.append((candidate.slope,
                                                calibrated.lower_intercept))
                            upper_lines.append((candidate.slope,
                                                calibrated.upper_intercept))
                    lower_lines.append((selection.lower.slope,
                                        selection.lower.intercept))
                    upper_lines.append((selection.upper.slope,
                                        selection.upper.intercept))
                lower_lines = list(dict.fromkeys(lower_lines))
                upper_lines = list(dict.fromkeys(upper_lines))
                for line_index, (slope, intercept) in enumerate(lower_lines):
                    problem += (
                        activation[node_position][hidden]
                        >= slope * preactivation[node_position][hidden] + intercept,
                        f"sigmoid_lower_l{layer}_n{node}_h{hidden}_k{line_index}",
                    )
                for line_index, (slope, intercept) in enumerate(upper_lines):
                    problem += (
                        activation[node_position][hidden]
                        <= slope * preactivation[node_position][hidden] + intercept,
                        f"sigmoid_upper_l{layer}_n{node}_h{hidden}_k{line_index}",
                    )
        previous_activations = activation

    output_weight = weights[-1]
    output_bias = biases[-1]
    output_block = graph.adjacency_blocks[-1]
    logits = [
        pulp.LpVariable(
            f"logit_target{graph.target}_class{class_index}",
            lowBound=float(bounds.output_lower[class_index]),
            upBound=float(bounds.output_upper[class_index]),
        )
        for class_index in range(output_weight.shape[1])
    ]
    previous_nodes = graph.layer_nodes[-2]
    for class_index, logit in enumerate(logits):
        terms = []
        for source_position, _ in enumerate(previous_nodes):
            coefficient_a = float(output_block[0, source_position])
            if coefficient_a == 0.0:
                continue
            for hidden in range(output_weight.shape[0]):
                coefficient = coefficient_a * float(output_weight[hidden, class_index])
                if coefficient != 0.0:
                    terms.append(coefficient * previous_activations[source_position][hidden])
        problem += (
            logit == float(output_bias[class_index]) + pulp.lpSum(terms),
            f"output_class{class_index}",
        )
    # PuLP does not count variables until expressions/constraints are attached.
    return _LPArtifacts(
        model=problem,
        logits=logits,
        attack_variables=attacks,
        variable_count=len(problem.variables()),
        constraint_count=len(problem.constraints),
        input_abstraction=input_abstraction,
        potential_input_variable_count=potential_input_variables,
    )


def _audit_solution(problem, tolerance: float) -> tuple[bool, float, str | None]:
    try:
        import pulp
    except ImportError:  # pragma: no cover
        return False, math.inf, "PuLP unavailable"
    maximum_violation = 0.0
    for variable in problem.variables():
        value = variable.varValue
        if value is None or not math.isfinite(float(value)):
            return False, math.inf, f"non-finite variable {variable.name}"
        value = float(value)
        if variable.lowBound is not None:
            maximum_violation = max(maximum_violation,
                                    float(variable.lowBound) - value)
        if variable.upBound is not None:
            maximum_violation = max(maximum_violation,
                                    value - float(variable.upBound))
    for name, constraint in problem.constraints.items():
        residual = constraint.value()
        if residual is None or not math.isfinite(float(residual)):
            return False, math.inf, f"non-finite constraint {name}"
        residual = float(residual)
        if constraint.sense == pulp.LpConstraintEQ:
            violation = abs(residual)
        elif constraint.sense == pulp.LpConstraintLE:
            violation = max(0.0, residual)
        elif constraint.sense == pulp.LpConstraintGE:
            violation = max(0.0, -residual)
        else:
            return False, math.inf, f"unknown constraint sense for {name}"
        maximum_violation = max(maximum_violation, violation)
    if maximum_violation > tolerance:
        return False, maximum_violation, (
            f"maximum primal violation {maximum_violation:.6g} exceeds {tolerance:.6g}"
        )
    return True, maximum_violation, None


@dataclass(frozen=True)
class RivalSolveResult:
    rival: int
    status: str
    solution_status: str
    trusted: bool
    lower_bound: float | None
    clean_margin: float
    seconds: float
    solver_profile: str
    maximum_primal_violation: float | None
    error: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    schema_version: str
    target: int
    method: str
    depth: int
    hidden_dims: tuple[int, ...]
    bias: bool
    mode: str
    q: int
    global_budget: float
    prediction: int
    clean_margin: float
    exact_worst_margin: float | None
    lp_lower_bound: float | None
    approximation_gap: float | None
    certified: bool
    solver_status: str
    solver_trusted: bool
    rival_results: tuple[RivalSolveResult, ...]
    variable_count: int
    constraint_count: int
    input_abstraction: str
    potential_input_variable_count: int
    first_layer_bound_seconds: float
    bounds_cache_hit: bool
    solver_seconds: float
    total_seconds: float
    candidate_source_count: int
    candidate_sources: tuple[int, ...]
    candidate_mean_cost: float
    outer_bounds_summary: dict
    under_domain_summary: dict | None
    envelope_summary: dict
    envelope_details: list
    soundness_error: str | None

    def as_dict(self, *, include_envelope_details: bool = True) -> dict:
        payload = asdict(self)
        if not include_envelope_details:
            payload.pop("envelope_details", None)
        return payload


class DeepSigmoidVerifier:
    """Reusable verifier holding one model/graph feature matrix."""

    def __init__(
        self,
        model: DeepSigmoidGCN,
        adjacency: torch.Tensor,
        clean_x: torch.Tensor | np.ndarray,
        *,
        node_costs: Sequence[float] | None = None,
        method: str = "baseline",
        samples: int = 64,
        seed: int = 0,
        solver_timeout: float = 30.0,
        solver_tolerance: float = 1e-5,
        certification_tolerance: float = 1e-8,
        input_abstraction: str = "auto",
        explicit_input_max_variables: int = 50_000,
        shared_bounds_cache: dict | None = None,
    ) -> None:
        if method not in ("baseline", "dual"):
            raise ValueError("method must be baseline or dual")
        if input_abstraction not in ("auto", "explicit", "budget-box"):
            raise ValueError("input_abstraction must be auto, explicit, or budget-box")
        if (samples < 0 or solver_timeout < 0 or solver_tolerance <= 0
                or explicit_input_max_variables < 0):
            raise ValueError("invalid sample/solver/abstraction settings")
        self.model = model
        self.adjacency = adjacency
        self.clean_x = (clean_x.detach().cpu().double().numpy().copy()
                        if torch.is_tensor(clean_x)
                        else np.asarray(clean_x, dtype=np.float64).copy())
        if self.clean_x.ndim != 2 or self.clean_x.shape[1] != model.input_dim:
            raise ValueError("feature matrix shape does not match model")
        if not np.all((self.clean_x == 0.0) | (self.clean_x == 1.0)):
            raise ValueError("clean features must be binary")
        node_count = self.clean_x.shape[0]
        if adjacency.shape != (node_count, node_count):
            raise ValueError("adjacency and feature node counts differ")
        self.node_costs = (np.ones(node_count, dtype=np.float64)
                           if node_costs is None
                           else np.asarray(node_costs, dtype=np.float64).copy())
        if (self.node_costs.shape != (node_count,)
                or np.any(~np.isfinite(self.node_costs))
                or np.any(self.node_costs <= 0.0)):
            raise ValueError("node_costs must be one finite positive value per node")
        self.method = method
        self.samples = int(samples)
        self.seed = int(seed)
        self.solver_timeout = float(solver_timeout)
        self.solver_tolerance = float(solver_tolerance)
        self.certification_tolerance = float(certification_tolerance)
        self.input_abstraction = input_abstraction
        self.explicit_input_max_variables = int(explicit_input_max_variables)
        self.shared_bounds_cache = ({} if shared_bounds_cache is None
                                    else shared_bounds_cache)
        self.weights, self.biases = _validate_parameters(model)

    def verify(
        self,
        target: int,
        *,
        mode: str,
        q: int,
        global_budget: float,
        exact: ExactEnumerationResult | None = None,
    ) -> VerificationResult:
        started = time.perf_counter()
        mode = canonical_threat_mode(mode)
        if q < 0 or int(q) != q or not math.isfinite(float(global_budget)) \
                or global_budget < 0.0:
            raise ValueError("q must be an integer and global budget non-negative")
        q, global_budget = int(q), float(global_budget)
        weighted = mode in WEIGHTED_MODES
        base_key = ("base-v1", int(target))
        base = self.shared_bounds_cache.get(base_key)
        if base is None:
            graph = build_computation_graph(
                self.adjacency, target, self.model.depth
            )
            source_costs = self.node_costs[
                np.asarray(graph.candidate_sources, dtype=int)
            ]
            local_clean = self.clean_x[
                np.asarray(graph.candidate_sources, dtype=int)
            ]
            clean_logits, _ = evaluate_computation_graph(
                graph, local_clean, self.weights, self.biases
            )
            base = (graph, source_costs, local_clean, clean_logits)
            self.shared_bounds_cache[base_key] = base
        graph, source_costs, local_clean, clean_logits = base
        potential_input_variables = int(local_clean.size)
        input_abstraction = self.input_abstraction
        if input_abstraction == "auto":
            input_abstraction = (
                "explicit" if potential_input_variables
                <= self.explicit_input_max_variables else "budget-box"
            )
        threat = ThreatSpecification(
            mode=mode, q=q, global_budget=global_budget,
            source_costs=tuple(float(value) for value in source_costs),
        )
        prediction = int(np.argmax(clean_logits))
        rivals = [index for index in range(len(clean_logits)) if index != prediction]
        clean_margins = {
            rival: float(clean_logits[prediction] - clean_logits[rival])
            for rival in rivals
        }
        clean_margin = min(clean_margins.values())
        # Any positive LP budget permits a fractional z even when no full
        # weighted binary flip fits. Bounds therefore solve the complete
        # continuous grouped attack LP, not merely the integer under-domain.
        attack_possible = q > 0 and global_budget > 0.0
        bounds_key = (
            "bounds-v1", int(target), q, float(global_budget).hex(), weighted
        )
        cached_bounds = self.shared_bounds_cache.get(bounds_key)
        bounds_cache_hit = cached_bounds is not None
        if cached_bounds is None:
            first_lower, first_upper, first_summary = (
                budget_aware_first_layer_bounds(
                    graph, self.clean_x, self.weights[0], self.biases[0],
                    q=q, global_budget=global_budget,
                    source_costs=source_costs, weighted=weighted,
                )
            )
            bounds = interval_propagation(
                graph, self.clean_x, self.weights, self.biases,
                attack_possible=attack_possible,
                first_layer_override=(first_lower, first_upper),
                input_bound_method=first_summary["method"],
                first_layer_bound_seconds=first_summary["seconds"],
            )
            cached_bounds = (bounds, first_summary)
            self.shared_bounds_cache[bounds_key] = cached_bounds
        bounds, first_summary = cached_bounds
        under_domain = None
        if self.method == "dual":
            under_domain = monte_carlo_under_domain(
                graph, self.clean_x, self.weights, self.biases,
                q=q, global_budget=global_budget, source_costs=source_costs,
                weighted=weighted, samples=self.samples,
                seed=self.seed + 1_000_003 * int(target),
            )
        selections = select_network_envelopes(
            bounds, method=self.method, under_domain=under_domain
        )
        envelope_summary, envelope_details = envelope_diagnostics(
            selections, graph.layer_nodes[1:-1]
        )
        envelope_summary["lp_construction"] = (
            "all-outer-intersect-under-range-and-selected-lines-v2"
            if self.method == "dual" else "all-outer-intersection-v2"
        )
        artifacts = _build_lp(
            graph, self.clean_x, self.weights, self.biases,
            bounds, selections, threat,
            input_abstraction=input_abstraction,
        )

        try:
            import pulp
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("PuLP/CBC is required") from error
        rival_results = []
        solve_started = time.perf_counter()
        deadline = (solve_started + self.solver_timeout
                    if self.solver_timeout > 0.0 else None)
        for rival in rivals:
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                rival_results.append(RivalSolveResult(
                    rival=rival, status="TimedOut", solution_status="NoSolution",
                    trusted=False, lower_bound=None,
                    clean_margin=clean_margins[rival], seconds=0.0,
                    solver_profile="not-run-deadline",
                    maximum_primal_violation=None,
                    error="target solver deadline exceeded before rival solve",
                ))
                continue
            remaining = None if deadline is None else max(0.001, deadline - now)
            artifacts.model.setObjective(
                artifacts.logits[prediction] - artifacts.logits[rival]
            )
            one_started = time.perf_counter()
            error_text = None
            solver_profile = CBC_PRIMARY_PROFILE
            try:
                def solve_and_audit(profile: str, limit: float | None):
                    artifacts.model.solve(_cbc_solver(
                        pulp, time_limit=limit, profile=profile
                    ))
                    status = pulp.LpStatus.get(
                        artifacts.model.status, str(artifacts.model.status)
                    )
                    solution_status = pulp.LpSolution.get(
                        getattr(artifacts.model, "sol_status", 0),
                        str(getattr(artifacts.model, "sol_status", 0)),
                    )
                    timed_out = (
                        deadline is not None
                        and time.perf_counter() > deadline + 0.05
                    )
                    optimal = (
                        status == "Optimal"
                        and getattr(artifacts.model, "sol_status", None)
                        == pulp.LpSolutionOptimal
                        and not timed_out
                    )
                    objective = pulp.value(artifacts.model.objective)
                    if objective is None or not math.isfinite(float(objective)):
                        return (
                            status, solution_status, timed_out, False, None,
                            False, None, "objective is NaN/non-finite",
                        )
                    lower_bound = float(objective)
                    audited, maximum_violation, audit_error = (
                        _audit_solution(artifacts.model, self.solver_tolerance)
                        if optimal else (False, None, None)
                    )
                    error = audit_error if optimal and not audited else None
                    if optimal and audited and (
                        lower_bound
                        > clean_margins[rival] + self.solver_tolerance
                    ):
                        audited = False
                        error = (
                            "LP lower bound exceeds the feasible clean rival margin"
                        )
                    if not optimal and timed_out:
                        status = "TimedOut"
                        error = error or "target solver deadline exceeded"
                    return (
                        status, solution_status, timed_out, optimal,
                        lower_bound, audited, maximum_violation, error,
                    )

                (
                    status, solution_status, timed_out, optimal, lower_bound,
                    audited, maximum_violation, error_text,
                ) = solve_and_audit(solver_profile, remaining)

                # CBC may report Optimal while the mapped-back solution fails
                # the original-model residual audit after scaling/presolve.
                # Discard that solution and deterministically resolve the same
                # continuous LP without either transformation.  The audit
                # tolerance and certification decision remain unchanged.
                retry_now = time.perf_counter()
                retry_allowed = (
                    optimal and not audited and not timed_out
                    and (deadline is None or retry_now < deadline)
                )
                if retry_allowed:
                    solver_profile = CBC_AUDIT_RETRY_PROFILE
                    retry_remaining = (
                        None if deadline is None
                        else max(0.001, deadline - retry_now)
                    )
                    (
                        status, solution_status, timed_out, optimal, lower_bound,
                        audited, maximum_violation, error_text,
                    ) = solve_and_audit(solver_profile, retry_remaining)

                elapsed = time.perf_counter() - one_started
                trusted = bool(optimal and audited and lower_bound is not None)
                if not trusted:
                    lower_bound = None
                rival_results.append(RivalSolveResult(
                    rival=rival, status=status,
                    solution_status=solution_status, trusted=trusted,
                    lower_bound=lower_bound,
                    clean_margin=clean_margins[rival], seconds=elapsed,
                    solver_profile=solver_profile,
                    maximum_primal_violation=maximum_violation,
                    error=error_text,
                ))
            except Exception as error:
                elapsed = time.perf_counter() - one_started
                rival_results.append(RivalSolveResult(
                    rival=rival, status="SolverError", solution_status="NoSolution",
                    trusted=False, lower_bound=None,
                    clean_margin=clean_margins[rival], seconds=elapsed,
                    solver_profile=solver_profile,
                    maximum_primal_violation=None,
                    error=f"{type(error).__name__}: {error}",
                ))
        solver_seconds = time.perf_counter() - solve_started
        all_trusted = bool(rival_results) and all(result.trusted
                                                  for result in rival_results)
        lp_lower = (min(float(result.lower_bound) for result in rival_results
                        if result.lower_bound is not None)
                    if all_trusted else None)
        exact_margin = None if exact is None else float(exact.worst_margin)
        soundness_error = None
        if exact is not None:
            if exact.target != int(target) or exact.prediction != prediction:
                soundness_error = "exact enumeration target/prediction mismatch"
            elif lp_lower is not None and lp_lower > exact_margin + self.solver_tolerance:
                soundness_error = (
                    f"LP lower {lp_lower:.12g} exceeds exact worst "
                    f"{exact_margin:.12g}"
                )
        if soundness_error is not None:
            all_trusted = False
        benchmark = exact_margin if exact_margin is not None else clean_margin
        approximation_gap = (None if lp_lower is None
                             else float(benchmark - lp_lower))
        certified = bool(
            all_trusted and lp_lower is not None
            and lp_lower > max(self.certification_tolerance,
                               self.solver_tolerance)
            and soundness_error is None
        )
        solver_status = ("Optimal" if all_trusted else
                         ("SoundnessError" if soundness_error else
                          ";".join(sorted({result.status
                                          for result in rival_results}))))
        total_seconds = time.perf_counter() - started
        return VerificationResult(
            schema_version=VERIFIER_SCHEMA_VERSION,
            target=int(target), method=self.method, depth=self.model.depth,
            hidden_dims=self.model.hidden_dims, bias=self.model.bias_enabled,
            mode=mode, q=q, global_budget=global_budget,
            prediction=prediction, clean_margin=clean_margin,
            exact_worst_margin=exact_margin, lp_lower_bound=lp_lower,
            approximation_gap=approximation_gap, certified=certified,
            solver_status=solver_status, solver_trusted=all_trusted,
            rival_results=tuple(rival_results),
            variable_count=artifacts.variable_count,
            constraint_count=artifacts.constraint_count,
            input_abstraction=artifacts.input_abstraction,
            potential_input_variable_count=artifacts.potential_input_variable_count,
            first_layer_bound_seconds=float(first_summary["seconds"]),
            bounds_cache_hit=bounds_cache_hit,
            solver_seconds=solver_seconds, total_seconds=total_seconds,
            candidate_source_count=len(graph.candidate_sources),
            candidate_sources=graph.candidate_sources,
            candidate_mean_cost=float(math.fsum(source_costs)
                                      / len(source_costs)),
            outer_bounds_summary={
                **bounds.summary(),
                "first_layer_budget": {
                    key: value for key, value in first_summary.items()
                    if key != "seconds"
                },
            },
            under_domain_summary=(None if under_domain is None
                                  else under_domain.summary()),
            envelope_summary=envelope_summary,
            envelope_details=envelope_details,
            soundness_error=soundness_error,
        )
