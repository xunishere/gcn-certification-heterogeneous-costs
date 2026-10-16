"""Exact, sound feature-flip bounds under count-local / cost-global budgets.

A candidate item is one binary-feature flip ``z[v, j]`` at source node ``v``.
Every mode uses the *count* constraint ``sum_j z[v, j] <= q``.  Weighted modes
add only the global monetary constraint ``sum_{v,j} c[v] z[v,j] <= Q_cost``;
count modes use ``sum_{v,j} z[v,j] <= Q`` instead.

For a fixed source and flip count, selecting its largest non-negative gains is
optimal.  The weighted solver therefore forms exact source options and runs a
Pareto multiple-choice-knapsack DP.  It never rounds costs.  ``EPS`` is used
only to include numerically boundary-feasible attacks, which is conservative
for certification.
"""
from bisect import bisect_right
import math
import time

EPS = 1e-12


def _check_local_count(local_count):
    if math.isinf(local_count):
        return
    if local_count < 0 or int(local_count) != local_count:
        raise ValueError("q/local_count must be a non-negative integer")


def _pareto(states):
    """Remove only exactly dominated (cost, gain) states."""
    states = sorted(states, key=lambda item: (item[0], -item[1]))
    out, best_gain = [], float("-inf")
    for cost, gain in states:
        if gain > best_gain:
            out.append((cost, gain))
            best_gain = gain
    return out


def source_flip_options(gains, source_cost, local_count):
    """Return exact ``(spent_cost, gain, flip_count)`` options for one source.

    Local feasibility depends only on ``flip_count <= q``.  Consequently a
    source with cost greater than q can still flip a feature when q >= 1.
    """
    _check_local_count(local_count)
    source_cost = float(source_cost)
    if not math.isfinite(source_cost) or source_cost <= 0:
        raise ValueError("source costs must be finite and positive")
    values = sorted((max(0.0, float(g)) for g in gains), reverse=True)
    # Zero-gain flips can never improve a maximization under <= constraints.
    positive = [value for value in values if value > 0.0]
    limit = len(positive) if math.isinf(local_count) else min(len(positive), int(local_count))
    options = [(0.0, 0.0, 0)]
    cumulative = 0.0
    for count, value in enumerate(positive[:limit], 1):
        cumulative += value
        options.append((count * source_cost, cumulative, count))
    return options


def count_gain_frontier(gains_by_source, global_count, local_count):
    """Exact frontier for local-count q and global-count Q constraints."""
    _check_local_count(local_count)
    if global_count < 0:
        raise ValueError("global count budget must be non-negative")
    limit = int(math.floor(float(global_count) + EPS))
    candidates = []
    for gains in gains_by_source:
        values = sorted((max(0.0, float(g)) for g in gains), reverse=True)
        local_limit = len(values) if math.isinf(local_count) else min(len(values), int(local_count))
        candidates.extend(value for value in values[:local_limit] if value > 0.0)
    candidates.sort(reverse=True)
    frontier, cumulative = [(0.0, 0.0)], 0.0
    for count, value in enumerate(candidates[:limit], 1):
        cumulative += value
        frontier.append((float(count), cumulative))
    return frontier


def weighted_gain_frontier(gains_by_source, source_costs, global_budget,
                           local_count, deadline=None):
    """Return the exact Pareto frontier up to ``global_budget``.

    ``deadline`` is an optional absolute ``time.monotonic()`` value.  A
    ``TimeoutError`` aborts without returning an approximate (potentially
    unsound) answer.
    """
    _check_local_count(local_count)
    if not math.isfinite(float(global_budget)) or global_budget < 0:
        raise ValueError("global monetary budget must be finite and non-negative")
    if len(gains_by_source) != len(source_costs):
        raise ValueError("one cost is required per source node")
    costs = [float(cost) for cost in source_costs]
    if any(not math.isfinite(cost) or cost <= 0 for cost in costs):
        raise ValueError("source costs must be finite and positive")

    # This exact fast path also guarantees that all-unit costs (for example,
    # alpha=0) recover the legacy count threat set and identical gain sums.
    if all(cost == 1.0 for cost in costs):
        return count_gain_frontier(gains_by_source, global_budget, local_count)

    frontier = [(0.0, 0.0)]
    for source_index, (gains, cost) in enumerate(zip(gains_by_source, costs)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("weighted Pareto DP deadline exceeded")
        options = source_flip_options(gains, cost, local_count)
        merged = []
        for state_index, (spent, value) in enumerate(frontier):
            if (deadline is not None and state_index % 2048 == 0
                    and time.monotonic() >= deadline):
                raise TimeoutError("weighted Pareto DP deadline exceeded")
            remaining = float(global_budget) + EPS - spent
            for option_cost, option_gain, _ in options:
                if option_cost <= remaining:
                    merged.append((spent + option_cost, value + option_gain))
                else:
                    # Source options are ordered by increasing flip count/cost.
                    break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("weighted Pareto DP deadline exceeded")
        frontier = _pareto(merged)
        if not frontier:
            # The no-flip option always makes this unreachable.
            raise RuntimeError("weighted frontier unexpectedly became empty")
        if deadline is not None and source_index % 8 == 7 and time.monotonic() >= deadline:
            raise TimeoutError("weighted Pareto DP deadline exceeded")
    return frontier


def frontier_max_gain(frontier, budget):
    """Query an increasing-cost Pareto frontier conservatively at ``budget``."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    costs = [state[0] for state in frontier]
    index = bisect_right(costs, float(budget) + EPS) - 1
    return 0.0 if index < 0 else frontier[index][1]


def count_max_gain(gains_by_source, global_count, local_count):
    return frontier_max_gain(
        count_gain_frontier(gains_by_source, global_count, local_count),
        global_count,
    )


def weighted_max_gain(gains_by_source, source_costs, global_budget,
                      local_count=float("inf"), deadline=None):
    """Exact adverse gain with local flip-count q and global monetary cost."""
    frontier = weighted_gain_frontier(
        gains_by_source, source_costs, global_budget, local_count, deadline
    )
    return frontier_max_gain(frontier, global_budget)


def weighted_interval(nominal, upper_gains_by_source, lower_gains_by_source,
                      source_costs, global_budget, local_count=float("inf"),
                      deadline=None):
    upper = float(nominal) + weighted_max_gain(
        upper_gains_by_source, source_costs, global_budget, local_count, deadline
    )
    lower = float(nominal) - weighted_max_gain(
        lower_gains_by_source, source_costs, global_budget, local_count, deadline
    )
    return lower, upper
