import itertools
import math
import unittest

from VRP.experiment_config import ExperimentConfig
from VRP.graph_semantics import (
    GRAPH_SEMANTICS_VERSION,
    canonical_graph_sha256,
    canonical_undirected_edges,
)
from VRP.node_importance import attack_costs, compute_node_importance
from VRP.analyze_experiments import aggregate_metrics, per_seed_metrics
from VRP.dataset_registry import (
    canonical_dataset_name,
    dataset_spec,
    q0_values_for_dataset,
    q_values_for_dataset,
)
from VRP.risk_budget import (
    COST_BUDGET_POLICY,
    RISK_BUDGET_POLICY,
    allocate_fixed_mean_budgets,
    certification_budget,
    fixed_resource_budget,
    target_risk,
)
from VRP.run_heterogeneous import prepare
from VRP.weighted_bounds import (
    count_max_gain,
    source_flip_options,
    weighted_max_gain,
)


def brute(groups, costs, total_cost, local_count):
    """Exhaust all item subsets under corrected count-local/cost-global rules."""
    items = [(source, value)
             for source, group in enumerate(groups) for value in group]
    best = 0.0
    for mask in range(1 << len(items)):
        counts = [0] * len(groups)
        spent = value = 0.0
        for index, (source, gain) in enumerate(items):
            if mask >> index & 1:
                counts[source] += 1
                spent += costs[source]
                value += max(0.0, gain)
        if (spent <= total_cost + 1e-12
                and max(counts, default=0) <= local_count):
            best = max(best, value)
    return best


class HeterogeneousTests(unittest.TestCase):
    def test_canonical_graph_is_shared_by_cost_and_gcn(self):
        import torch

        from VRP.deep_gcn import normalized_adjacency
        from VRP.real_experiments import FIELDS, normalized_adj

        raw = [(0, 1), (0, 1), (1, 1), (2, 1)]
        canonical = ((0, 1), (1, 0), (1, 2), (2, 1))
        self.assertEqual(canonical_undirected_edges(3, raw), canonical)
        self.assertEqual(
            compute_node_importance(3, raw),
            compute_node_importance(3, canonical),
        )
        edge_index = torch.tensor(raw, dtype=torch.long).t().contiguous()
        one_layer = normalized_adj(edge_index, 3, torch.device("cpu")).to_dense()
        deep = normalized_adjacency(edge_index, 3).to_dense()
        torch.testing.assert_close(one_layer, deep, rtol=0, atol=0)
        torch.testing.assert_close(one_layer, one_layer.T, rtol=0, atol=0)
        self.assertTrue(bool(torch.all(torch.diag(one_layer) > 0)))
        self.assertEqual(
            canonical_graph_sha256(3, raw),
            canonical_graph_sha256(3, canonical),
        )
        self.assertNotIn("source_cost", FIELDS)
        for field in ("target_cost", "candidate_sources",
                      "candidate_source_costs", "uses_heterogeneous_cost"):
            self.assertIn(field, FIELDS)

    def test_importance_normalized_and_symmetric(self):
        importance = compute_node_importance(3, [(0, 1), (1, 2)])
        self.assertTrue(all(0 <= value <= 1 for value in importance))
        self.assertAlmostEqual(importance[0], importance[2])
        self.assertGreater(importance[1], importance[0])
        self.assertEqual(attack_costs([0, 1], 2), [1, 3])

    def test_risk_then_fixed_resource_cost_budget(self):
        self.assertAlmostEqual(target_risk(.8, .6, .25), .5)
        # Retain regression coverage for the pre-v3 independent helper, while
        # proving that the actual risk mapping requires a joint allocation.
        self.assertEqual(certification_budget(3, .8, .6, .25, 5), 5)
        allocation = allocate_fixed_mean_budgets({0: .5, 1: .3}, 3, 5)
        self.assertEqual(allocation.budgets, {0: 4, 1: 2})
        budget = fixed_resource_budget(
            3, use_risk=True, use_cost=True,
            allocated_q_t=allocation.budgets[0],
        )
        self.assertEqual(budget["Q_t"], 4)
        self.assertEqual(budget["Q_equiv"], 4)
        self.assertEqual(budget["Q_cost"], 4.0)
        self.assertEqual(budget["global_budget"], 4.0)
        self.assertEqual(budget["cost_budget_policy"], COST_BUDGET_POLICY)
        self.assertEqual(budget["global_budget_unit"], "cost")
        fixed = fixed_resource_budget(3, use_risk=False, use_cost=True)
        self.assertEqual(fixed["Q_cost"], 3.0)
        self.assertEqual(fixed["global_budget"], 3.0)
        with self.assertRaises(ValueError):
            fixed_resource_budget(3, use_risk=True, use_cost=False)

    def test_fixed_mean_budget_conservation_integer_and_monotone(self):
        risks = {9: .01, 2: .8, 7: .2, 4: .5, 1: .5}
        for q0 in (0, 1, 3, 20):
            for beta in (0, .5, 10, 1000):
                allocation = allocate_fixed_mean_budgets(risks, q0, beta)
                values = allocation.budgets
                self.assertEqual(sum(values.values()), len(risks) * q0)
                self.assertTrue(all(isinstance(value, int) and value >= 0
                                    for value in values.values()))
                ordered = sorted(risks, key=lambda target: risks[target])
                for lower, higher in itertools.combinations(ordered, 2):
                    if risks[lower] < risks[higher]:
                        self.assertLessEqual(values[lower], values[higher])
                self.assertEqual(allocation.policy, RISK_BUDGET_POLICY)
                self.assertEqual(len(allocation.checksum), 64)

    def test_fixed_mean_beta_zero_and_constant_risk(self):
        beta_zero = allocate_fixed_mean_budgets({8: .1, 2: .9, 5: .4}, 3, 0)
        constant = allocate_fixed_mean_budgets({8: .4, 2: .4, 5: .4}, 3, 1000)
        self.assertEqual(beta_zero.budgets, {2: 3, 5: 3, 8: 3})
        self.assertEqual(constant.budgets, {2: 3, 5: 3, 8: 3})

    def test_fixed_mean_low_q0_projection_and_order_stability(self):
        forward = [(11, .9), (3, .1), (7, .5), (2, .5)]
        reverse = list(reversed(forward))
        first = allocate_fixed_mean_budgets(forward, 1, 100)
        second = allocate_fixed_mean_budgets(reverse, 1, 100)
        self.assertEqual(first.budgets, second.budgets)
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(first.total, 4)
        self.assertEqual(first.minimum, 0)
        zero = allocate_fixed_mean_budgets(forward, 0, 100)
        self.assertEqual(zero.budgets, {2: 0, 3: 0, 7: 0, 11: 0})
        tied = allocate_fixed_mean_budgets({5: .6, 2: .6, 9: .2}, 1, 4)
        self.assertEqual(tied.budgets, {2: 2, 5: 1, 9: 0})

    def test_risk_and_combined_share_exact_mapping(self):
        graph = (3, [(0, 1), (1, 2)], [.9, .6, .8])
        risk_rows = prepare(
            ExperimentConfig(mode="risk-budget", beta=10, q0=1, q=1), *graph
        )
        combined_rows = prepare(
            ExperimentConfig(mode="combined", beta=10, q0=1, q=1), *graph
        )
        self.assertEqual(
            [(row["target"], row["Q_t"]) for row in risk_rows],
            [(row["target"], row["Q_t"]) for row in combined_rows],
        )
        self.assertEqual(
            {row["risk_budget_mapping_checksum"] for row in risk_rows},
            {row["risk_budget_mapping_checksum"] for row in combined_rows},
        )
        self.assertEqual(sum(row["Q_t"] for row in risk_rows), 3)

    def test_dataset_registry_aliases_and_q_grids(self):
        self.assertEqual(canonical_dataset_name("Citeseer"), "CiteSeer")
        self.assertEqual(canonical_dataset_name("CiteSeer"), "CiteSeer")
        self.assertEqual(dataset_spec("Cora").main_q, 14)
        self.assertEqual(dataset_spec("CiteSeer").main_q, 37)
        self.assertEqual(q_values_for_dataset("Cora", sensitivity=True),
                         (14,))
        self.assertEqual(q_values_for_dataset("Citeseer", sensitivity=True),
                         (37,))
        self.assertEqual(q0_values_for_dataset("Cora"), (14,))
        self.assertEqual(q0_values_for_dataset("CiteSeer"), (37,))
        self.assertEqual(q0_values_for_dataset("Cora", sensitivity=True),
                         (3, 7, 14, 28))
        self.assertEqual(q0_values_for_dataset("Citeseer", sensitivity=True),
                         (9, 18, 37, 74))

    def test_experiment_clis_resolve_dataset_specific_local_and_global_budgets(self):
        import contextlib
        import io

        from VRP.deep_experiments import (
            _q0_values_for_run as deep_q0_values,
            _q_values_for_run as deep_q_values,
            parse as parse_deep,
        )
        from VRP.real_experiments import _run_configuration, parse as parse_one

        defaults = parse_one([
            "--datasets", "Cora", "CiteSeer", "--output-dir", "unused",
        ])
        configuration = _run_configuration(defaults, "cpu")
        self.assertEqual(defaults.modes, ["legacy", "heterogeneous-cost"])
        self.assertEqual(configuration["q_values_by_dataset"],
                         {"Cora": [14], "CiteSeer": [37]})
        self.assertEqual(configuration["Q0_values_by_dataset"],
                         {"Cora": [14], "CiteSeer": [37]})

        sensitivity = parse_deep([
            "--datasets", "Cora", "CiteSeer", "--q-sensitivity",
            "--q0-sensitivity", "--output-dir", "unused",
        ])
        self.assertEqual(sensitivity.modes, ["legacy", "heterogeneous-cost"])
        self.assertEqual(
            {name: deep_q_values(sensitivity, name) for name in sensitivity.datasets},
            {"Cora": (14,), "CiteSeer": (37,)},
        )
        self.assertEqual(
            {name: deep_q0_values(sensitivity, name) for name in sensitivity.datasets},
            {"Cora": (3, 7, 14, 28), "CiteSeer": (9, 18, 37, 74)},
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_one([
                    "--datasets", "Cora", "--modes", "risk-budget",
                    "--output-dir", "unused",
                ])

    def test_analysis_groups_dataset_and_scope_without_mixing(self):
        import pandas as pd

        rows = []
        for dataset in ("Cora", "CiteSeer"):
            rows.append({
                "scope": "formal", "dataset": dataset, "seed": 0,
                "mode": "legacy", "q": 1, "Q0": 3, "target": 0,
                "certified": 1, "correct": 1, "certified_and_correct": 1,
                "key_node": 1, "risk_weight": .5, "seconds": .01,
                "clean_test_accuracy": .7, "Q_t": 3,
            })
        per_seed = per_seed_metrics(pd.DataFrame(rows))
        aggregate = aggregate_metrics(per_seed)
        self.assertEqual(len(per_seed), 2)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(set(aggregate["dataset"]), {"Cora", "CiteSeer"})

    def test_weighted_exact_against_exhaustive(self):
        groups = [[7, 4], [6, 5], [3]]
        costs = [1.5, 2.0, .75]
        for q, total in itertools.product((0, 1, 2), (0, 1.5, 2.75, 4.0, 6.0)):
            got = weighted_max_gain(groups, costs, total, q)
            self.assertAlmostEqual(got, brute(groups, costs, total, q))

    def test_all_sources_can_flip_once_when_q_and_global_allow(self):
        # Costs larger than q do not block a local flip.  With Q_cost=sum(c),
        # one feature from every source can be changed simultaneously.
        groups = [[1.0], [2.0], [4.0]]
        costs = [1.2, 1.5, 3.0]
        self.assertEqual(weighted_max_gain(groups, costs, sum(costs), 1), 7.0)
        for gains, cost in zip(groups, costs):
            self.assertIn((cost, gains[0], 1), source_flip_options(gains, cost, 1))

    def test_heterogeneous_cost_changes_global_not_local_capacity(self):
        low = source_flip_options([9, 8], 1.2, 2)
        high = source_flip_options([9, 8], 1.5, 2)
        self.assertEqual([option[2] for option in low], [0, 1, 2])
        self.assertEqual([option[2] for option in high], [0, 1, 2])
        self.assertEqual(low[1][0], 1.2)
        self.assertEqual(high[1][0], 1.5)
        self.assertEqual(weighted_max_gain([[9], [8]], [1.2, 1.5], 1.3, 1), 9)

    def test_alpha_zero_mean_one_exact_legacy_degeneration(self):
        groups = [[8, 5, 2], [7, 1], [6]]
        for q, budget in itertools.product((1, 2, 3), (0, 1, 2, 4, 7)):
            legacy = count_max_gain(groups, budget, q)
            heterogeneous = weighted_max_gain(groups, [1.0] * 3, budget, q)
            self.assertEqual(legacy, heterogeneous)
        edges, confidence = [(0, 1)], [.9, .8]
        legacy_rows = prepare(
            ExperimentConfig(mode="legacy", alpha=0, q0=4, q=1),
            2, edges, confidence,
        )
        cost_rows = prepare(
            ExperimentConfig(mode="heterogeneous-cost", alpha=0, q0=4, q=1),
            2, edges, confidence,
        )
        for legacy_row, cost_row in zip(legacy_rows, cost_rows):
            for field in ("target_cost", "candidate_sources",
                          "candidate_source_costs", "Q0", "Q_t", "Q_equiv",
                          "Q_cost", "q", "mean_cost", "global_budget"):
                self.assertEqual(legacy_row[field], cost_row[field])
            self.assertEqual(legacy_row["uses_heterogeneous_cost"], 0)
            self.assertEqual(cost_row["uses_heterogeneous_cost"], 1)
            self.assertEqual(
                len(cost_row["candidate_sources"]),
                len(cost_row["candidate_source_costs"]),
            )
        self.assertTrue(all(row["mean_cost"] == 1 for row in cost_rows))
        self.assertTrue(all(row["Q_cost"] == 4 for row in cost_rows))

    def test_combined_records_mean_without_scaling_fixed_budget(self):
        rows = prepare(
            ExperimentConfig(mode="combined", alpha=1, eta=.5, beta=10, q0=14, q=3),
            3, [(0, 1), (1, 2)], [.9, .6, .8],
        )
        self.assertEqual(rows[0]["candidate_source_count"], 2)  # self + node 1
        self.assertEqual(rows[1]["candidate_source_count"], 3)
        for row in rows:
            self.assertEqual(row["graph_semantics_version"],
                             GRAPH_SEMANTICS_VERSION)
            self.assertNotIn("source_attack_cost", row)
            self.assertEqual(row["uses_heterogeneous_cost"], 1)
            self.assertEqual(row["target_cost"],
                             row["candidate_source_costs"][
                                 row["candidate_sources"].index(row["target"])
                             ])
            self.assertEqual(len(row["candidate_sources"]),
                             row["candidate_source_count"])
            self.assertEqual(len(row["candidate_source_costs"]),
                             row["candidate_source_count"])
            self.assertEqual(row["Q_equiv"], row["Q_t"])
            self.assertEqual(row["Q_cost"], float(row["Q_t"]))
            self.assertEqual(row["global_budget"], float(row["Q_t"]))
            self.assertEqual(row["cost_budget_policy"], COST_BUDGET_POLICY)
            self.assertEqual(row["q"], 3)

        count_only = prepare(
            ExperimentConfig(mode="legacy", alpha=1, q0=14, q=3),
            3, [(0, 1), (1, 2)], [.9, .6, .8],
        )
        self.assertEqual(count_only[1]["uses_heterogeneous_cost"], 0)
        self.assertGreater(count_only[1]["target_cost"], 1.0)
        self.assertTrue(any(
            cost > 1.0 for cost in count_only[1]["candidate_source_costs"]
        ))

    def test_lp_budget_constraints_if_pulp_available(self):
        try:
            import pulp
            from VRP.LP import add_attack_budget_constraints
        except ImportError:
            self.skipTest("PuLP/scientific LP dependencies not installed")
        model = pulp.LpProblem("budget-semantics", pulp.LpMaximize)
        attack = [[pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
                   for j in range(2)] for i in range(2)]
        model += pulp.lpSum(variable for row in attack for variable in row)
        add_attack_budget_constraints(model, attack, q=1, global_budget=2.7,
                                      source_costs=[1.2, 1.5])
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pulp.LpStatus[model.status], "Optimal")
        # One flip at each node is feasible even though both c_v > q.
        self.assertEqual(model.objective.value(), 2.0)


if __name__ == "__main__":
    unittest.main()
