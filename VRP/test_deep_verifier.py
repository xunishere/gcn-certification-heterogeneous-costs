"""Correctness gate for the depth-2/depth-3 sigmoid GCN verifier.

Every certification test compares the LP relaxation with exhaustive binary
attacks on tiny graphs.  These tests are intentionally mandatory in the deep
real-data CLI.
"""
from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
import torch

import VRP.deep_verifier as deep_verifier_module

from VRP.deep_gcn import (
    NORMALIZATION_VERSION,
    DeepSigmoidGCN,
    TrainingHyperparameters,
    model_cache_description,
    normalized_adjacency,
    seed_all,
)
from VRP.deep_verifier import (
    CBC_AUDIT_RETRY_PROFILE,
    CBC_RANDOM_SEED,
    DeepSigmoidVerifier,
    WEIGHTED_MODES,
    budget_aware_first_layer_bounds,
    build_computation_graph,
    evaluate_computation_graph,
    exhaustive_worst_margin,
    grouped_fractional_knapsack_max,
    interval_propagation,
)
from VRP.graph_semantics import GRAPH_SEMANTICS_VERSION
from VRP.sigmoid_envelope import (
    calibrate_intercepts,
    select_envelope,
    sigmoid,
)


def tiny_graph(dtype=torch.float64):
    # Undirected path 0--1--2, represented in both directions.
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )
    return normalized_adjacency(edge_index, 3, dtype=dtype)


def tiny_instance(seed: int, depth: int, bias: bool):
    seed_all(seed)
    x_options = [
        [[1, 0], [0, 1], [1, 1]],
        [[0, 1], [1, 0], [0, 0]],
        [[1, 1], [0, 0], [1, 0]],
    ]
    x = torch.tensor(x_options[seed % len(x_options)], dtype=torch.float64)
    model = DeepSigmoidGCN(
        2, [2] * (depth - 1), 2, depth=depth, bias=bias
    ).double()
    # Avoid accidental nearly tied clean logits, which are uninteresting for
    # a numerical soundness gate but can make strict robustness flaky.
    with torch.no_grad():
        for layer, weight in enumerate(model.weights):
            weight.mul_(0.7 + 0.1 * layer)
        if model.bias_enabled:
            model.biases[-1].add_(torch.tensor([0.17, -0.11]))
    return model, tiny_graph(), x


def all_feasible_local_masks(source_count, feature_count, q, budget,
                             source_costs, weighted):
    for bits in itertools.product((0, 1), repeat=source_count * feature_count):
        mask = np.asarray(bits, dtype=np.uint8).reshape(source_count, feature_count)
        if np.any(mask.sum(axis=1) > q):
            continue
        spent = math.fsum(
            int(mask[source].sum())
            * (source_costs[source] if weighted else 1.0)
            for source in range(source_count)
        )
        if spent <= budget + 1e-12:
            yield mask


class SigmoidEnvelopeHighPrecisionTests(unittest.TestCase):
    def test_analytic_intercepts_and_full_interval_validity(self):
        mp.mp.dps = 90
        intervals = [
            (-12.0, -0.2), (0.1, 11.0), (-7.0, 9.0), (-1.0, 1.0),
            (-1e-8, 2e-8), (0.0, 0.0), (-3.25, -3.25),
            (-750.0, 750.0),
        ]
        generator = np.random.default_rng(20260805)
        for _ in range(20):
            first, second = sorted(generator.uniform(-15, 15, size=2))
            intervals.append((float(first), float(second)))
        slopes = [-.7, -.01, 0.0, math.nextafter(0.0, 1.0),
                  1e-9, .01, .08, .2, .25, .31, 1.2]
        slopes.extend(float(value) for value in generator.uniform(-.4, .7, size=12))

        for lower, upper in intervals:
            for slope in slopes:
                calibrated = calibrate_intercepts(lower, upper, slope)
                points = [mp.mpf(lower), mp.mpf(upper)]
                a = mp.mpf(slope)
                if 0 < a <= mp.mpf("0.25"):
                    if a == mp.mpf("0.25"):
                        roots = [mp.mpf(0)]
                    else:
                        root_probability = (1 - mp.sqrt(1 - 4 * a)) / 2
                        low = mp.log(root_probability / (1 - root_probability))
                        roots = [low, -low]
                    points.extend(root for root in roots
                                  if mp.mpf(lower) <= root <= mp.mpf(upper))
                values = [1 / (1 + mp.e ** (-point)) - a * point
                          for point in points]
                exact_lower, exact_upper = min(values), max(values)
                raw_tolerance = mp.mpf("3e-15") * max(
                    1, abs(exact_lower), abs(exact_upper)
                )
                self.assertLessEqual(
                    abs(mp.mpf(calibrated.raw_lower_intercept) - exact_lower),
                    raw_tolerance,
                )
                self.assertLessEqual(
                    abs(mp.mpf(calibrated.raw_upper_intercept) - exact_upper),
                    raw_tolerance,
                )
                # Dense high-precision checks plus every analytic extremum.
                grid = [mp.mpf(lower) + (mp.mpf(upper) - mp.mpf(lower)) * index / 128
                        for index in range(129)] if lower != upper else [mp.mpf(lower)]
                for point in grid + points:
                    exact = 1 / (1 + mp.e ** (-point))
                    affine_lower = (mp.mpf(slope) * point
                                    + mp.mpf(calibrated.lower_intercept))
                    affine_upper = (mp.mpf(slope) * point
                                    + mp.mpf(calibrated.upper_intercept))
                    self.assertLessEqual(affine_lower, exact + mp.mpf("3e-15"))
                    self.assertLessEqual(exact, affine_upper + mp.mpf("3e-15"))

        # Cancellation stress: the unpadded intercept is near zero although
        # both terms are O(1); safety padding must still enclose exact sigmoid.
        cancellation_slope = sigmoid(1.0)
        cancellation = calibrate_intercepts(1.0, 1.0, cancellation_slope)
        exact = 1 / (1 + mp.e ** -1)
        self.assertLessEqual(
            mp.mpf(cancellation_slope) + mp.mpf(cancellation.lower_intercept),
            exact,
        )
        self.assertLessEqual(
            exact,
            mp.mpf(cancellation_slope) + mp.mpf(cancellation.upper_intercept),
        )

    def test_dual_recalibrates_outer_and_degenerate_fallback(self):
        dual = select_envelope(
            -5.0, 4.0, method="dual",
            under_values=[-.7, -.2, .1, .6, 1.2],
        )
        self.assertTrue(dual.used_under_domain)
        for point in np.linspace(-5, 4, 2001):
            exact = sigmoid(float(point))
            self.assertLessEqual(dual.lower.value(point), exact + 2e-15)
            self.assertLessEqual(exact, dual.upper.value(point) + 2e-15)
        fallback = select_envelope(
            -2.0, 3.0, method="dual", under_values=[.4] * 10
        )
        baseline = select_envelope(-2.0, 3.0, method="baseline")
        self.assertFalse(fallback.used_under_domain)
        self.assertEqual(fallback.fallback_reason, "under_domain_no_variation")
        self.assertEqual(fallback.lower, baseline.lower)
        self.assertEqual(fallback.upper, baseline.upper)


class DeepModelAndBoundsTests(unittest.TestCase):
    def test_cbc_seed_is_fixed_and_nonzero(self):
        # CBC interprets randomSeed=0 as a time-of-day seed.
        self.assertEqual(CBC_RANDOM_SEED, 1)

    def test_forward_matches_exact_depth_hop_graph_with_bias_and_without(self):
        for depth, bias in itertools.product((2, 3), (False, True)):
            model, adjacency, x = tiny_instance(4 + depth, depth, bias)
            with torch.no_grad():
                full_logits = model(x, adjacency).detach().cpu().numpy()
            for target in range(3):
                graph = build_computation_graph(adjacency, target, depth)
                weights, biases = model.numpy_parameters()
                local = x.numpy()[np.asarray(graph.candidate_sources)]
                logits, _ = evaluate_computation_graph(
                    graph, local, weights, biases
                )
                np.testing.assert_allclose(logits, full_logits[target],
                                           rtol=1e-12, atol=1e-12)
                self.assertIn(target, graph.candidate_sources)
            self.assertEqual(len(build_computation_graph(
                adjacency, 0, depth
            ).layer_nodes), depth + 1)

    def test_outer_intervals_cover_every_count_and_weighted_attack(self):
        costs_all = np.asarray([1.0, 1.35, 1.8])
        cases = [
            (2, True, "legacy", 1, 2.0),
            (3, False, "heterogeneous-cost", 1, 2.2),
            (3, True, "legacy", 2, 2.0),
            (2, False, "heterogeneous-cost", 0, 4.0),
            (3, True, "legacy", 1, 0.0),
        ]
        for depth, bias, mode, q, budget in cases:
            model, adjacency, x = tiny_instance(10 + depth, depth, bias)
            graph = build_computation_graph(adjacency, 1, depth)
            weights, biases = model.numpy_parameters()
            source_costs = costs_all[np.asarray(graph.candidate_sources)]
            attack_possible = q > 0 and budget > 0
            bounds = interval_propagation(
                graph, x.numpy(), weights, biases,
                attack_possible=attack_possible,
            )
            local_clean = x.numpy()[np.asarray(graph.candidate_sources)]
            weighted = mode in WEIGHTED_MODES
            for mask in all_feasible_local_masks(
                len(graph.candidate_sources), x.shape[1], q, budget,
                source_costs, weighted,
            ):
                attacked = local_clean + (1 - 2 * local_clean) * mask
                logits, hidden_pre = evaluate_computation_graph(
                    graph, attacked, weights, biases
                )
                for layer, values in enumerate(hidden_pre):
                    self.assertTrue(np.all(
                        values >= bounds.pre_lower[layer] - 2e-12
                    ))
                    self.assertTrue(np.all(
                        values <= bounds.pre_upper[layer] + 2e-12
                    ))
                self.assertTrue(np.all(logits >= bounds.output_lower - 2e-12))
                self.assertTrue(np.all(logits <= bounds.output_upper + 2e-12))

    def test_budget_aware_first_layer_bounds_cover_all_feasible_attacks(self):
        costs_all = np.asarray([1.0, 1.35, 1.8])
        for depth, mode, q, budget in (
            (2, "legacy", 1, 2.0),
            (3, "heterogeneous-cost", 1, 1.7),
            (3, "legacy", 2, 3.0),
        ):
            model, adjacency, x = tiny_instance(70 + depth, depth, True)
            graph = build_computation_graph(adjacency, 1, depth)
            weights, biases = model.numpy_parameters()
            source_costs = costs_all[np.asarray(graph.candidate_sources)]
            lower, upper, summary = budget_aware_first_layer_bounds(
                graph, x.numpy(), weights[0], biases[0], q=q,
                global_budget=budget, source_costs=source_costs,
                weighted=mode in WEIGHTED_MODES,
            )
            local_clean = x.numpy()[np.asarray(graph.candidate_sources)]
            for mask in all_feasible_local_masks(
                len(graph.candidate_sources), x.shape[1], q, budget,
                source_costs, mode in WEIGHTED_MODES,
            ):
                attacked = local_clean + (1 - 2 * local_clean) * mask
                _, hidden_pre = evaluate_computation_graph(
                    graph, attacked, weights, biases
                )
                self.assertTrue(np.all(hidden_pre[0] >= lower - 2e-12))
                self.assertTrue(np.all(hidden_pre[0] <= upper + 2e-12))
            self.assertEqual(
                summary["method"],
                "per-neuron-exact-grouped-fractional-knapsack-v1",
            )

        # Independent continuous-LP check for weighted fractional use.
        benefits = [[5.0, 3.0, -2.0], [6.0, 1.0], [4.0, 2.0]]
        value = grouped_fractional_knapsack_max(
            benefits, [1.0, 2.0, 1.5], q=2,
            global_budget=2.25, weighted=True,
        )
        # Items are (5/1), (3/1), (6/2), (4/1.5), ...; select 5 and
        # 1.25/2 of benefit 6.
        self.assertGreaterEqual(value, 8.75)
        self.assertLess(value, 8.750000001)

    def test_normalization_seed_and_cache_identity(self):
        adjacency = tiny_graph().to_dense().numpy()
        np.testing.assert_allclose(adjacency, adjacency.T, atol=0, rtol=0)
        self.assertTrue(np.all(np.diag(adjacency) > 0))
        seed_all(123)
        first = DeepSigmoidGCN(2, [3], 2, depth=2, bias=True).double()
        seed_all(123)
        second = DeepSigmoidGCN(2, [3], 2, depth=2, bias=True).double()
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

        data = SimpleNamespace(
            num_nodes=3, num_features=2,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            x=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
            y=torch.tensor([0, 1, 0]),
        )
        split = (
            torch.tensor([1, 0, 0], dtype=torch.bool),
            torch.tensor([0, 1, 0], dtype=torch.bool),
            torch.tensor([0, 0, 1], dtype=torch.bool),
        )
        description = model_cache_description(
            dataset_name="synthetic", seed=0, model=first,
            hyperparameters=TrainingHyperparameters(10, .01, 0, 3),
            data=data, train_mask=split[0], val_mask=split[1],
            test_mask=split[2], device=torch.device("cpu"),
        )
        self.assertEqual(description["normalization_version"], NORMALIZATION_VERSION)
        self.assertEqual(description["graph_semantics_version"],
                         GRAPH_SEMANTICS_VERSION)
        self.assertIn("canonical_graph_sha256", description["data"])
        self.assertNotIn("edge_index_sha256", description["data"])
        self.assertIn("architecture", description)
        self.assertIn("training", description)
        self.assertIn("split_sha256", description)
        changed = model_cache_description(
            dataset_name="synthetic", seed=0, model=first,
            hyperparameters=TrainingHyperparameters(11, .01, 0, 3),
            data=data, train_mask=split[0], val_mask=split[1],
            test_mask=split[2], device=torch.device("cpu"),
        )
        self.assertNotEqual(description, changed)

        equivalent_encoding = SimpleNamespace(
            num_nodes=3, num_features=2,
            edge_index=torch.tensor([[0, 0, 1], [1, 1, 1]]),
            x=data.x, y=data.y,
        )
        equivalent = model_cache_description(
            dataset_name="synthetic", seed=0, model=first,
            hyperparameters=TrainingHyperparameters(10, .01, 0, 3),
            data=equivalent_encoding, train_mask=split[0], val_mask=split[1],
            test_mask=split[2], device=torch.device("cpu"),
        )
        self.assertEqual(description, equivalent)

        different_graph = SimpleNamespace(
            num_nodes=3, num_features=2,
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            x=data.x, y=data.y,
        )
        different = model_cache_description(
            dataset_name="synthetic", seed=0, model=first,
            hyperparameters=TrainingHyperparameters(10, .01, 0, 3),
            data=different_graph, train_mask=split[0], val_mask=split[1],
            test_mask=split[2], device=torch.device("cpu"),
        )
        self.assertNotEqual(description, different)


class DeepVerifierExhaustiveCorrectnessTests(unittest.TestCase):
    tolerance = 3e-5

    def assert_sound(self, model, adjacency, x, target, mode, q, budget,
                     costs, method, samples=12):
        exact = exhaustive_worst_margin(
            model, adjacency, x, target,
            mode=mode, q=q, global_budget=budget,
            node_costs=costs, max_variables=12,
        )
        verifier = DeepSigmoidVerifier(
            model, adjacency, x, node_costs=costs,
            method=method, samples=samples, seed=91,
            solver_timeout=8.0, solver_tolerance=2e-5,
        )
        result = verifier.verify(
            target, mode=mode, q=q, global_budget=budget, exact=exact
        )
        self.assertTrue(
            result.solver_trusted,
            result.as_dict(include_envelope_details=False),
        )
        self.assertIsNone(result.soundness_error)
        self.assertIsNotNone(result.lp_lower_bound)
        self.assertLessEqual(
            result.lp_lower_bound, exact.worst_margin + self.tolerance,
            result.as_dict(include_envelope_details=False),
        )
        if result.certified:
            self.assertTrue(exact.robust)
        return exact, result

    def test_randomized_depth_method_count_weighted_soundness(self):
        costs = [1.0, 1.35, 1.8]
        cases_checked = 0
        for seed in range(3):
            for depth in (2, 3):
                bias = bool((seed + depth) % 2)
                model, adjacency, x = tiny_instance(seed, depth, bias)
                if seed % 2:
                    mode, q, budget = "heterogeneous-cost", 1, 1.7
                else:
                    mode, q, budget = "legacy", 1 + (seed == 2), 2.0
                for method in ("baseline", "dual"):
                    self.assert_sound(
                        model, adjacency, x, seed % 3,
                        mode, q, budget, costs, method,
                    )
                    cases_checked += 1
        self.assertEqual(cases_checked, 12)

    def test_lp_certified_targets_are_subset_of_exhaustive_robust_targets(self):
        costs = [1.0, 1.35, 1.8]
        certified_total = 0
        for depth, method, mode in itertools.product(
                (2, 3), ("baseline", "dual"),
                ("legacy", "heterogeneous-cost")):
            model, adjacency, x = tiny_instance(20 + depth, depth, bias=True)
            budget = 1.0 if mode == "legacy" else 1.4
            exact_robust, lp_certified = set(), set()
            for target in range(3):
                exact, result = self.assert_sound(
                    model, adjacency, x, target, mode, 1, budget,
                    costs, method, samples=10,
                )
                if exact.robust:
                    exact_robust.add(target)
                if result.certified:
                    lp_certified.add(target)
            self.assertTrue(lp_certified <= exact_robust)
            certified_total += len(lp_certified)
        self.assertGreater(certified_total, 0)

    def test_zero_budget_q_zero_bias_and_forward_consistency(self):
        costs = [1.0, 1.35, 1.8]
        for depth, method, bias in itertools.product(
                (2, 3), ("baseline", "dual"), (False, True)):
            model, adjacency, x = tiny_instance(31 + depth, depth, bias)
            # Alternate whether zero is supplied through q or through Q.
            q, budget = ((0, 2.0) if bias else (1, 0.0))
            exact, result = self.assert_sound(
                model, adjacency, x, 1, "legacy", q, budget,
                costs, method, samples=8,
            )
            self.assertAlmostEqual(exact.worst_margin, exact.clean_margin,
                                   delta=1e-12)
            self.assertAlmostEqual(result.lp_lower_bound, exact.clean_margin,
                                   delta=self.tolerance)
            if method == "dual":
                self.assertEqual(
                    result.envelope_summary["used_under_neurons"], 0
                )
                self.assertGreater(
                    result.envelope_summary["fallback_counts"].get(
                        "under_domain_no_variation", 0
                    ), 0,
                )

    def test_known_nonrobust_targets_never_certified(self):
        costs = [1.0, 1.35, 1.8]
        # These fixed seeds have a positive clean margin but a non-positive
        # exhaustive worst margin under q=2,Q=4.
        for depth, seed in ((2, 14), (3, 24)):
            model, adjacency, x = tiny_instance(seed, depth, bool(seed % 2))
            exact = exhaustive_worst_margin(
                model, adjacency, x, 0, mode="legacy", q=2,
                global_budget=4, node_costs=costs,
            )
            self.assertGreater(exact.clean_margin, 0.0)
            self.assertFalse(exact.robust)
            for method in ("baseline", "dual"):
                verifier = DeepSigmoidVerifier(
                    model, adjacency, x, node_costs=costs,
                    method=method, samples=12, seed=5,
                    solver_timeout=8, solver_tolerance=2e-5,
                )
                result = verifier.verify(
                    0, mode="legacy", q=2, global_budget=4, exact=exact
                )
                self.assertTrue(result.solver_trusted)
                self.assertLessEqual(
                    result.lp_lower_bound, exact.worst_margin + self.tolerance
                )
                self.assertFalse(result.certified)

    def test_budget_box_abstraction_is_sound_and_removes_input_variables(self):
        costs = [1.0, 1.35, 1.8]
        for depth, method, mode in itertools.product(
                (2, 3), ("baseline", "dual"),
                ("legacy", "heterogeneous-cost")):
            model, adjacency, x = tiny_instance(83 + depth, depth, True)
            budget = 2.0 if mode == "legacy" else 2.2
            exact = exhaustive_worst_margin(
                model, adjacency, x, 1, mode=mode, q=1,
                global_budget=budget, node_costs=costs,
            )
            verifier = DeepSigmoidVerifier(
                model, adjacency, x, node_costs=costs, method=method,
                samples=12, seed=17, solver_timeout=8,
                solver_tolerance=2e-5, input_abstraction="budget-box",
            )
            result = verifier.verify(
                1, mode=mode, q=1, global_budget=budget, exact=exact
            )
            self.assertTrue(result.solver_trusted)
            self.assertEqual(result.input_abstraction, "budget-box")
            self.assertGreater(result.potential_input_variable_count, 0)
            self.assertLess(result.variable_count,
                            result.potential_input_variable_count + result.variable_count)
            self.assertLessEqual(
                result.lp_lower_bound, exact.worst_margin + self.tolerance
            )
            if result.certified:
                self.assertTrue(exact.robust)

    def test_nonoptimal_or_timeout_is_never_certified(self):
        model, adjacency, x = tiny_instance(44, 3, True)
        verifier = DeepSigmoidVerifier(
            model, adjacency, x, method="baseline", samples=0,
            solver_timeout=1e-12,
        )
        result = verifier.verify(
            1, mode="legacy", q=1, global_budget=1
        )
        self.assertFalse(result.solver_trusted)
        self.assertFalse(result.certified)
        self.assertIsNone(result.lp_lower_bound)
        self.assertIn("TimedOut", result.solver_status)

    def test_failed_primary_residual_audit_uses_unscaled_lp_retry(self):
        model, adjacency, x = tiny_instance(45, 2, True)
        verifier = DeepSigmoidVerifier(
            model, adjacency, x, method="baseline", samples=0,
            solver_timeout=8.0, solver_tolerance=1e-6,
        )
        original_audit = deep_verifier_module._audit_solution
        audit_calls = 0

        def fail_first_audit(lp_model, tolerance):
            nonlocal audit_calls
            audit_calls += 1
            if audit_calls == 1:
                return False, 1.7e-6, "forced primary residual failure"
            return original_audit(lp_model, tolerance)

        with patch.object(
            deep_verifier_module, "_audit_solution",
            side_effect=fail_first_audit,
        ):
            result = verifier.verify(
                1, mode="legacy", q=1, global_budget=1
            )

        self.assertGreaterEqual(audit_calls, 2)
        self.assertTrue(result.solver_trusted)
        self.assertTrue(result.rival_results)
        self.assertTrue(all(
            rival.solver_profile == CBC_AUDIT_RETRY_PROFILE
            for rival in result.rival_results
        ))
        self.assertTrue(all(
            rival.maximum_primal_violation is not None
            and rival.maximum_primal_violation <= verifier.solver_tolerance
            for rival in result.rival_results
        ))

    def test_same_outer_bounds_and_empirical_tightness_is_reported_not_assumed(self):
        model, adjacency, x = tiny_instance(55, 3, True)
        costs = [1.0, 1.35, 1.8]
        exact = exhaustive_worst_margin(
            model, adjacency, x, 1, mode="heterogeneous-cost", q=1,
            global_budget=1.7, node_costs=costs,
        )
        results = {}
        for method in ("baseline", "dual"):
            verifier = DeepSigmoidVerifier(
                model, adjacency, x, node_costs=costs,
                method=method, samples=16, seed=7,
                solver_timeout=8,
            )
            results[method] = verifier.verify(
                1, mode="heterogeneous-cost", q=1,
                global_budget=1.7, exact=exact,
            )
        self.assertEqual(results["baseline"].outer_bounds_summary,
                         results["dual"].outer_bounds_summary)
        delta = (results["dual"].lp_lower_bound
                 - results["baseline"].lp_lower_bound)
        self.assertTrue(math.isfinite(delta))
        # Dual contains every baseline line plus additional outer-calibrated
        # lines, hence its exact LP optimum cannot be lower. Allow only the
        # configured solver/numerical tolerance.
        self.assertGreaterEqual(delta, -self.tolerance)


if __name__ == "__main__":
    unittest.main()
