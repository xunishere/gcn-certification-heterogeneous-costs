"""Independent resumable CLI for deep sigmoid-GCN LP certification.

Unlike :mod:`VRP.real_experiments`, this entry point trains depth-2/depth-3
nonlinear models and invokes the sound affine-envelope LP verifier.  It runs
the synthetic exhaustive correctness gate before loading or training a real
dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import unittest
from typing import Sequence

import numpy as np

from .dataset_registry import (
    cli_dataset_name,
    dataset_spec,
    load_dataset,
    q0_values_for_dataset,
    q_values_for_dataset,
)
from .deep_gcn import (
    DEEP_MODEL_CACHE_VERSION,
    NORMALIZATION_VERSION,
    TrainingHyperparameters,
    choose_device,
    train_or_load_deep,
)
from .deep_verifier import (
    CBC_RANDOM_SEED,
    VERIFIER_SCHEMA_VERSION,
    DeepSigmoidVerifier,
    build_computation_graph,
)
from .experiment_config import MODES, PAPER_MODES
from .graph_semantics import GRAPH_SEMANTICS_VERSION
from .node_importance import attack_costs, compute_node_importance
from .real_experiments import masks, stratified_targets
from .risk_budget import (
    COST_BUDGET_POLICY,
    RISK_BUDGET_POLICY,
    allocate_fixed_mean_budgets,
    fixed_resource_budget,
    target_risk,
)

DEEP_EXPERIMENT_SCHEMA = "deep-sigmoid-experiment-v6"
CONFIG_SLUG_VERSION = "dsv6"
METHODS = ("baseline", "dual")

FIELDS = [
    "schema_version", "verifier_schema_version", "graph_semantics_version",
    "canonical_graph_sha256", "dataset", "target_scope",
    "seed", "method", "depth", "hidden_dims", "bias", "samples",
    "normalization_version", "model_cache_version", "model_cache_key",
    "model_cache_hit", "target_cohort_checksum", "cohort_risk_bin",
    "mode", "budget_policy", "cost_budget_policy", "risk_mapping_checksum",
    "q", "Q0", "Q_t", "Q_equiv", "Q_cost", "global_budget",
    "global_budget_unit", "mean_cost", "uses_heterogeneous_cost",
    "target_cost", "target", "label", "prediction", "confidence",
    "importance", "risk_weight", "candidate_source_count",
    "candidate_sources", "candidate_source_costs", "clean_margin",
    "exact_worst_margin",
    "lp_lower_bound", "approximation_gap", "certified", "correct",
    "certified_and_correct", "solver_status", "solver_trusted",
    "solver_random_seed",
    "rival_solver_results", "soundness_error", "variable_count",
    "constraint_count", "input_abstraction", "potential_input_variable_count",
    "input_variable_reduction_ratio", "first_layer_bound_seconds",
    "bounds_cache_hit", "solver_seconds", "total_seconds",
    "under_generated_samples", "under_unique_samples", "under_used_neurons",
    "under_fallback_counts", "lower_source_counts", "upper_source_counts",
    "mean_selected_lower_gap", "mean_selected_upper_gap",
    "outer_bounds_sha256", "diagnostics_path", "clean_train_accuracy",
    "clean_validation_accuracy", "clean_test_accuracy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable_json(path: Path, payload) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable resume manifest mismatch: {path}")
    else:
        atomic_json(path, payload)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _sha256_json(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def hidden_dims_for_depth(values, depth: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in values)
    if len(values) == 1:
        return values * (depth - 1)
    if len(values) < depth - 1:
        raise ValueError(f"depth {depth} needs {depth-1} hidden dimensions")
    return values[:depth - 1]


def _count_records(records: Sequence[dict], fields: Sequence[str]) -> dict:
    counts = Counter(tuple(record[field] for field in fields) for record in records)
    return {
        "|".join(str(value) for value in key): count
        for key, count in sorted(counts.items())
    }


def select_targets(model_run, dataset_name: str, target_scope: str,
                   sample_size: int, seed: int,
                   *, importance: Sequence[float], eta: float):
    """Select a fixed auditable cohort, nested as ``sample_size`` grows.

    ``test-stratified`` is restricted to the standard test mask and greedily
    covers true class, clean correctness, and terciles of cohort risk before
    matching the population's joint-stratum proportions. Ties and within-
    stratum order are deterministic for ``(dataset, seed, depth model)``.
    """
    labels = model_run.y.detach().cpu().numpy()
    predictions = model_run.prediction.detach().cpu().numpy()
    probabilities = model_run.probability.detach().cpu().numpy()
    if target_scope == "test-stratified":
        pool = sorted(int(value) for value in
                      model_run.test_mask.nonzero().flatten().tolist())
        if sample_size > len(pool):
            raise ValueError("sample size exceeds the standard test mask")
        risks = {
            target: target_risk(
                importance[target],
                float(probabilities[target, predictions[target]]), eta,
            ) for target in pool
        }
        risk_order = sorted(pool, key=lambda target: (risks[target], target))
        risk_bins = {
            target: min(2, rank * 3 // max(1, len(pool)))
            for rank, target in enumerate(risk_order)
        }
        records = [{
            "target": target,
            "label": int(labels[target]),
            "correct": int(predictions[target] == labels[target]),
            "risk": risks[target],
            "risk_bin": risk_bins[target],
        } for target in pool]
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for record in records:
            key = (record["label"], record["correct"], record["risk_bin"])
            grouped.setdefault(key, []).append(record["target"])
        stable_seed = int(hashlib.sha256(
            f"{dataset_name}|{seed}|{model_run.cache_key}".encode("utf-8")
        ).hexdigest()[:16], 16)
        generator = np.random.default_rng(stable_seed)
        tie_order = {}
        for key in sorted(grouped):
            values = np.asarray(grouped[key], dtype=int)
            grouped[key] = [int(value) for value in
                            values[generator.permutation(len(values))]]
            tie_order[key] = float(generator.random())
        selected: list[int] = []
        selected_by_group = Counter()
        covered_class, covered_correct, covered_risk = set(), set(), set()
        population = Counter(
            (record["label"], record["correct"], record["risk_bin"])
            for record in records
        )
        while len(selected) < sample_size:
            choices = []
            for key, values in grouped.items():
                used = selected_by_group[key]
                if used >= len(values):
                    continue
                label, correct, risk_bin = key
                coverage = (1000 * (label not in covered_class)
                            + 500 * (correct not in covered_correct)
                            + 250 * (risk_bin not in covered_risk))
                deficit = (population[key] / len(pool) * (len(selected) + 1)
                           - selected_by_group[key])
                choices.append((coverage + deficit, -tie_order[key], key))
            if not choices:
                raise RuntimeError("cohort stratification exhausted unexpectedly")
            _, _, key = max(choices)
            target = grouped[key][selected_by_group[key]]
            selected_by_group[key] += 1
            selected.append(target)
            covered_class.add(key[0])
            covered_correct.add(key[1])
            covered_risk.add(key[2])
        targets = sorted(selected)
        selected_set = set(targets)
        selected_records = [record for record in records
                            if record["target"] in selected_set]
        scope_label = (
            f"stratified-{len(targets)}-from-planetoid_test_mask-{len(pool)}"
            if dataset_spec(dataset_name).family == "Planetoid"
            else f"stratified-{len(targets)}-from-test_mask-{len(pool)}"
        )
        audit = {
            "policy": "nested-greedy-class-correct-risk-tercile-v1",
            "pool_size": len(pool),
            "requested_size": sample_size,
            "actual_size": len(targets),
            "pool_counts_class_correct_risk": _count_records(
                records, ("label", "correct", "risk_bin")
            ),
            "selected_counts_class_correct_risk": _count_records(
                selected_records, ("label", "correct", "risk_bin")
            ),
            "records": selected_records,
        }
        audit["cohort_checksum_sha256"] = _sha256_json({
            "policy": audit["policy"], "targets": targets,
            "records": selected_records,
        })
        return targets, len(pool), scope_label, audit, risk_bins

    if target_scope == "test":
        targets = sorted(int(value) for value in
                         model_run.test_mask.nonzero().flatten().tolist())
        label = ("planetoid_test_mask"
                 if dataset_spec(dataset_name).family == "Planetoid"
                 else "test_mask")
        pool_size = len(targets)
    elif target_scope == "all":
        targets, pool_size, label = list(range(len(labels))), len(labels), "all_nodes"
    else:
        targets = stratified_targets(labels, sample_size, seed)
        pool_size, label = len(labels), f"stratified-{len(targets)}-from-all_nodes"
    risk_bins = {target: 0 for target in targets}
    audit = {
        "policy": f"legacy-{target_scope}", "pool_size": pool_size,
        "requested_size": len(targets), "actual_size": len(targets),
        "records": [{"target": target, "risk_bin": 0} for target in targets],
    }
    audit["cohort_checksum_sha256"] = _sha256_json({
        "policy": audit["policy"], "targets": targets
    })
    return targets, pool_size, label, audit, risk_bins


def run_synthetic_gate(output: Path, verbosity: int = 1) -> dict:
    """Run the mandatory exhaustive/soundness tests before real data."""
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "VRP.test_deep_verifier"
    )
    started = time.perf_counter()
    result = unittest.TextTestRunner(
        stream=stream, verbosity=verbosity, failfast=True
    ).run(suite)
    payload = {
        "schema_version": DEEP_EXPERIMENT_SCHEMA,
        "updated_at": utc_now(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "seconds": time.perf_counter() - started,
        "log": stream.getvalue(),
    }
    history_path = (output / "synthetic-gate-history"
                    / f"gate-{time.time_ns()}.json")
    payload["history_path"] = str(history_path)
    atomic_json(history_path, payload)
    atomic_json(output / "synthetic_gate.json", payload)
    if not result.wasSuccessful():
        raise RuntimeError(
            "synthetic exhaustive correctness gate failed; real experiment blocked"
        )
    return payload


def _config_slug(dataset: str, seed: int, depth: int,
                 hidden_dims: tuple[int, ...], method: str, mode: str,
                 q: int, q0: int, samples: int) -> Path:
    hidden = "x".join(str(value) for value in hidden_dims)
    mode_slug = {
        "legacy": "leg", "heterogeneous-cost": "het",
        "risk-budget": "risk", "combined": "comb",
    }[mode]
    method_slug = {"baseline": "base", "dual": "dual"}[method]
    # Compact slugs keep diagnostics below the legacy Windows MAX_PATH limit.
    return (Path("c") / CONFIG_SLUG_VERSION / dataset / f"s{seed}"
            / f"d{depth}-h{hidden}" / f"n{samples}" / method_slug / mode_slug
            / f"q{q}" / f"Q{q0}")


def _read_done_targets(csv_path: Path, checksum: str, cache_key: str,
                       cohort_checksum: str) -> set[int]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("schema_version") != DEEP_EXPERIMENT_SCHEMA:
                raise RuntimeError(f"incompatible deep result schema: {csv_path}")
            if row.get("risk_mapping_checksum") != checksum:
                raise RuntimeError(f"risk mapping mismatch on resume: {csv_path}")
            if row.get("model_cache_key") != cache_key:
                raise RuntimeError(f"model cache mismatch on resume: {csv_path}")
            if row.get("target_cohort_checksum") != cohort_checksum:
                raise RuntimeError(f"target cohort mismatch on resume: {csv_path}")
            target = int(row["target"])
            if target in done:
                raise RuntimeError(f"duplicate target in {csv_path}: {target}")
            done.add(target)
    return done


def _diagnostic_path(output: Path, slug: Path, target: int) -> Path:
    compact = Path(*slug.parts[1:]) if slug.parts and slug.parts[0] == "c" else slug
    return output / "d" / compact / f"t{target}.json"


def run_one_config(
    args,
    output: Path,
    *,
    dataset_name: str,
    seed: int,
    depth: int,
    hidden_dims: tuple[int, ...],
    method: str,
    mode: str,
    q: int,
    q0: int,
    targets: list[int],
    target_scope: str,
    data,
    model_run,
    verifier: DeepSigmoidVerifier,
    importance: Sequence[float],
    heterogeneous_costs: Sequence[float],
    allocation,
    allocation_path: Path,
    cohort_checksum: str,
    cohort_risk_bins: dict[int, int],
) -> tuple[str, Path, float]:
    slug = _config_slug(dataset_name, seed, depth, hidden_dims,
                        method, mode, q, q0, args.samples)
    csv_path = output / slug.with_suffix(".csv")
    status_path = output / slug.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_targets(
        csv_path, allocation.checksum, model_run.cache_key, cohort_checksum
    )
    unexpected = done - set(targets)
    if unexpected:
        raise RuntimeError(f"resume CSV contains unexpected targets: {unexpected}")
    if done == set(targets):
        repaired = (json.loads(status_path.read_text(encoding="utf-8"))
                    if status_path.exists() else {})
        repaired.update({
            "schema_version": DEEP_EXPERIMENT_SCHEMA,
            "status": "done",
            "updated_at": utc_now(),
            "completed_targets": len(done),
            "expected_targets": len(targets),
            "csv": str(csv_path),
            "risk_mapping_checksum": allocation.checksum,
            "model_cache_key": model_run.cache_key,
            "error": None,
        })
        atomic_json(status_path, repaired)
        return "done", csv_path, 0.0

    status = {
        "schema_version": DEEP_EXPERIMENT_SCHEMA,
        "status": "running",
        "updated_at": utc_now(),
        "dataset": dataset_name,
        "seed": seed,
        "depth": depth,
        "hidden_dims": list(hidden_dims),
        "method": method,
        "samples": args.samples,
        "mode": mode,
        "q": q,
        "Q0": q0,
        "model_cache_key": model_run.cache_key,
        "target_cohort_checksum": cohort_checksum,
        "risk_mapping_checksum": allocation.checksum,
        "risk_mapping_path": str(allocation_path),
        "completed_targets": len(done),
        "expected_targets": len(targets),
        "csv": str(csv_path),
        "error": None,
    }
    atomic_json(status_path, status)
    append = csv_path.exists() and csv_path.stat().st_size > 0
    handle = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if not append:
        writer.writeheader()
        handle.flush()
    started = time.perf_counter()
    deadline = (started + args.config_timeout_seconds
                if args.config_timeout_seconds > 0 else None)
    final_status = "done"
    processed_this_attempt = 0
    try:
        for target in targets:
            if target in done:
                continue
            if (args.invocation_target_limit > 0
                    and processed_this_attempt >= args.invocation_target_limit):
                final_status = "paused"
                status["error"] = None
                break
            if deadline is not None and time.perf_counter() >= deadline:
                final_status = "timed_out"
                status["error"] = "configuration wall-clock deadline exceeded"
                break
            graph = build_computation_graph(model_run.adjacency, target, depth)
            candidate_costs = [heterogeneous_costs[node]
                               for node in graph.candidate_sources]
            candidate_mean = math.fsum(candidate_costs) / len(candidate_costs)
            canonical_graph_hash = model_run.cache_description[
                "data"
            ]["canonical_graph_sha256"]
            use_risk = mode in ("risk-budget", "combined")
            use_cost = mode in ("heterogeneous-cost", "combined")
            budget = fixed_resource_budget(
                q0,
                use_risk=use_risk,
                use_cost=use_cost,
                allocated_q_t=allocation.budgets[target],
            )
            result = verifier.verify(
                target, mode=mode, q=q,
                global_budget=budget["global_budget"],
            )
            if not result.solver_trusted:
                errors = "; ".join(
                    rival.error or rival.status
                    for rival in result.rival_results if not rival.trusted
                )
                raise RuntimeError(
                    f"untrusted solver result for target {target}: {errors}"
                )
            if result.prediction != int(model_run.prediction[target]):
                raise RuntimeError(
                    "full training forward and local verifier prediction disagree"
                )
            if tuple(result.candidate_sources) != tuple(graph.candidate_sources):
                raise RuntimeError(
                    "persisted candidate cost order disagrees with verifier sources"
                )
            confidence = float(model_run.probability[
                target, result.prediction
            ].detach().cpu())
            risk = target_risk(
                importance[target], confidence, args.eta
            )
            correct = int(result.prediction == int(model_run.y[target]))
            detailed = {
                "schema_version": DEEP_EXPERIMENT_SCHEMA,
                "dataset": dataset_name,
                "seed": seed,
                "method": method,
                "depth": depth,
                "hidden_dims": list(hidden_dims),
                "mode": mode,
                "q": q,
                "Q0": q0,
                "budget": budget,
                "risk_mapping_checksum": allocation.checksum,
                "model_cache_key": model_run.cache_key,
                "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                "canonical_graph_sha256": canonical_graph_hash,
                "attack_cost_audit": {
                    "uses_heterogeneous_cost": use_cost,
                    "target_cost": heterogeneous_costs[target],
                    "candidate_sources": list(graph.candidate_sources),
                    "candidate_source_costs": candidate_costs,
                },
                "verification": result.as_dict(include_envelope_details=True),
                "solver_random_seed": CBC_RANDOM_SEED,
            }
            diagnostics_path = _diagnostic_path(output, slug, target)
            # A crash can occur after the detailed JSON is committed but before
            # its CSV row is flushed. Preserve that orphan rather than
            # overwriting history, and attach the retried row to a new suffix.
            if diagnostics_path.exists():
                retry = 1
                while True:
                    candidate = diagnostics_path.with_name(
                        f"target-{target}-retry-{retry}.json"
                    )
                    if not candidate.exists():
                        diagnostics_path = candidate
                        break
                    retry += 1
                detailed["resumed_after_orphan_diagnostic"] = True
            atomic_json(diagnostics_path, detailed)
            under = result.under_domain_summary or {}
            envelope = result.envelope_summary
            outer_hash = _sha256_json(result.outer_bounds_summary)
            row = {
                "schema_version": DEEP_EXPERIMENT_SCHEMA,
                "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
                "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                "canonical_graph_sha256": canonical_graph_hash,
                "dataset": dataset_name,
                "target_scope": target_scope,
                "seed": seed,
                "method": method,
                "depth": depth,
                "hidden_dims": _json(list(hidden_dims)),
                "bias": int(args.bias),
                "samples": args.samples,
                "normalization_version": NORMALIZATION_VERSION,
                "model_cache_version": DEEP_MODEL_CACHE_VERSION,
                "model_cache_key": model_run.cache_key,
                "model_cache_hit": int(model_run.cache_hit),
                "target_cohort_checksum": cohort_checksum,
                "cohort_risk_bin": cohort_risk_bins[target],
                "mode": mode,
                "budget_policy": RISK_BUDGET_POLICY,
                "cost_budget_policy": COST_BUDGET_POLICY,
                "risk_mapping_checksum": allocation.checksum,
                "q": q,
                "Q0": q0,
                "Q_t": budget["Q_t"],
                "Q_equiv": budget["Q_equiv"],
                "Q_cost": budget["Q_cost"],
                "global_budget": budget["global_budget"],
                "global_budget_unit": budget["global_budget_unit"],
                "mean_cost": candidate_mean if use_cost else 1.0,
                "uses_heterogeneous_cost": int(use_cost),
                "target_cost": heterogeneous_costs[target],
                "target": target,
                "label": int(model_run.y[target]),
                "prediction": result.prediction,
                "confidence": confidence,
                "importance": importance[target],
                "risk_weight": risk,
                "candidate_source_count": result.candidate_source_count,
                "candidate_sources": _json(result.candidate_sources),
                "candidate_source_costs": _json(candidate_costs),
                "clean_margin": result.clean_margin,
                "exact_worst_margin": "",
                "lp_lower_bound": ("" if result.lp_lower_bound is None
                                   else result.lp_lower_bound),
                "approximation_gap": ("" if result.approximation_gap is None
                                      else result.approximation_gap),
                "certified": int(result.certified),
                "correct": correct,
                "certified_and_correct": int(result.certified) * correct,
                "solver_status": result.solver_status,
                "solver_trusted": int(result.solver_trusted),
                "solver_random_seed": CBC_RANDOM_SEED,
                "rival_solver_results": _json([
                    rival.as_dict() for rival in result.rival_results
                ]),
                "soundness_error": result.soundness_error or "",
                "variable_count": result.variable_count,
                "constraint_count": result.constraint_count,
                "input_abstraction": result.input_abstraction,
                "potential_input_variable_count": result.potential_input_variable_count,
                "input_variable_reduction_ratio": (
                    1.0 - result.variable_count
                    / max(1, result.potential_input_variable_count
                          + result.variable_count)
                ),
                "first_layer_bound_seconds": result.first_layer_bound_seconds,
                "bounds_cache_hit": int(result.bounds_cache_hit),
                "solver_seconds": result.solver_seconds,
                "total_seconds": result.total_seconds,
                "under_generated_samples": under.get("generated_attack_samples", 0),
                "under_unique_samples": under.get("unique_attack_samples", 0),
                "under_used_neurons": envelope["used_under_neurons"],
                "under_fallback_counts": _json(envelope["fallback_counts"]),
                "lower_source_counts": _json(envelope["lower_source_counts"]),
                "upper_source_counts": _json(envelope["upper_source_counts"]),
                "mean_selected_lower_gap": envelope["mean_selected_lower_gap"],
                "mean_selected_upper_gap": envelope["mean_selected_upper_gap"],
                "outer_bounds_sha256": outer_hash,
                "diagnostics_path": str(diagnostics_path),
                "clean_train_accuracy": model_run.clean_train_accuracy,
                "clean_validation_accuracy": model_run.clean_validation_accuracy,
                "clean_test_accuracy": model_run.clean_test_accuracy,
            }
            writer.writerow(row)
            handle.flush()
            done.add(target)
            processed_this_attempt += 1
            status["completed_targets"] = len(done)
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
    except Exception as error:
        final_status = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        handle.close()
        status["status"] = final_status
        status["updated_at"] = utc_now()
        status["attempt_seconds"] = time.perf_counter() - started
        status["completed_targets"] = len(done)
        atomic_json(status_path, status)
    return final_status, csv_path, time.perf_counter() - started


def consolidate(output: Path, paths: list[Path]) -> int:
    destination = output / "nodes.csv"
    temporary = destination.with_suffix(".csv.tmp")
    rows = 0
    with temporary.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as source_handle:
                for row in csv.DictReader(source_handle):
                    writer.writerow({field: row.get(field, "") for field in FIELDS})
                    rows += 1
    temporary.replace(destination)
    return rows


def compare_tightness(output: Path) -> dict:
    nodes_path = output / "nodes.csv"
    rows = list(csv.DictReader(nodes_path.open(encoding="utf-8", newline="")))
    key_fields = [
        "dataset", "target_scope", "seed", "depth", "hidden_dims", "bias",
        "samples", "mode", "q", "Q0", "target", "model_cache_key",
    ]
    paired = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        paired.setdefault(key, {})[row["method"]] = row
    comparisons = []
    counts = Counter()
    tolerance = 1e-6
    for key, methods in sorted(paired.items()):
        if not all(method in methods for method in METHODS):
            continue
        baseline, dual = methods["baseline"], methods["dual"]
        outer_match = baseline["outer_bounds_sha256"] == dual["outer_bounds_sha256"]
        trusted = (baseline["solver_trusted"] == "1"
                   and dual["solver_trusted"] == "1"
                   and baseline["lp_lower_bound"] != ""
                   and dual["lp_lower_bound"] != "")
        delta = None
        outcome = "untrusted"
        if trusted:
            delta = (float(dual["lp_lower_bound"])
                     - float(baseline["lp_lower_bound"]))
            if delta > tolerance:
                outcome = "dual_tighter"
            elif delta < -tolerance:
                outcome = "dual_monotonicity_violation"
            else:
                outcome = "equal"
        if not outer_match:
            outcome = "outer_mismatch"
        counts[outcome] += 1
        record = dict(zip(key_fields, key))
        record.update({
            "outer_bounds_match": int(outer_match),
            "baseline_lp_lower": baseline["lp_lower_bound"],
            "dual_lp_lower": dual["lp_lower_bound"],
            "dual_minus_baseline": "" if delta is None else delta,
            "baseline_certified": baseline["certified"],
            "dual_certified": dual["certified"],
            "outcome": outcome,
        })
        comparisons.append(record)
    fields = key_fields + [
        "outer_bounds_match", "baseline_lp_lower", "dual_lp_lower",
        "dual_minus_baseline", "baseline_certified", "dual_certified", "outcome",
    ]
    path = output / "tightness.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {
        "comparison_pairs": len(comparisons),
        "outcome_counts": dict(sorted(counts.items())),
        "all_outer_bounds_match": all(
            record["outer_bounds_match"] for record in comparisons
        ),
        "claim": (
            "Dual intersects every baseline outer line with additional sound "
            "under-proposed lines, so exact LP monotonicity is required; a "
            "negative delta beyond tolerance is a numerical integrity failure."
        ),
        "csv": str(path),
    }
    atomic_json(output / "tightness_summary.json", summary)
    return summary


def _q_values_for_run(args, dataset_name):
    return q_values_for_dataset(
        dataset_name, explicit_q=args.q, sensitivity=args.q_sensitivity
    )


def _q0_values_for_run(args, dataset_name):
    return q0_values_for_dataset(
        dataset_name, explicit_q0=args.q0, sensitivity=args.q0_sensitivity
    )


def run(args) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gate = run_synthetic_gate(output, args.gate_verbosity)
    if args.synthetic_only:
        summary = {
            "schema_version": DEEP_EXPERIMENT_SCHEMA,
            "status": "synthetic_only_passed",
            "updated_at": utc_now(),
            "synthetic_gate": gate,
        }
        atomic_json(output / "run_summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return summary

    device = choose_device(args.device)
    architecture = {
        str(depth): list(hidden_dims_for_depth(args.hidden_dims, depth))
        for depth in args.depths
    }
    run_config = {
        "schema_version": DEEP_EXPERIMENT_SCHEMA,
        "config_slug_version": CONFIG_SLUG_VERSION,
        "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "canonical_graph_rule": (
            "remove input self-loops, collapse duplicates, symmetrize every "
            "edge, then add exactly one GCN self-loop per node"
        ),
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "depths": list(args.depths),
        "hidden_dims_by_depth": architecture,
        "bias": args.bias,
        "methods": list(args.methods),
        "samples": args.samples,
        "modes": list(args.modes),
        "q_values_by_dataset": {
            name: list(_q_values_for_run(args, name)) for name in args.datasets
        },
        "Q0_values_by_dataset": {
            name: list(_q0_values_for_run(args, name)) for name in args.datasets
        },
        "target_scope": args.target_scope,
        "sample_size": args.sample_size,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "device": str(device),
        },
        "alpha": args.alpha,
        "eta": args.eta,
        "beta": args.beta,
        "importance_weights": list(args.importance_weights),
        "solver_timeout": args.solver_timeout,
        "solver_tolerance": args.solver_tolerance,
        "solver_random_seed": CBC_RANDOM_SEED,
        "input_abstraction": args.input_abstraction,
        "explicit_input_max_variables": args.explicit_input_max_variables,
        "first_layer_bounds": "per-neuron-exact-grouped-fractional-knapsack-v1",
        "risk_budget_policy": RISK_BUDGET_POLICY,
        "cost_budget_policy": COST_BUDGET_POLICY,
        "fixed_cost_budget": (
            "heterogeneous-cost: Q_cost=Q0; combined: Q_cost=Q_t; "
            "candidate mean cost is audit metadata only"
        ),
        "local_constraint": "sum_j z[v,j] <= q",
        "count_global_modes": ["legacy", "risk-budget"],
        "weighted_global_modes": ["heterogeneous-cost", "combined"],
        "candidate_sources": "exact depth-hop normalized-adjacency receptive set including self",
    }
    immutable_json(output / "run_config.json", run_config)

    hyperparameters = TrainingHyperparameters(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    cache_dir = Path(args.cache_dir)
    model_cache_dir = (Path(args.model_cache_dir) if args.model_cache_dir
                       else output / "model-cache")
    expected_paths = []
    statuses = Counter()
    model_records = []
    started = time.perf_counter()

    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, cache_dir)
        data = dataset[0]
        edges = data.edge_index.t().tolist()
        importance = compute_node_importance(
            int(data.num_nodes), edges, args.importance_weights
        )
        heterogeneous_costs = attack_costs(importance, args.alpha)
        sample_size = (args.sample_size if args.sample_size is not None
                       else dataset_spec(dataset_name).default_sample_size)
        q_values = _q_values_for_run(args, dataset_name)
        q0_values = _q0_values_for_run(args, dataset_name)
        for seed in args.seeds:
            split_masks = masks(data, seed)
            for depth in args.depths:
                hidden_dims = hidden_dims_for_depth(args.hidden_dims, depth)
                model_run = train_or_load_deep(
                    data, dataset_name, seed, device,
                    depth=depth, hidden_dims=hidden_dims, bias=args.bias,
                    hyperparameters=hyperparameters, masks=split_masks,
                    model_cache_dir=model_cache_dir,
                )
                (targets, target_pool_size, target_scope, cohort_audit,
                 cohort_risk_bins) = select_targets(
                    model_run, dataset_name, args.target_scope,
                    sample_size, seed, importance=importance, eta=args.eta,
                )
                if not targets:
                    raise RuntimeError("selected target set is empty")
                target_manifest = {
                    "schema_version": DEEP_EXPERIMENT_SCHEMA,
                    "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                    "canonical_graph_sha256": model_run.cache_description[
                        "data"
                    ]["canonical_graph_sha256"],
                    "dataset": dataset_name,
                    "seed": seed,
                    "depth": depth,
                    "hidden_dims": list(hidden_dims),
                    "model_cache_key": model_run.cache_key,
                    "target_scope": target_scope,
                    "target_pool_size": target_pool_size,
                    "targets": targets,
                    "cohort": cohort_audit,
                }
                target_path = (output / "targets" / dataset_name
                               / f"seed-{seed}" / f"depth-{depth}.json")
                immutable_json(target_path, target_manifest)
                risks = {
                    target: target_risk(
                        importance[target],
                        float(model_run.probability[
                            target, model_run.prediction[target]
                        ].detach().cpu()),
                        args.eta,
                    )
                    for target in targets
                }
                allocations = {}
                allocation_paths = {}
                for q0 in q0_values:
                    allocation = allocate_fixed_mean_budgets(
                        risks, q0, args.beta
                    )
                    if allocation.total != len(targets) * q0:
                        raise RuntimeError("fixed-mean risk mapping failed conservation")
                    mapping_path = (output / "budget-mappings" / dataset_name
                                    / f"seed-{seed}" / f"depth-{depth}"
                                    / f"Q0-{q0}.json")
                    manifest = {
                        "schema_version": DEEP_EXPERIMENT_SCHEMA,
                        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                        "canonical_graph_sha256": model_run.cache_description[
                            "data"
                        ]["canonical_graph_sha256"],
                        "dataset": dataset_name,
                        "seed": seed,
                        "depth": depth,
                        "hidden_dims": list(hidden_dims),
                        "model_cache_key": model_run.cache_key,
                        "target_scope": target_scope,
                        "eta": args.eta,
                        **allocation.as_manifest(),
                    }
                    immutable_json(mapping_path, manifest)
                    allocations[q0] = allocation
                    allocation_paths[q0] = mapping_path
                model_records.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "depth": depth,
                    "hidden_dims": list(hidden_dims),
                    "bias": args.bias,
                    "cache_key": model_run.cache_key,
                    "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                    "canonical_graph_sha256": model_run.cache_description[
                        "data"
                    ]["canonical_graph_sha256"],
                    "cache_path": model_run.cache_path,
                    "cache_hit": model_run.cache_hit,
                    "best_epoch": model_run.best_epoch,
                    "train_seconds": model_run.train_seconds,
                    "clean_train_accuracy": model_run.clean_train_accuracy,
                    "clean_validation_accuracy": model_run.clean_validation_accuracy,
                    "clean_test_accuracy": model_run.clean_test_accuracy,
                    "targets": len(targets),
                })
                shared_bounds_cache = {}
                verifiers = {
                    method: DeepSigmoidVerifier(
                        model_run.model, model_run.adjacency, model_run.x,
                        node_costs=heterogeneous_costs,
                        method=method, samples=args.samples,
                        seed=seed, solver_timeout=args.solver_timeout,
                        solver_tolerance=args.solver_tolerance,
                        input_abstraction=args.input_abstraction,
                        explicit_input_max_variables=(
                            args.explicit_input_max_variables
                        ),
                        shared_bounds_cache=shared_bounds_cache,
                    )
                    for method in args.methods
                }
                for method in args.methods:
                    for mode in args.modes:
                        for q in q_values:
                            for q0 in q0_values:
                                status, path, _ = run_one_config(
                                    args, output,
                                    dataset_name=dataset_name, seed=seed,
                                    depth=depth, hidden_dims=hidden_dims,
                                    method=method, mode=mode, q=q, q0=q0,
                                    targets=targets, target_scope=target_scope,
                                    data=data, model_run=model_run,
                                    verifier=verifiers[method],
                                    importance=importance,
                                    heterogeneous_costs=heterogeneous_costs,
                                    allocation=allocations[q0],
                                    allocation_path=allocation_paths[q0],
                                    cohort_checksum=cohort_audit[
                                        "cohort_checksum_sha256"
                                    ],
                                    cohort_risk_bins=cohort_risk_bins,
                                )
                                statuses[status] += 1
                                expected_paths.append(path)

    row_count = consolidate(output, expected_paths)
    tightness = compare_tightness(output)
    summary_path = output / "run_summary.json"
    previous_summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                        if summary_path.exists() else {})
    updated_at = utc_now()
    wall_seconds = time.perf_counter() - started
    invocation = {
        "updated_at": updated_at,
        "wall_seconds": wall_seconds,
        "configuration_status_counts": dict(sorted(statuses.items())),
        "rows": row_count,
    }
    invocations = list(previous_summary.get("invocations", []))
    if previous_summary and not invocations:
        invocations.append({
            "updated_at": previous_summary.get("updated_at"),
            "wall_seconds": previous_summary.get("wall_seconds"),
            "configuration_status_counts": previous_summary.get(
                "configuration_status_counts", {}
            ),
            "rows": previous_summary.get("rows"),
            "migrated_from_summary_without_history": True,
        })
    invocations.append(invocation)
    summary = {
        "schema_version": DEEP_EXPERIMENT_SCHEMA,
        "status": ("done" if not statuses.get("failed")
                   and not statuses.get("timed_out")
                   and not statuses.get("paused") else "partial"),
        "updated_at": updated_at,
        "synthetic_gate": gate,
        "configuration_status_counts": dict(sorted(statuses.items())),
        "rows": row_count,
        "models": model_records,
        "tightness": tightness,
        "wall_seconds": wall_seconds,
        "invocations": invocations,
        "nodes_csv": str(output / "nodes.csv"),
        "cost_budget_policy": COST_BUDGET_POLICY,
        "solver_random_seed": CBC_RANDOM_SEED,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=cli_dataset_name,
                        default=["Karate"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--depths", nargs="+", type=int, choices=(2, 3),
                        default=[2, 3])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[16])
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--samples", type=int, default=128,
                        help="feasible Monte Carlo attack samples; clean is added")
    parser.add_argument("--modes", nargs="+", choices=PAPER_MODES,
                        default=list(PAPER_MODES))
    parser.add_argument(
        "--q", nargs="+", type=int, default=None,
        help=("explicit local count values for every selected dataset; default "
              "floor(0.01*D): Cora 14, CiteSeer 37, Karate 1"),
    )
    parser.add_argument(
        "--q-sensitivity", action="store_true",
        help=("use registry local-q values; the paper protocol fixes "
              "Cora 14 and CiteSeer 37"),
    )
    parser.add_argument(
        "--q0", nargs="+", type=int, default=None,
        help=("explicit global-budget values for every selected dataset; default "
              "Cora 14, CiteSeer 37, Karate 1"),
    )
    parser.add_argument(
        "--q0-sensitivity", action="store_true",
        help=("use Q=floor(lambda*q), lambda={.25,.5,1,2}: "
              "Cora 3,7,14,28; CiteSeer 9,18,37,74"),
    )
    parser.add_argument(
        "--target-scope",
        choices=("test-stratified", "stratified", "test", "all"),
        default="stratified",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--cache-dir", default=".cache/pyg")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--solver-timeout", type=float, default=30.0,
                        help="total CBC wall deadline per target")
    parser.add_argument("--solver-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--input-abstraction", choices=("auto", "explicit", "budget-box"),
        default="auto",
    )
    parser.add_argument("--explicit-input-max-variables", type=int,
                        default=50_000)
    parser.add_argument("--config-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--invocation-target-limit", type=int, default=0,
        help="operational resume control: process at most N new targets/config",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--importance-weights", nargs=3, type=float,
                        default=[.4, .4, .2])
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--gate-verbosity", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets must be unique after alias normalization")
    for values, name in ((args.seeds, "seeds"), (args.depths, "depths"),
                         (args.methods, "methods"), (args.modes, "modes")):
        if len(set(values)) != len(values):
            parser.error(f"{name} values must be unique")
    if args.q is not None and len(set(args.q)) != len(args.q):
        parser.error("q values must be unique")
    if args.q is not None and args.q_sensitivity:
        parser.error("choose explicit --q or --q-sensitivity, not both")
    if args.q0 is not None and len(set(args.q0)) != len(args.q0):
        parser.error("q0 values must be unique")
    if args.q0 is not None and args.q0_sensitivity:
        parser.error("choose explicit --q0 or --q0-sensitivity, not both")
    if any(value <= 0 for value in args.hidden_dims):
        parser.error("hidden dimensions must be positive")
    if (args.samples < 0
            or (args.q0 is not None and any(value < 0 for value in args.q0))):
        parser.error("samples and Q0 must be non-negative")
    if args.q is not None and any(value < 0 for value in args.q):
        parser.error("q must be non-negative")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("sample size must be positive")
    if (args.epochs <= 0 or args.learning_rate <= 0 or args.weight_decay < 0
            or args.patience <= 0):
        parser.error("invalid training hyperparameters")
    if (args.solver_timeout < 0 or args.solver_tolerance <= 0
            or args.config_timeout_seconds < 0
            or args.explicit_input_max_variables < 0
            or args.invocation_target_limit < 0):
        parser.error("invalid solver/config timeout/abstraction settings")
    if args.alpha < 0 or args.beta < 0 or not 0 <= args.eta <= 1:
        parser.error("alpha/beta must be non-negative and eta in [0,1]")
    return args


if __name__ == "__main__":
    run(parse())
