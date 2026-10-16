"""Resumable real-data training and exact one-layer GCN certification.

Correct threat semantics
------------------------
Every source node obeys the local feature-flip *count* constraint
``sum_j z[v,j] <= q``.  Legacy/risk-only modes use a global count budget.
Heterogeneous/combined modes use only the global monetary constraint
``sum_vj c[v] z[v,j] <= Q_cost``.  The monetary limit is fixed rather than
scaled by a target's candidate mean cost: heterogeneous-cost uses ``Q_cost=Q0``
and combined uses ``Q_cost=Q_t``.  Candidate mean cost remains audit metadata
only.  Risk budgets are allocated jointly over the complete evaluated target
set: their non-negative integer sum is exactly ``N*Q0`` and they are weakly
monotone in target risk.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, degree

from .dataset_registry import (
    cli_dataset_name,
    dataset_spec,
    load_dataset,
    q0_values_for_dataset,
    q_values_for_dataset,
)
from .experiment_config import MODES, PAPER_MODES
from .graph_semantics import (
    GRAPH_SEMANTICS_VERSION,
    canonical_graph_sha256,
    canonical_undirected_edge_index,
)
from .node_importance import attack_costs, compute_node_importance
from .risk_budget import (
    COST_BUDGET_POLICY,
    RISK_BUDGET_POLICY,
    allocate_fixed_mean_budgets,
    fixed_resource_budget,
    target_risk,
)
from .weighted_bounds import count_max_gain, weighted_max_gain

SCHEMA_VERSION = 6
MODEL_CACHE_VERSION = 4
CONFIG_SLUG_VERSION = "schema-v6-canonical-graph-auditable-costs-v1"

FIELDS = [
    "schema_version", "graph_semantics_version", "canonical_graph_sha256",
    "dataset", "target_scope", "seed", "mode",
    "budget_policy", "cost_budget_policy", "risk_budget_mapping_checksum",
    "risk_budget_target_count", "risk_budget_total", "risk_budget_mean",
    "risk_budget_min", "risk_budget_max",
    "Q0", "Q_t", "Q_equiv", "Q_cost", "q", "mean_cost",
    "candidate_mean_cost", "global_budget", "global_budget_unit",
    "target", "label", "prediction", "confidence", "importance",
    "uses_heterogeneous_cost", "target_cost", "candidate_source_count",
    "candidate_sources", "candidate_source_costs", "certified_margin",
    "certified", "correct", "certified_and_correct", "key_node",
    "risk_weight", "seconds", "sampled", "sample_size",
    "target_pool_size", "num_nodes", "num_features",
    "clean_target_accuracy", "clean_test_accuracy", "model_cache_key",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(specification):
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if specification == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(specification)


def normalized_adj(edge_index, num_nodes, device):
    canonical = canonical_undirected_edge_index(
        edge_index, num_nodes, device=device
    )
    edge_with_loops, _ = add_remaining_self_loops(
        canonical, num_nodes=num_nodes
    )
    row, col = edge_with_loops
    degrees = degree(col, num_nodes, dtype=torch.float32)
    values = (degrees[row].clamp(min=1).pow(-.5)
              * degrees[col].clamp(min=1).pow(-.5))
    adjacency = torch.sparse_coo_tensor(
        edge_with_loops, values, (num_nodes, num_nodes), device=device,
        check_invariants=False,
    ).coalesce()
    return adjacency


def adjacency_support(adjacency):
    """Stable target-row supports ``[(source, normalized_weight), ...]``."""
    indices = adjacency.indices().detach().cpu()
    values = adjacency.values().detach().cpu()
    supports = [[] for _ in range(adjacency.shape[0])]
    for target, source, value in zip(indices[0].tolist(), indices[1].tolist(),
                                     values.tolist()):
        if abs(value) > 0:
            supports[target].append((source, float(value)))
    for support in supports:
        support.sort(key=lambda item: item[0])
    return supports


def masks(data, seed):
    num_nodes = data.num_nodes
    if (hasattr(data, "train_mask") and hasattr(data, "val_mask")
            and hasattr(data, "test_mask") and data.train_mask is not None
            and bool(data.train_mask.any())):
        train_mask = data.train_mask.clone()
        val_mask = data.val_mask.clone()
        test_mask = data.test_mask.clone()
        if train_mask.ndim > 1:
            train_mask, val_mask, test_mask = (
                train_mask[:, 0], val_mask[:, 0], test_mask[:, 0]
            )
        return train_mask, val_mask, test_mask

    generator = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for class_id in data.y.unique().tolist():
        indices = (data.y == class_id).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        first = max(1, round(.6 * len(indices)))
        second = max(first + 1, round(.8 * len(indices)))
        train_mask[indices[:first]] = True
        val_mask[indices[first:second]] = True
        test_mask[indices[second:]] = True
    return train_mask, val_mask, test_mask


def _model_cache_key(dataset_name, seed, epochs, data, device):
    graph_sha256 = canonical_graph_sha256(
        data.num_nodes, data.edge_index.t().tolist()
    )
    description = {
        "version": MODEL_CACHE_VERSION,
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "canonical_graph_sha256": graph_sha256,
        "dataset": dataset_name,
        "seed": int(seed),
        "epochs": int(epochs),
        "nodes": int(data.num_nodes),
        "features": int(data.num_features),
        "classes": int(data.y.max()) + 1,
        "device_type": device.type,
    }
    digest = hashlib.sha256(
        json.dumps(description, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return digest, description


@dataclass
class ModelRun:
    x: torch.Tensor
    y: torch.Tensor
    adjacency: torch.Tensor
    supports: list
    weights: torch.Tensor
    logits: torch.Tensor
    probability: torch.Tensor
    prediction: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    clean_test_accuracy: float
    clean_validation_accuracy: float
    cache_key: str
    cache_path: str
    cache_hit: bool
    train_seconds: float
    canonical_graph_sha256: str


def train_or_load(data, dataset_name, seed, device, epochs, model_cache_dir):
    seed_all(seed)
    x = data.x.float().to(device)
    y = data.y.to(device)
    if not bool(torch.all((x == 0) | (x == 1))):
        raise ValueError("exact flip certificate requires binary node features")
    adjacency = normalized_adj(data.edge_index, data.num_nodes, device)
    aggregated_x = torch.sparse.mm(adjacency, x)
    train_mask, val_mask, test_mask = masks(data, seed)
    train_device = train_mask.to(device)
    val_device = val_mask.to(device)
    test_device = test_mask.to(device)

    cache_key, cache_description = _model_cache_key(
        dataset_name, seed, epochs, data, device
    )
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_cache_dir / f"{dataset_name}-seed{seed}-{cache_key}.pt"
    cache_hit = False
    original_train_seconds = 0.0
    weights = None
    if cache_path.exists():
        payload = torch.load(cache_path, map_location=device, weights_only=True)
        if payload.get("description") == cache_description:
            weights = payload["weights"].to(device)
            original_train_seconds = float(payload.get("train_seconds", 0.0))
            cache_hit = True

    if weights is None:
        start = time.perf_counter()
        parameter = torch.nn.Parameter(torch.empty(
            x.shape[1], int(y.max()) + 1, device=device
        ))
        torch.nn.init.xavier_uniform_(parameter)
        optimizer = torch.optim.Adam([parameter], lr=.05, weight_decay=5e-4)
        best_weights, best_validation = None, -1.0
        patience = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = aggregated_x @ parameter
            loss = F.cross_entropy(logits[train_device], y[train_device])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation = ((logits[val_device].argmax(1) == y[val_device])
                              .float().mean().item()) if bool(val_device.any()) else 0.0
            if validation > best_validation:
                best_validation = validation
                best_weights = parameter.detach().clone()
                patience = 0
            else:
                patience += 1
            if patience >= 100:
                break
        weights = best_weights
        original_train_seconds = time.perf_counter() - start
        temporary = cache_path.with_suffix(".tmp")
        torch.save({
            "description": cache_description,
            "weights": weights.detach().cpu(),
            "train_seconds": original_train_seconds,
            "created_at": utc_now(),
        }, temporary)
        temporary.replace(cache_path)

    logits = aggregated_x @ weights
    probability = logits.softmax(1)
    prediction = logits.argmax(1)
    test_accuracy = float(
        (prediction[test_device] == y[test_device]).float().mean().item()
    ) if bool(test_device.any()) else float("nan")
    validation_accuracy = float(
        (prediction[val_device] == y[val_device]).float().mean().item()
    ) if bool(val_device.any()) else float("nan")
    return ModelRun(
        x=x,
        y=y,
        adjacency=adjacency,
        supports=adjacency_support(adjacency),
        weights=weights,
        logits=logits,
        probability=probability,
        prediction=prediction,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        clean_test_accuracy=test_accuracy,
        clean_validation_accuracy=validation_accuracy,
        cache_key=cache_key,
        cache_path=str(cache_path),
        cache_hit=cache_hit,
        train_seconds=original_train_seconds,
        canonical_graph_sha256=cache_description["canonical_graph_sha256"],
    )


def stratified_targets(labels, max_count, seed, pool=None):
    labels = np.asarray(labels)
    pool_array = (np.arange(len(labels), dtype=int) if pool is None
                  else np.asarray(pool, dtype=int))
    if max_count <= 0 or len(pool_array) <= max_count:
        return sorted(pool_array.tolist())
    generator = np.random.default_rng(seed)
    selected = []
    pool_labels = labels[pool_array]
    for class_id in np.unique(pool_labels):
        class_indices = pool_array[pool_labels == class_id]
        count = max(1, round(max_count * len(class_indices) / len(pool_array)))
        selected.extend(generator.choice(
            class_indices, min(count, len(class_indices)), replace=False
        ).tolist())
    if len(selected) > max_count:
        selected = generator.choice(selected, max_count, replace=False).tolist()
    elif len(selected) < max_count:
        selected_set = set(selected)
        remaining = [index for index in pool_array.tolist()
                     if index not in selected_set]
        selected.extend(generator.choice(
            remaining, max_count - len(selected), replace=False
        ).tolist())
    return sorted(int(index) for index in selected)


def select_targets(model_run, dataset_name, target_scope, sample_size, seed):
    labels = model_run.y.detach().cpu().numpy()
    specification = dataset_spec(dataset_name)
    if target_scope == "test":
        pool = model_run.test_mask.nonzero().flatten().tolist()
        label = ("planetoid_test_mask" if specification.family == "Planetoid"
                 else "test_mask")
        return sorted(pool), len(pool), label
    if target_scope == "all":
        targets = list(range(len(labels)))
        return targets, len(targets), "all_nodes"
    # Development targets are sampled reproducibly from all nodes.  The generic
    # sample-size option retains the previous Cora selection rule exactly.
    targets = stratified_targets(labels, sample_size, seed)
    return targets, len(labels), "all_nodes"


@dataclass
class TargetProblem:
    target: int
    sources: list
    source_costs: list
    candidate_mean_cost: float
    nominal_margins: list
    gains_by_rival: list
    risk: float
    confidence: float
    prediction: int


@dataclass
class CertificateEngine:
    model: ModelRun
    importance: list
    heterogeneous_costs: list
    max_q: int
    eta: float
    problems: dict = field(default_factory=dict)

    def problem(self, target):
        if target in self.problems:
            return self.problems[target]
        support = self.model.supports[target]
        if not support:
            raise RuntimeError(f"target {target} has no normalized-adjacency support")
        sources = [source for source, _ in support]
        adjacency_weights = torch.tensor(
            [weight for _, weight in support],
            dtype=self.model.x.dtype,
            device=self.model.x.device,
        )
        source_costs = [self.heterogeneous_costs[source] for source in sources]
        candidate_mean = float(sum(source_costs) / len(source_costs))
        predicted = int(self.model.prediction[target])
        confidence = float(self.model.probability[target, predicted])
        risk = target_risk(self.importance[target], confidence, self.eta)
        source_features = self.model.x[sources]
        flip_sign = 1.0 - 2.0 * source_features
        nominal_margins, gains_by_rival = [], []
        for rival in range(self.model.weights.shape[1]):
            if rival == predicted:
                continue
            difference = self.model.weights[:, predicted] - self.model.weights[:, rival]
            nominal_margins.append(float(
                self.model.logits[target, predicted] - self.model.logits[target, rival]
            ))
            delta = adjacency_weights[:, None] * flip_sign * difference[None, :]
            adverse = torch.clamp(-delta, min=0.0)
            top_count = min(self.max_q, adverse.shape[1])
            top = torch.topk(adverse, k=top_count, dim=1).values
            groups = []
            for row in top.detach().cpu().tolist():
                groups.append([float(value) for value in row if value > 0.0])
            gains_by_rival.append(groups)
        problem = TargetProblem(
            target=target,
            sources=sources,
            source_costs=source_costs,
            candidate_mean_cost=candidate_mean,
            nominal_margins=nominal_margins,
            gains_by_rival=gains_by_rival,
            risk=risk,
            confidence=confidence,
            prediction=predicted,
        )
        self.problems[target] = problem
        return problem

    def certify(self, target, mode, q, q0, allocated_q_t, deadline=None):
        problem = self.problem(target)
        use_risk = mode in ("risk-budget", "combined")
        use_cost = mode in ("heterogeneous-cost", "combined")
        budget = fixed_resource_budget(
            q0,
            use_risk=use_risk,
            use_cost=use_cost,
            allocated_q_t=allocated_q_t,
        )
        start = time.perf_counter()
        worst_margin = float("inf")
        for nominal, groups in zip(problem.nominal_margins,
                                   problem.gains_by_rival):
            if use_cost:
                loss = weighted_max_gain(
                    groups, problem.source_costs, budget["Q_cost"], q, deadline
                )
            else:
                loss = count_max_gain(groups, budget["Q_equiv"], q)
            worst_margin = min(worst_margin, nominal - loss)
        elapsed = time.perf_counter() - start
        return worst_margin, worst_margin > 0.0, budget, elapsed, problem


def _config_slug(dataset, seed, mode, q, q0):
    return (Path("configs") / CONFIG_SLUG_VERSION / dataset / f"seed-{seed}"
            / mode / f"q-{q}" / f"Q0-{q0}")


def _read_done_targets(csv_path, expected_mapping_checksum):
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError(f"incompatible result schema on resume: {csv_path}")
            if row.get("risk_budget_mapping_checksum") != expected_mapping_checksum:
                raise RuntimeError(f"risk-budget mapping mismatch on resume: {csv_path}")
            target = int(row["target"])
            if target in done:
                raise RuntimeError(f"duplicate target on resume: {csv_path}: {target}")
            done.add(target)
    return done


def _write_target_manifest(path, payload):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"target manifest mismatch on resume: {path}")
    else:
        atomic_json(path, payload)


def run_one_config(args, output, dataset_name, seed, mode, q, q0, targets,
                   scope_label, target_pool_size, data, model, engine,
                   key_threshold, allocation, allocation_manifest_path):
    slug = _config_slug(dataset_name, seed, mode, q, q0)
    csv_path = output / slug.with_suffix(".csv")
    status_path = output / slug.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_targets = _read_done_targets(csv_path, allocation.checksum)
    target_set = set(targets)
    unexpected = done_targets - target_set
    if unexpected:
        raise RuntimeError(f"config contains unexpected targets: {csv_path}")
    if done_targets == target_set:
        existing_status = (json.loads(status_path.read_text(encoding="utf-8"))
                           if status_path.exists() else {})
        if existing_status:
            if int(existing_status.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError(f"incompatible status schema: {status_path}")
            if existing_status.get("risk_budget_mapping_checksum") != allocation.checksum:
                raise RuntimeError(f"status mapping mismatch: {status_path}")
        if existing_status.get("status") != "done":
            atomic_json(status_path, {
                "schema_version": SCHEMA_VERSION,
                "status": "done", "updated_at": utc_now(),
                "completed_targets": len(done_targets),
                "expected_targets": len(targets), "csv": str(csv_path),
                "budget_policy": RISK_BUDGET_POLICY,
                "risk_budget_mapping_checksum": allocation.checksum,
                "risk_budget_mapping": str(allocation_manifest_path),
            })
        return "done", 0.0

    attempt_started = time.perf_counter()
    deadline = (time.monotonic() + args.config_timeout_seconds
                if args.config_timeout_seconds > 0 else None)
    status = {
        "schema_version": SCHEMA_VERSION,
        "config_slug_version": CONFIG_SLUG_VERSION,
        "dataset": dataset_name,
        "seed": seed,
        "mode": mode,
        "q": q,
        "Q0": q0,
        "budget_policy": RISK_BUDGET_POLICY,
        "risk_budget_mapping_checksum": allocation.checksum,
        "risk_budget_mapping": str(allocation_manifest_path),
        "risk_budget_total": allocation.total,
        "risk_budget_mean": allocation.mean,
        "risk_budget_min": allocation.minimum,
        "risk_budget_max": allocation.maximum,
        "risk_budget_distribution": allocation.distribution,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_targets": len(done_targets),
        "expected_targets": len(targets),
        "csv": str(csv_path),
        "timeout_seconds_per_attempt": args.config_timeout_seconds,
        "error": None,
    }
    atomic_json(status_path, status)
    append = csv_path.exists() and csv_path.stat().st_size > 0
    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if not append:
        writer.writeheader()
        handle.flush()

    target_accuracy = float(np.mean([
        int(model.prediction[target]) == int(model.y[target]) for target in targets
    ]))
    final_status = "done"
    try:
        for target in targets:
            if target in done_targets:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("configuration wall-clock deadline exceeded")
            margin, certified, budget, seconds, problem = engine.certify(
                target, mode, q, q0, allocation.budgets[target], deadline
            )
            use_cost = mode in ("heterogeneous-cost", "combined")
            correct = int(problem.prediction == int(model.y[target]))
            row = {
                "schema_version": SCHEMA_VERSION,
                "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                "canonical_graph_sha256": model.canonical_graph_sha256,
                "dataset": dataset_name,
                "target_scope": scope_label,
                "seed": seed,
                "mode": mode,
                "budget_policy": RISK_BUDGET_POLICY,
                "cost_budget_policy": COST_BUDGET_POLICY,
                "risk_budget_mapping_checksum": allocation.checksum,
                "risk_budget_target_count": allocation.target_count,
                "risk_budget_total": allocation.total,
                "risk_budget_mean": allocation.mean,
                "risk_budget_min": allocation.minimum,
                "risk_budget_max": allocation.maximum,
                "Q0": budget["Q0"],
                "Q_t": budget["Q_t"],
                "Q_equiv": budget["Q_equiv"],
                "Q_cost": budget["Q_cost"],
                "q": q,
                "mean_cost": (problem.candidate_mean_cost if use_cost else 1.0),
                "candidate_mean_cost": problem.candidate_mean_cost,
                "global_budget": budget["global_budget"],
                "global_budget_unit": budget["global_budget_unit"],
                "target": target,
                "label": int(model.y[target]),
                "prediction": problem.prediction,
                "confidence": problem.confidence,
                "importance": engine.importance[target],
                "uses_heterogeneous_cost": int(use_cost),
                "target_cost": engine.heterogeneous_costs[target],
                "candidate_source_count": len(problem.sources),
                "candidate_sources": json.dumps(
                    problem.sources, separators=(",", ":")
                ),
                "candidate_source_costs": json.dumps(
                    problem.source_costs, separators=(",", ":")
                ),
                "certified_margin": margin,
                "certified": int(certified),
                "correct": correct,
                "certified_and_correct": int(certified) * correct,
                "key_node": int(engine.importance[target] >= key_threshold),
                "risk_weight": problem.risk,
                "seconds": seconds,
                "sampled": int(len(targets) < target_pool_size),
                "sample_size": len(targets),
                "target_pool_size": target_pool_size,
                "num_nodes": data.num_nodes,
                "num_features": data.num_features,
                "clean_target_accuracy": target_accuracy,
                "clean_test_accuracy": model.clean_test_accuracy,
                "model_cache_key": model.cache_key,
            }
            writer.writerow(row)
            handle.flush()
            done_targets.add(target)
            status["completed_targets"] = len(done_targets)
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
    except TimeoutError as error:
        final_status = "timed_out"
        status["error"] = str(error)
    except Exception as error:
        final_status = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        handle.close()
        elapsed = time.perf_counter() - attempt_started
        status["status"] = final_status
        status["attempt_seconds"] = elapsed
        status["completed_targets"] = len(done_targets)
        status["updated_at"] = utc_now()
        atomic_json(status_path, status)
    return final_status, elapsed


def _consolidate(output, expected_configs):
    destination = output / "nodes.csv"
    temporary = destination.with_suffix(".csv.tmp")
    rows = 0
    with temporary.open("w", newline="", encoding="utf-8") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=FIELDS)
        writer.writeheader()
        for csv_path in expected_configs:
            if not csv_path.exists():
                continue
            with csv_path.open(encoding="utf-8", newline="") as source_handle:
                for row in csv.DictReader(source_handle):
                    writer.writerow({field: row.get(field, "") for field in FIELDS})
                    rows += 1
    temporary.replace(destination)
    return rows


def _sample_size_for_dataset(args, dataset_name):
    return (int(args.sample_size) if args.sample_size is not None
            else dataset_spec(dataset_name).default_sample_size)


def _q_values_for_run(args, dataset_name):
    return q_values_for_dataset(
        dataset_name, explicit_q=args.q, sensitivity=args.q_sensitivity
    )


def _q0_values_for_run(args, dataset_name):
    return q0_values_for_dataset(
        dataset_name, explicit_q0=args.q0, sensitivity=args.q0_sensitivity
    )


def _run_configuration(args, device):
    return {
        "schema_version": SCHEMA_VERSION,
        "config_slug_version": CONFIG_SLUG_VERSION,
        "model_cache_version": MODEL_CACHE_VERSION,
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "canonical_graph_rule": (
            "remove input self-loops, collapse duplicates, symmetrize every "
            "edge, then add exactly one GCN self-loop per node"
        ),
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "modes": list(args.modes),
        "q_values_by_dataset": {
            name: list(_q_values_for_run(args, name)) for name in args.datasets
        },
        "Q0_values_by_dataset": {
            name: list(_q0_values_for_run(args, name)) for name in args.datasets
        },
        "target_scope": args.target_scope,
        "sample_size_by_dataset": {
            name: _sample_size_for_dataset(args, name) for name in args.datasets
        },
        "legacy_cora_sample_argument": args.cora_sample,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "eta": args.eta,
        "beta": args.beta,
        "importance_weights": list(args.importance_weights),
        "device": str(device),
        "risk_budget_policy": RISK_BUDGET_POLICY,
        "cost_budget_policy": COST_BUDGET_POLICY,
        "risk_budget_constraint": (
            "for the complete evaluated (dataset,seed,target_scope) target set, "
            "Q_t are non-negative integers, sum_t Q_t=N*Q0, and are weakly "
            "monotone in r_t; risk-budget and combined share one mapping"
        ),
        "risk_continuous_score": "Q0 + beta * (r_t - mean_t r_t)",
        "risk_projection_integerization": (
            "non-negative simplex projection followed by deterministic largest "
            "remainder; increasing target id breaks fractional ties"
        ),
        "mean_cost_candidate_set": (
            "target-specific nonzero normalized-adjacency row support including self; "
            "reported for audit only and never used to scale the global budget"
        ),
        "local_constraint": "sum_j z[v,j] <= q (count in every mode)",
        "legacy_global_constraint": "sum_vj z[v,j] <= Q_equiv",
        "heterogeneous_global_constraint": "sum_vj c[v] z[v,j] <= Q_cost",
        "fixed_cost_budget": (
            "heterogeneous-cost: Q_cost=Q0; combined: Q_cost=Q_t; "
            "candidate mean cost does not change either budget"
        ),
    }


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = (Path(args.model_cache_dir) if args.model_cache_dir
                       else output / "model-cache")
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    run_config = _run_configuration(args, device)
    run_config_path = output / "run_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError(
                f"refusing incompatible resume in {output}; use a new output directory"
            )
    else:
        atomic_json(run_config_path, run_config)

    metadata_path = output / "metadata.json"
    metadata = ({"schema_version": SCHEMA_VERSION, "created_at": utc_now(), "runs": []}
                if not metadata_path.exists()
                else json.loads(metadata_path.read_text(encoding="utf-8")))
    existing_runs = {(run["dataset"], int(run["seed"])) for run in metadata["runs"]}
    expected_csv_paths = []
    budget_mapping_paths = []
    statuses = {"done": 0, "timed_out": 0, "failed": 0}
    wall_start = time.perf_counter()

    for dataset_name in args.datasets:
        q_values = _q_values_for_run(args, dataset_name)
        q0_values = _q0_values_for_run(args, dataset_name)
        dataset = load_dataset(dataset_name, cache_dir)
        data = dataset[0]
        edges = data.edge_index.t().tolist()
        importance = compute_node_importance(
            data.num_nodes, edges, args.importance_weights
        )
        heterogeneous_costs = attack_costs(importance, args.alpha)
        key_threshold = float(np.quantile(importance, .9))
        for seed in args.seeds:
            model = train_or_load(
                data, dataset_name, seed, device, args.epochs, model_cache_dir
            )
            targets, target_pool_size, target_pool = select_targets(
                model, dataset_name, args.target_scope,
                _sample_size_for_dataset(args, dataset_name), seed,
            )
            if not targets:
                raise RuntimeError(f"empty evaluated target set: {dataset_name}, seed={seed}")
            scope_label = (
                f"stratified-{len(targets)}-from-{target_pool}"
                if args.target_scope == "stratified" else target_pool
            )
            target_manifest = {
                "schema_version": SCHEMA_VERSION,
                "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                "canonical_graph_sha256": model.canonical_graph_sha256,
                "dataset": dataset_name,
                "seed": seed,
                "target_scope": scope_label,
                "target_pool_size": target_pool_size,
                "target_count": len(targets),
                "targets": targets,
            }
            _write_target_manifest(
                output / "targets" / dataset_name / f"seed-{seed}.json",
                target_manifest,
            )
            engine = CertificateEngine(
                model=model,
                importance=importance,
                heterogeneous_costs=heterogeneous_costs,
                max_q=max(q_values),
                eta=args.eta,
            )

            # Materialize every evaluated target problem first.  Consequently
            # every r_t is known before any joint budget mapping is allocated.
            target_problems = [engine.problem(target) for target in targets]
            target_risks = {
                problem.target: problem.risk for problem in target_problems
            }
            allocations = {}
            allocation_paths = {}
            for q0 in q0_values:
                allocation = allocate_fixed_mean_budgets(
                    target_risks, q0, args.beta
                )
                if allocation.total != len(targets) * q0:
                    raise RuntimeError("risk-budget conservation check failed")
                allocation_path = (
                    output / "budget-mappings" / dataset_name / f"seed-{seed}"
                    / f"Q0-{q0}.json"
                )
                allocation_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                    "canonical_graph_sha256": model.canonical_graph_sha256,
                    "dataset": dataset_name,
                    "seed": seed,
                    "target_scope": scope_label,
                    "target_pool_size": target_pool_size,
                    "eta": args.eta,
                } | allocation.as_manifest()
                _write_target_manifest(allocation_path, allocation_manifest)
                allocations[q0] = allocation
                allocation_paths[q0] = allocation_path
                budget_mapping_paths.append(allocation_path)

            if (dataset_name, seed) not in existing_runs:
                metadata["runs"].append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "nodes": data.num_nodes,
                    "features": data.num_features,
                    "classes": int(data.y.max()) + 1,
                    "targets": len(targets),
                    "target_scope": scope_label,
                    "target_pool_size": target_pool_size,
                    "q_values": list(q_values),
                    "train_mask_nodes": int(model.train_mask.sum()),
                    "validation_mask_nodes": int(model.val_mask.sum()),
                    "test_mask_nodes": int(model.test_mask.sum()),
                    "clean_validation_accuracy": model.clean_validation_accuracy,
                    "clean_test_accuracy": model.clean_test_accuracy,
                    "model_train_seconds": model.train_seconds,
                    "model_cache_hit": model.cache_hit,
                    "model_cache_key": model.cache_key,
                    "model_cache_path": model.cache_path,
                    "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
                    "canonical_graph_sha256": model.canonical_graph_sha256,
                    "key_threshold_global_top10_percent": key_threshold,
                    "risk_budget_policy": RISK_BUDGET_POLICY,
                    "cost_budget_policy": COST_BUDGET_POLICY,
                    "risk_budget_mappings": {
                        str(q0): {
                            "path": str(allocation_paths[q0]),
                            "checksum": allocations[q0].checksum,
                            "total": allocations[q0].total,
                            "mean": allocations[q0].mean,
                            "minimum": allocations[q0].minimum,
                            "maximum": allocations[q0].maximum,
                            "distribution": allocations[q0].distribution,
                        }
                        for q0 in q0_values
                    },
                })
                existing_runs.add((dataset_name, seed))
                atomic_json(metadata_path, metadata)

            for mode in args.modes:
                for q in q_values:
                    for q0 in q0_values:
                        csv_path = output / _config_slug(
                            dataset_name, seed, mode, q, q0
                        ).with_suffix(".csv")
                        expected_csv_paths.append(csv_path)
                        status, _ = run_one_config(
                            args, output, dataset_name, seed, mode, q, q0,
                            targets, scope_label, target_pool_size, data, model,
                            engine, key_threshold, allocations[q0],
                            allocation_paths[q0],
                        )
                        statuses[status] = statuses.get(status, 0) + 1

    rows = _consolidate(output, expected_csv_paths)
    summary_path = output / "run_summary.json"
    previous_summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                        if summary_path.exists() else None)
    invocation_wall = time.perf_counter() - wall_start
    invocation_record = {
        "updated_at": utc_now(),
        "wall_seconds": invocation_wall,
        "configuration_status_counts": statuses,
        "rows": rows,
    }
    invocation_history = []
    if previous_summary is not None:
        invocation_history = list(previous_summary.get("invocations", []))
        if not invocation_history:
            invocation_history.append({
                "updated_at": previous_summary.get("updated_at"),
                "wall_seconds": previous_summary.get("wall_seconds_this_invocation"),
                "configuration_status_counts": previous_summary.get(
                    "configuration_status_counts"
                ),
                "rows": previous_summary.get("rows"),
                "migrated_from_pre_history_summary": True,
            })
    invocation_history.append(invocation_record)
    completed_now = (statuses.get("timed_out", 0) == 0
                     and statuses.get("failed", 0) == 0)
    if previous_summary is not None and previous_summary.get("status") == "done":
        first_completed_wall = previous_summary.get(
            "first_completed_wall_seconds",
            previous_summary.get("wall_seconds_this_invocation"),
        )
    else:
        first_completed_wall = invocation_wall if completed_now else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "done" if completed_now else "partial",
        "updated_at": invocation_record["updated_at"],
        "configuration_status_counts": statuses,
        "rows": rows,
        "expected_configurations": len(expected_csv_paths),
        "risk_budget_policy": RISK_BUDGET_POLICY,
        "cost_budget_policy": COST_BUDGET_POLICY,
        "risk_budget_mapping_count": len(budget_mapping_paths),
        "budget_conservation_verified": True,
        "wall_seconds_this_invocation": invocation_wall,
        "first_completed_wall_seconds": first_completed_wall,
        "invocations": invocation_history,
        "nodes_csv": str(output / "nodes.csv"),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", type=cli_dataset_name, default=["Cora"],
        help=("registered PyG datasets: Cora, CiteSeer (alias Citeseer), "
              "and Karate")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--modes", nargs="+", choices=PAPER_MODES,
                        default=list(PAPER_MODES))
    parser.add_argument(
        "--q0", nargs="+", type=int, default=None,
        help=("explicit global-budget values for every selected dataset; when "
              "omitted use each registry main Q (Cora 14, CiteSeer 37, Karate 1)")
    )
    parser.add_argument(
        "--q0-sensitivity", action="store_true",
        help=("use registry global-Q sensitivity grids when --q0 is omitted: "
              "Cora {3,7,14,28}, CiteSeer {9,18,37,74}")
    )
    parser.add_argument(
        "--q", nargs="+", type=int, default=None,
        help=("explicit local count values for every selected dataset; when omitted "
              "use floor(0.01*D): Cora 14, CiteSeer 37, Karate 1")
    )
    parser.add_argument(
        "--q-sensitivity", action="store_true",
        help=("use registry local-q values when --q is omitted; the paper "
              "protocol fixes q=floor(0.01*D): Cora 14, CiteSeer 37")
    )
    parser.add_argument("--target-scope", choices=("stratified", "test", "all"),
                        default="stratified")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="generic stratified development sample size")
    parser.add_argument(
        "--cora-sample", type=int, default=None,
        help="deprecated alias for --sample-size; retained for old scripts"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache-dir", default=".cache/pyg")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--config-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--importance-weights", nargs=3, type=float,
                        default=[.4, .4, .2])
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets must be unique after alias canonicalization")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.q is not None and args.q_sensitivity:
        parser.error("choose either explicit --q values or --q-sensitivity")
    if args.q0 is not None and args.q0_sensitivity:
        parser.error("choose either explicit --q0 values or --q0-sensitivity")
    if ((args.q is not None and any(q < 0 for q in args.q))
            or (args.q0 is not None and any(q0 < 0 for q0 in args.q0))):
        parser.error("q and Q0 must be non-negative")
    if ((args.q is not None and len(set(args.q)) != len(args.q))
            or (args.q0 is not None and len(set(args.q0)) != len(args.q0))):
        parser.error("q and Q0 values must be unique")
    if args.sample_size is not None and args.cora_sample is not None:
        if args.sample_size != args.cora_sample:
            parser.error("--sample-size and --cora-sample disagree")
    if args.sample_size is None and args.cora_sample is not None:
        args.sample_size = args.cora_sample
    if args.sample_size is not None and args.sample_size < 0:
        parser.error("sample size must be non-negative")
    if args.epochs <= 0 or args.config_timeout_seconds < 0:
        parser.error("epochs must be positive and timeout non-negative")
    if args.alpha < 0 or args.beta < 0 or not 0 <= args.eta <= 1:
        parser.error("alpha/beta must be non-negative and eta in [0,1]")
    return args


if __name__ == "__main__":
    run(parse())
