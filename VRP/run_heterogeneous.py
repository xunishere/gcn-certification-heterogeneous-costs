"""Prepare auditable heterogeneous-cost graph inputs.

This dependency-light driver does not train a GNN.  It mirrors the real
experiment's corrected threat semantics and writes every budget component.
"""
import argparse
import csv
import json
import random
from pathlib import Path

try:
    from .experiment_config import ExperimentConfig, PAPER_MODES
    from .graph_semantics import (
        GRAPH_SEMANTICS_VERSION,
        canonical_graph_sha256,
        canonical_undirected_edges,
    )
    from .node_importance import attack_costs, compute_node_importance
    from .risk_budget import (
        COST_BUDGET_POLICY,
        allocate_fixed_mean_budgets,
        fixed_resource_budget,
        target_risk,
    )
except ImportError:
    from experiment_config import ExperimentConfig, PAPER_MODES
    from graph_semantics import (
        GRAPH_SEMANTICS_VERSION,
        canonical_graph_sha256,
        canonical_undirected_edges,
    )
    from node_importance import attack_costs, compute_node_importance
    from risk_budget import (
        COST_BUDGET_POLICY,
        allocate_fixed_mean_budgets,
        fixed_resource_budget,
        target_risk,
    )


def candidate_sources(num_nodes, edges):
    """One-hop candidate sources including self, matching a one-layer GCN."""
    sources = [{node} for node in range(num_nodes)]
    for left, right in canonical_undirected_edges(num_nodes, edges):
        sources[left].add(right)
    return [sorted(group) for group in sources]


def prepare(config, num_nodes, edges, confidence, return_allocation=False):
    edges = tuple(tuple(edge) for edge in edges)
    importance = compute_node_importance(num_nodes, edges)
    heterogeneous_cost = attack_costs(importance, config.alpha)
    source_sets = candidate_sources(num_nodes, edges)
    graph_sha256 = canonical_graph_sha256(num_nodes, edges)

    # Risk is computed for the complete evaluated target set before any Q_t is
    # allocated.  The same deterministic mapping is therefore reusable by the
    # risk-only and combined modes.
    risks = {
        target: target_risk(importance[target], confidence[target], config.eta)
        for target in range(num_nodes)
    }
    allocation = allocate_fixed_mean_budgets(risks, config.q0, config.beta)
    rows = []
    for target, sources in enumerate(source_sets):
        risk = risks[target]
        source_costs = [heterogeneous_cost[source] for source in sources]
        candidate_mean = sum(source_costs) / len(source_costs)
        budget = fixed_resource_budget(
            config.q0,
            use_risk=config.use_risk_budget,
            use_cost=config.use_heterogeneous_cost,
            allocated_q_t=allocation.budgets[target],
        )
        rows.append({
            "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
            "canonical_graph_sha256": graph_sha256,
            "target": target,
            "importance": importance[target],
            "confidence": confidence[target],
            "risk": risk,
            "uses_heterogeneous_cost": int(config.use_heterogeneous_cost),
            "target_cost": heterogeneous_cost[target],
            "candidate_source_count": len(sources),
            "candidate_sources": sources,
            "candidate_source_costs": source_costs,
            "Q0": budget["Q0"],
            "Q_t": budget["Q_t"],
            "Q_equiv": budget["Q_equiv"],
            "Q_cost": budget["Q_cost"],
            "q": config.q,
            "mean_cost": (candidate_mean
                          if config.use_heterogeneous_cost else 1.0),
            "cost_budget_policy": COST_BUDGET_POLICY,
            "global_budget": budget["global_budget"],
            "global_budget_unit": budget["global_budget_unit"],
            "risk_budget_mapping_checksum": allocation.checksum,
            "risk_budget_total": allocation.total,
            "risk_budget_mean": allocation.mean,
            "risk_budget_min": allocation.minimum,
            "risk_budget_max": allocation.maximum,
            # Backward-compatible, explicitly unit-labelled alias.
            "target_certification_budget": budget["global_budget"],
        })
    return (rows, allocation) if return_allocation else rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=PAPER_MODES,
                        default="heterogeneous-cost")
    parser.add_argument(
        "--graph-json", required=True,
        help='{"num_nodes":N,"edges":[[u,v],...],"confidence":[...]}'
    )
    parser.add_argument("--output-dir", default="results/heterogeneous")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--q0", type=int, default=14,
                        help="global attack budget (Cora-style default)")
    parser.add_argument("--q", type=int, default=3,
                        help="per-source feature-flip count (never monetary)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        args.mode, args.alpha, args.eta, args.beta, args.q0, args.q, args.seed
    )
    random.seed(config.seed)
    payload = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    num_nodes = int(payload["num_nodes"])
    confidence = payload["confidence"]
    if len(confidence) != num_nodes:
        raise ValueError("confidence length must equal num_nodes")
    rows, allocation = prepare(
        config, num_nodes, payload["edges"], confidence, return_allocation=True
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(
        json.dumps({
            "config": config.as_dict(),
            "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
            "canonical_graph_sha256": canonical_graph_sha256(
                num_nodes, payload["edges"]
            ),
            "mean_cost_candidate_set": (
                "target-specific one-hop normalized-adjacency support, including self; "
                "reported for audit only and never used to scale the global budget"
            ),
            "cost_budget_policy": COST_BUDGET_POLICY,
            "risk_budget_allocation": allocation.as_manifest(),
            "nodes": rows,
        }, indent=2),
        encoding="utf-8",
    )
    with (output / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["target"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "json": str(output / "run.json"),
        "csv": str(output / "nodes.csv"),
        "nodes": num_nodes,
    }))


if __name__ == "__main__":
    main()
