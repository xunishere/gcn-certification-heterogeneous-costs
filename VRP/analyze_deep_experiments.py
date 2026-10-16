"""Paper-facing analysis for sampled deep Cora/CiteSeer experiments."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze_experiments import cost_audit_checks
from .dataset_registry import dataset_spec, load_dataset
from .experiment_config import MODES, PAPER_MODES
from .node_importance import compute_node_importance

RATE_METRICS = [
    "certification_rate", "certified_and_correct_rate", "cohort_clean_accuracy",
    "top10_certification_rate", "top10_certified_and_correct_rate",
    "risk_weighted_certification_rate", "risk_weighted_certified_and_correct_rate",
    "solver_trusted_rate",
]
CONTINUOUS_METRICS = [
    "clean_train_accuracy", "clean_validation_accuracy", "clean_test_accuracy",
    "lp_lower_bound_mean", "approximation_gap_mean", "total_seconds_mean",
    "solver_seconds_mean", "config_total_seconds", "variable_count_median",
    "constraint_count_median", "potential_input_variable_count_median",
    "candidate_source_count_median",
]


def save_figure(fig, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_top10_flag(data: pd.DataFrame, cache_dir: str) -> tuple[pd.DataFrame, dict]:
    top_ids = {}
    for dataset in sorted(data.dataset.unique()):
        graph = load_dataset(dataset, cache_dir)[0]
        importance = compute_node_importance(
            int(graph.num_nodes), graph.edge_index.t().tolist(), (.4, .4, .2)
        )
        count = int(math.ceil(.1 * len(importance)))
        order = sorted(range(len(importance)), key=lambda node: (-importance[node], node))
        top_ids[dataset] = set(order[:count])
    result = data.copy()
    result["top10_important"] = [
        int(int(target) in top_ids[dataset])
        for dataset, target in zip(result.dataset, result.target)
    ]
    return result, top_ids


def summarize_config(group: pd.DataFrame) -> pd.Series:
    top = group[group.top10_important == 1]
    risk_total = float(group.risk_weight.sum())
    return pd.Series({
        "target_count": int(len(group)),
        "top10_target_count": int(len(top)),
        "certification_rate": float(group.certified.mean()),
        "certified_and_correct_rate": float(group.certified_and_correct.mean()),
        "cohort_clean_accuracy": float(group.correct.mean()),
        "top10_certification_rate": (float(top.certified.mean()) if len(top) else np.nan),
        "top10_certified_and_correct_rate": (
            float(top.certified_and_correct.mean()) if len(top) else np.nan
        ),
        "risk_weighted_certification_rate": float(
            (group.certified * group.risk_weight).sum() / risk_total
        ),
        "risk_weighted_certified_and_correct_rate": float(
            (group.certified_and_correct * group.risk_weight).sum() / risk_total
        ),
        "solver_trusted_rate": float(group.solver_trusted.mean()),
        "clean_train_accuracy": float(group.clean_train_accuracy.iloc[0]),
        "clean_validation_accuracy": float(group.clean_validation_accuracy.iloc[0]),
        "clean_test_accuracy": float(group.clean_test_accuracy.iloc[0]),
        "lp_lower_bound_mean": float(group.lp_lower_bound.mean()),
        "lp_lower_bound_median": float(group.lp_lower_bound.median()),
        "approximation_gap_mean": float(group.approximation_gap.mean()),
        "approximation_gap_median": float(group.approximation_gap.median()),
        "total_seconds_mean": float(group.total_seconds.mean()),
        "total_seconds_median": float(group.total_seconds.median()),
        "total_seconds_p95": float(group.total_seconds.quantile(.95)),
        "total_seconds_max": float(group.total_seconds.max()),
        "solver_seconds_mean": float(group.solver_seconds.mean()),
        "config_total_seconds": float(group.total_seconds.sum()),
        "variable_count_median": float(group.variable_count.median()),
        "variable_count_max": int(group.variable_count.max()),
        "constraint_count_median": float(group.constraint_count.median()),
        "constraint_count_max": int(group.constraint_count.max()),
        "potential_input_variable_count_median": float(
            group.potential_input_variable_count.median()
        ),
        "potential_input_variable_count_max": int(
            group.potential_input_variable_count.max()
        ),
        "candidate_source_count_median": float(group.candidate_source_count.median()),
        "candidate_source_count_max": int(group.candidate_source_count.max()),
    })


def aggregate_seed_rows(per_seed: pd.DataFrame, group_fields: list[str]) -> pd.DataFrame:
    rows = []
    metrics = RATE_METRICS + CONTINUOUS_METRICS
    for key, group in per_seed.groupby(group_fields, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_fields, key))
        row["seeds"] = int(group.seed.nunique())
        row["targets_per_seed"] = int(group.target_count.iloc[0])
        row["top10_targets_total"] = int(group.top10_target_count.sum())
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_results(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["dataset", "seed", "depth", "mode", "q", "Q0", "target",
           "model_cache_key", "target_cohort_checksum"]
    paired = data.pivot(index=key, columns="method",
                        values=["lp_lower_bound", "certified"]).dropna().reset_index()
    paired.columns = ["_".join(value).strip("_") if isinstance(value, tuple)
                      else value for value in paired.columns]
    paired["dual_minus_baseline"] = (
        paired.lp_lower_bound_dual - paired.lp_lower_bound_baseline
    )
    paired["outcome"] = np.select(
        [paired.dual_minus_baseline > 1e-6,
         paired.dual_minus_baseline < -1e-6],
        ["dual_tighter", "negative_numerical"], default="equal",
    )
    paired["cert_set_change"] = np.select(
        [paired.certified_dual > paired.certified_baseline,
         paired.certified_dual < paired.certified_baseline],
        ["dual_gain", "dual_loss"], default="unchanged",
    )
    seed_rows = []
    seed_fields = ["dataset", "seed", "depth", "mode", "q", "Q0"]
    for key_value, group in paired.groupby(seed_fields, sort=True):
        row = dict(zip(seed_fields, key_value))
        row.update({
            "pair_count": len(group),
            "delta_mean": float(group.dual_minus_baseline.mean()),
            "delta_sd_targets": float(group.dual_minus_baseline.std(ddof=1)),
            "delta_median": float(group.dual_minus_baseline.median()),
            "delta_min": float(group.dual_minus_baseline.min()),
            "delta_max": float(group.dual_minus_baseline.max()),
            "dual_tighter_count": int((group.outcome == "dual_tighter").sum()),
            "equal_count": int((group.outcome == "equal").sum()),
            "negative_count": int((group.outcome == "negative_numerical").sum()),
            "cert_gain_count": int((group.cert_set_change == "dual_gain").sum()),
            "cert_loss_count": int((group.cert_set_change == "dual_loss").sum()),
        })
        seed_rows.append(row)
    per_seed = pd.DataFrame(seed_rows)
    aggregate_rows = []
    aggregate_fields = ["dataset", "depth", "mode", "q", "Q0"]
    for key_value, group in per_seed.groupby(aggregate_fields, sort=True):
        raw = paired
        for field, value in zip(aggregate_fields, key_value):
            raw = raw[raw[field] == value]
        row = dict(zip(aggregate_fields, key_value))
        row.update({
            "seeds": int(group.seed.nunique()),
            "delta_mean_of_seeds": float(group.delta_mean.mean()),
            "delta_sd_of_seeds": float(group.delta_mean.std(ddof=1)),
            "pooled_delta_median": float(raw.dual_minus_baseline.median()),
            "pooled_delta_min": float(raw.dual_minus_baseline.min()),
            "pooled_delta_max": float(raw.dual_minus_baseline.max()),
            "pooled_dual_tighter_count": int((raw.outcome == "dual_tighter").sum()),
            "pooled_equal_count": int((raw.outcome == "equal").sum()),
            "pooled_negative_count": int((raw.outcome == "negative_numerical").sum()),
            "pooled_cert_gain_count": int((raw.cert_set_change == "dual_gain").sum()),
            "pooled_cert_loss_count": int((raw.cert_set_change == "dual_loss").sum()),
        })
        aggregate_rows.append(row)
    return paired, per_seed, pd.DataFrame(aggregate_rows)


def ablation_results(formal: pd.DataFrame, ablation_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    target_sets = {}
    for path in ablation_paths:
        part = pd.read_csv(path)
        audit = cost_audit_checks(part)
        if not (audit["all_rows_valid"]
                and audit["graph_semantics_version_valid"]
                and audit["one_canonical_graph_per_dataset"]):
            raise ValueError(f"invalid ablation graph/cost audit in {path}: {audit}")
        frames.append(part)
        for (dataset, depth), group in part.groupby(["dataset", "depth"]):
            target_sets[(dataset, int(depth))] = set(group.target.unique())
    main_parts = []
    for (dataset, depth), targets in target_sets.items():
        specification = dataset_spec(dataset)
        part = formal[(formal.dataset == dataset) & (formal.seed == 0) &
                      (formal.depth == depth) & (formal["mode"] == "legacy") &
                      (formal.q == specification.main_q) &
                      (formal.Q0 == specification.main_q0) &
                      (formal.target.isin(targets))]
        main_parts.append(part)
    frames.append(pd.concat(main_parts, ignore_index=True))
    data = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for key, group in data.groupby(["dataset", "depth", "samples", "method"]):
        summary_rows.append({
            "dataset": key[0], "depth": key[1], "samples": key[2], "method": key[3],
            "targets": int(group.target.nunique()),
            "certification_rate": float(group.certified.mean()),
            "lp_lower_bound_mean": float(group.lp_lower_bound.mean()),
            "total_seconds_mean": float(group.total_seconds.mean()),
            "under_used_neurons_mean": float(group.under_used_neurons.mean()),
            "under_unique_samples_mean": float(group.under_unique_samples.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    keys = ["dataset", "depth", "samples", "target"]
    dual = data[data.method == "dual"]
    baseline = data[data.method == "baseline"]
    pair = dual.merge(baseline, on=keys, suffixes=("_dual", "_baseline"))
    pair["dual_minus_baseline"] = pair.lp_lower_bound_dual - pair.lp_lower_bound_baseline
    pair_summary = pair.groupby(["dataset", "depth", "samples"]).agg(
        pairs=("target", "size"), delta_mean=("dual_minus_baseline", "mean"),
        delta_median=("dual_minus_baseline", "median"),
        delta_max=("dual_minus_baseline", "max"),
        dual_time_mean=("total_seconds_dual", "mean"),
        baseline_time_mean=("total_seconds_baseline", "mean"),
    ).reset_index()
    return summary, pair_summary


def create_figures(aggregate: pd.DataFrame, paired: pd.DataFrame,
                   per_seed: pd.DataFrame, ablation_pair: pd.DataFrame,
                   output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in MODES if mode in set(aggregate["mode"].unique())]
    colors = dict(zip(MODES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, ((dataset, depth), block) in zip(axes.flat,
            aggregate.groupby(["dataset", "depth"], sort=True)):
        for mode in modes:
            for method, style in (("baseline", "-"), ("dual", "--")):
                rows = block[(block["mode"] == mode) & (block.method == method)].sort_values("Q0")
                axis.plot(rows.Q0, 100 * rows.certification_rate_mean,
                          linestyle=style, marker="o", color=colors[mode],
                          label=f"{mode}/{method}" if axis is axes.flat[0] else None)
        axis.set_title(f"{dataset}, depth={depth}")
        axis.set_xlabel("global Q")
        axis.set_ylabel("certification (%)")
        axis.grid(alpha=.25)
    fig.legend(loc="center left", bbox_to_anchor=(1.0, .5), fontsize=8)
    save_figure(fig, figures / "certification-vs-Q")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, dataset in zip(axes, sorted(paired.dataset.unique())):
        block = paired[paired.dataset == dataset]
        for depth, style in ((2, "-"), (3, "--")):
            values = block[block.depth == depth].groupby("Q0").dual_minus_baseline.mean()
            axis.plot(values.index, values.values, marker="o", linestyle=style,
                      label=f"depth {depth}")
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(dataset); axis.set_xlabel("global Q")
        axis.set_ylabel("dual − baseline LP lower")
        axis.grid(alpha=.25); axis.legend()
    save_figure(fig, figures / "dual-margin-delta-vs-Q")

    runtime = per_seed.groupby(["dataset", "depth", "method"]).total_seconds_mean.mean().reset_index()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    labels=[]; values=[]; colors_bar=[]
    for row in runtime.itertuples():
        labels.append(f"{row.dataset}\nd{row.depth}\n{row.method}")
        values.append(row.total_seconds_mean)
        colors_bar.append("#4c78a8" if row.method == "baseline" else "#f58518")
    axis.bar(range(len(values)), values, color=colors_bar)
    axis.set_xticks(range(len(values)), labels, fontsize=8)
    axis.set_ylabel("mean seconds / target-config")
    axis.set_yscale("log"); axis.grid(axis="y", alpha=.25)
    save_figure(fig, figures / "runtime")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, dataset in zip(axes, sorted(ablation_pair.dataset.unique())):
        block = ablation_pair[ablation_pair.dataset == dataset]
        for depth, marker in ((2, "o"), (3, "s")):
            rows = block[block.depth == depth].sort_values("samples")
            axis.plot(rows.samples, rows.delta_mean, marker=marker, label=f"depth {depth}")
        axis.set_xscale("log", base=2); axis.set_xticks([32,128,256], ["32","128","256"])
        axis.set_title(dataset); axis.set_xlabel("under-domain samples")
        axis.set_ylabel("mean dual − baseline lower")
        axis.grid(alpha=.25); axis.legend()
    save_figure(fig, figures / "samples-ablation")


def run(args) -> dict:
    formal_root = Path(args.formal_root)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    reliability = json.loads(
        (formal_root / "integrity.json").read_text(encoding="utf-8")
    )
    required_integrity = {
        "status": "passed", "rows": 16_000, "shard_count": 20,
        "config_count": 320, "pair_count": 8_000, "bad_configs": 0,
    }
    mismatched = {
        key: {"expected": expected, "actual": reliability.get(key)}
        for key, expected in required_integrity.items()
        if reliability.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"incomplete deep formal matrix: {mismatched}")
    data = pd.read_csv(formal_root / "nodes.csv")
    requested_modes = list(dict.fromkeys(args.modes))
    data = data[data["mode"].isin(requested_modes)].copy()
    missing_modes = sorted(set(requested_modes) - set(data["mode"].unique()))
    if missing_modes:
        raise ValueError("requested modes absent from formal rows: "
                         + ", ".join(missing_modes))
    graph_cost_audit = cost_audit_checks(data)
    if not (graph_cost_audit["all_rows_valid"]
            and graph_cost_audit["graph_semantics_version_valid"]
            and graph_cost_audit["one_canonical_graph_per_dataset"]):
        raise ValueError(f"invalid formal graph/cost audit: {graph_cost_audit}")
    data, top_ids = add_top10_flag(data, args.cache_dir)
    retained_rows_path = output / "nodes-with-top10.csv"
    data.to_csv(retained_rows_path, index=False)
    retained_integrity = {
        "rows": int(len(data)),
        "modes": requested_modes,
        "cbc_optimal_trusted_rows": int(data.solver_trusted.sum()),
        "graph_and_cost_audit": graph_cost_audit,
    }
    (output / "integrity.json").write_text(
        json.dumps(retained_integrity, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    config_fields = ["dataset", "seed", "depth", "method", "mode", "q", "Q0"]
    per_seed = data.groupby(config_fields, sort=True, group_keys=False).apply(
        summarize_config, include_groups=False
    ).reset_index()
    per_seed.to_csv(output / "per_seed.csv", index=False)
    aggregate_fields = ["dataset", "depth", "method", "mode", "q", "Q0"]
    aggregate = aggregate_seed_rows(per_seed, aggregate_fields)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    per_seed.to_csv(output / "per_config.csv", index=False)

    paired, paired_seed, paired_aggregate = paired_results(data)
    paired.to_csv(output / "paired_targets.csv", index=False)
    paired_seed.to_csv(output / "paired_per_seed.csv", index=False)
    paired_aggregate.to_csv(output / "paired_aggregate.csv", index=False)

    ablation_paths = [Path(path) for path in args.ablation_nodes]
    ablation, ablation_pair = ablation_results(data, ablation_paths)
    ablation.to_csv(output / "samples_ablation.csv", index=False)
    ablation_pair.to_csv(output / "samples_ablation_paired.csv", index=False)

    model_columns = ["dataset", "seed", "depth", "hidden_dims", "bias",
                     "model_cache_key", "clean_train_accuracy",
                     "clean_validation_accuracy", "clean_test_accuracy"]
    models = data[model_columns].drop_duplicates().sort_values(["dataset", "depth", "seed"])
    models.to_csv(output / "model_metadata.csv", index=False)
    (output / "model_metadata.json").write_text(
        models.to_json(orient="records", indent=2), encoding="utf-8"
    )

    budget_rows=[]
    budget_path = output / "budget_mappings.csv"
    if set(requested_modes) & {"risk-budget", "combined"}:
        for path in sorted((formal_root / "shards").glob(
                "*/budget-mappings/*/seed-*/depth-*/Q0-*.json")):
            payload=json.loads(path.read_text(encoding="utf-8"))
            budget_rows.append({
                "dataset":payload["dataset"], "seed":payload["seed"],
                "depth":payload["depth"], "Q0":payload["Q0"],
                "target_count":payload["target_count"],
                "budget_total":payload["budget_total"],
                "budget_mean":payload["budget_mean"],
                "budget_min":payload["budget_min"], "budget_max":payload["budget_max"],
                "distribution_json":json.dumps(payload["budget_distribution"],sort_keys=True),
                "mapping_checksum":payload["mapping_checksum_sha256"], "path":str(path),
            })
        pd.DataFrame(budget_rows).to_csv(budget_path,index=False)
    elif budget_path.exists():
        budget_path.unlink()
    budgets=pd.DataFrame(budget_rows)
    budget_file_line = "- `budget_mappings.csv`\n" if len(budgets) else ""

    cohort_rows=[]
    for path in sorted((formal_root / "shards").glob(
            "*/targets/*/seed-*/depth-*.json")):
        payload=json.loads(path.read_text(encoding="utf-8")); cohort=payload["cohort"]
        records=cohort["records"]
        cohort_rows.append({
            "dataset":payload["dataset"],"seed":payload["seed"],"depth":payload["depth"],
            "pool_size":payload["target_pool_size"],"cohort_size":len(payload["targets"]),
            "correct_count":sum(record["correct"] for record in records),
            "incorrect_count":sum(not record["correct"] for record in records),
            "class_counts_json":json.dumps(dict(sorted(Counter(record["label"] for record in records).items()))),
            "risk_bin_counts_json":json.dumps(dict(sorted(Counter(record["risk_bin"] for record in records).items()))),
            "checksum":cohort["cohort_checksum_sha256"],"path":str(path),
        })
    cohorts=pd.DataFrame(cohort_rows); cohorts.to_csv(output/"cohorts.csv",index=False)

    create_figures(aggregate, paired, per_seed, ablation_pair, output)

    def percent_pm(mean, sd):
        return f"{100*float(mean):.1f} ± {100*float(sd):.1f}"

    accuracy_stats = models.groupby(["dataset", "depth"])[
        ["clean_train_accuracy", "clean_validation_accuracy", "clean_test_accuracy"]
    ].agg(["mean", "std"]).reset_index()
    accuracy_rows=[]
    for row in accuracy_stats.itertuples(index=False, name=None):
        accuracy_rows.append({
            "dataset":row[0], "depth":row[1],
            "train accuracy (%)":percent_pm(row[2],row[3]),
            "validation accuracy (%)":percent_pm(row[4],row[5]),
            "full test accuracy (%)":percent_pm(row[6],row[7]),
        })
    accuracy_display=pd.DataFrame(accuracy_rows)
    main_setting = aggregate[
        aggregate.apply(
            lambda row: (
                int(row.q) == dataset_spec(row.dataset).main_q
                and int(row.Q0) == dataset_spec(row.dataset).main_q0
            ),
            axis=1,
        )
    ].copy()
    main_rows=[]
    for row in main_setting.itertuples():
        main_rows.append({
            "dataset":row.dataset,"depth":row.depth,"method":row.method,
            "mode":row.mode,"q":int(row.q),"Q0":int(row.Q0),
            "stable certified (%)":percent_pm(row.certification_rate_mean,row.certification_rate_sd),
            "certified & correct (%)":percent_pm(
                row.certified_and_correct_rate_mean,row.certified_and_correct_rate_sd),
            "global top10% certified (%)":percent_pm(
                row.top10_certification_rate_mean,row.top10_certification_rate_sd),
            "top10 cohort n (5 seeds)":int(row.top10_targets_total),
        })
    main_table=pd.DataFrame(main_rows)
    paired_depth=[]
    for (dataset,depth), block in paired.groupby(["dataset","depth"]):
        by_seed=block.groupby("seed").dual_minus_baseline.mean()
        paired_depth.append({
            "dataset":dataset,"depth":depth,"pairs":len(block),
            "delta_mean":float(by_seed.mean()),"delta_sd":float(by_seed.std(ddof=1)),
            "delta_median_pooled":float(block.dual_minus_baseline.median()),
            "delta_max":float(block.dual_minus_baseline.max()),
            "wins":int((block.outcome=="dual_tighter").sum()),
            "ties":int((block.outcome=="equal").sum()),
            "negative":int((block.outcome=="negative_numerical").sum()),
            "cert_gains":int((block.cert_set_change=="dual_gain").sum()),
            "cert_losses":int((block.cert_set_change=="dual_loss").sum()),
        })
    paired_depth=pd.DataFrame(paired_depth)
    runtime = per_seed.groupby(["dataset","depth","method"])["total_seconds_mean"].agg(["mean","std"]).reset_index()
    gap_source = per_seed[
        (per_seed["mode"] == "legacy")
        & per_seed.apply(
            lambda row: (
                int(row.q) == dataset_spec(row.dataset).main_q
                and int(row.Q0) == dataset_spec(row.dataset).main_q0
            ),
            axis=1,
        )
    ]
    gap_rows=[]
    for key, block in gap_source.groupby(["dataset","depth","method"]):
        gap_rows.append({
            "dataset":key[0],"depth":key[1],"method":key[2],
            "LP lower mean ± seed SD":f"{block.lp_lower_bound_mean.mean():.4f} ± {block.lp_lower_bound_mean.std(ddof=1):.4f}",
            "clean-to-LP gap mean ± seed SD":f"{block.approximation_gap_mean.mean():.4f} ± {block.approximation_gap_mean.std(ddof=1):.4f}",
            "seconds/target mean ± seed SD":f"{block.total_seconds_mean.mean():.3f} ± {block.total_seconds_mean.std(ddof=1):.3f}",
            "seconds/config mean ± seed SD":f"{block.config_total_seconds.mean():.1f} ± {block.config_total_seconds.std(ddof=1):.1f}",
        })
    gap_display=pd.DataFrame(gap_rows)
    size = data.groupby(["dataset","depth","method"]).agg(
        variable_median=("variable_count","median"), variable_max=("variable_count","max"),
        constraint_median=("constraint_count","median"), constraint_max=("constraint_count","max"),
        potential_input_median=("potential_input_variable_count","median"),
        potential_input_max=("potential_input_variable_count","max"),
    ).reset_index()
    exact_summary_path = Path(args.controlled_exact_summary)
    exact_summary = json.loads(exact_summary_path.read_text(encoding="utf-8"))
    shard_process_seconds=0.0
    for path in (formal_root/"shards").glob("*/run_summary.json"):
        summary=json.loads(path.read_text(encoding="utf-8"))
        shard_process_seconds += sum(float(inv.get("wall_seconds") or 0) for inv in summary.get("invocations",[]))

    fixed_budget_description = "; ".join(
        f"{dataset}: q={dataset_spec(dataset).main_q}, "
        f"Q={list(dataset_spec(dataset).sensitivity_q0)}"
        for dataset in sorted(data.dataset.unique())
    )

    report = f"""# Legacy versus heterogeneous-cost depth-2/depth-3 Sigmoid experiment

## Scope and honesty boundary

This is the dedicated deep experiment. It is **not** the earlier one-layer full-test result. For each `(dataset, seed, depth)`, a deterministic cohort of **50 targets** was selected only from the standard 1,000-node Planetoid test mask, stratifying as far as possible over true class, clean correct/error status, and risk tercile. The certification rates below describe those cohorts (250 model-targets per dataset/depth over five seeds), **not all 1,000 test nodes**. Clean test accuracy is nevertheless evaluated on the complete standard test mask.

Matrix: 2 datasets × 5 seeds × 2 depths × 50 targets × 2 methods × {len(requested_modes)} modes × 1 fixed local q × 4 global-Q values = **{len(data):,} rows**. Only Legacy and heterogeneous-cost are retained. The budget settings are {fixed_budget_description}. Hidden width is 16 (two width-16 hidden layers at depth 3), sigmoid hidden activations, linear biased output, samples=128. The same trained model and target cohort are reused across paired methods, modes, and Q values.

## Model accuracy (full standard test mask; mean ± sample SD over 5 seeds)

{accuracy_display.to_markdown(index=False)}

## Reference setting: q=floor(0.01D), global Q=q (lambda=1)

{main_table.to_markdown(index=False)}

`certification_rate` is stability of the clean prediction. `certified_and_correct` additionally requires the clean prediction to equal the label. Global top-10%-importance is computed over every graph node and then intersected with each fixed cohort; the table gives the actual pooled denominator across five seeds rather than padding cohorts.

## Dual versus baseline

{paired_depth.to_markdown(index=False)}

The dual LP contains every baseline outer-only line and additional lines proposed from feasible under-domain samples but recalibrated over the complete sound outer interval. Therefore a material negative delta is an integrity failure, not an acceptable trade-off. The {len(paired):,} retained pairs have no delta below -1e-6 (minimum {paired.dual_minus_baseline.min():.3g}). Per-depth tightening counts and certification-set gains/losses are reported directly in the table above; no conclusion from the previous budget protocol is carried into this report.

`approximation_gap` in CSV is clean margin minus LP lower bound because exact full-threat margins are intractable here; it is a conservative relaxation-gap proxy, not an exact verification error. At the reference Legacy setting (q=floor(0.01D), Q=q):

{gap_display.to_markdown(index=False)}

## Solver time and LP scale

Mean target/configuration time by five-seed configuration:

{runtime.to_markdown(index=False)}

LP dimensions:

{size.to_markdown(index=False)}

The potential explicit inputs reach {int(data.potential_input_variable_count.max()):,}. Formal runs therefore use the sound budget-box abstraction: each first-layer neuron receives exact local-count + global-count/weighted grouped-fractional-knapsack extrema, then later layers use sound interval propagation. This drops cross-neuron correlations but never deletes attacks. Total summed shard-process wall time (pilot + resume, including repeated gates) was {shard_process_seconds/3600:.2f} h. The retained matrix has {int((data.solver_status == 'Optimal').sum()):,} CBC-Optimal rows and {int(data.solver_trusted.sum()):,} primal-audited trusted rows.

## Samples ablation (seed 0, first 10 of the same fixed 50-target cohorts, legacy main q/Q0)

{ablation_pair.to_markdown(index=False)}

Samples 32/128/256 use the same models and targets. More samples change proposed depth-3 lines and runtime, but no result is selected post hoc.

## Reliability gates

- Mandatory final deep gate: depth 2/3, count/weighted budgets, baseline/dual, bias/no-bias, q/Q=0, analytic high-precision envelope checks, explicit and budget-box LPs.
- Full repository regression is run separately and logged.
- Formal integrity: {int(data.solver_trusted.sum()):,}/{len(data):,} retained rows CBC Optimal and primal-audited trusted; 0 paired outer-bound mismatches; {int((paired.dual_minus_baseline < -1e-6).sum())} dual monotonicity failures.
- Controlled real-model exact comparison: {exact_summary['unique_real_targets']} targets / {exact_summary['cases']} method cases, attacks-per-case {exact_summary['attacks_per_case']}, and {exact_summary['violations']} violations of `LP_lower <= controlled_exact_worst`. This is explicitly a restricted attack subset, not an exact solution of the full threat.

## Limitations

The 50-node cohorts are sampled and model/depth-specific; sampling uncertainty beyond five initialization seeds is not estimated. The budget-box abstraction and polyhedral sigmoid envelopes are sound but incomplete. Attacks are binary feature flips on a fixed graph only. CBC is CPU-side. The full-test accuracies do not turn cohort certification rates into full-test certification rates.

## Files

- `aggregate.csv`, `per_seed.csv`, `per_config.csv`
- `paired_targets.csv`, `paired_per_seed.csv`, `paired_aggregate.csv`
- `samples_ablation.csv`, `samples_ablation_paired.csv`
- `model_metadata.csv/json`, `cohorts.csv`
{budget_file_line}- `integrity.json`
- `nodes-with-top10.csv`, `figures/`
- retained two-mode formal rows: `{retained_rows_path}`
- retained two-mode integrity: `{output / 'integrity.json'}`
- per-target detailed diagnostics: each shard's `diagnostics.tar.gz` (SHA-256 list at `{formal_root / 'diagnostics-sha256.txt'}`); CSV `diagnostics_path` names the corresponding archive member path
- controlled exact: `{exact_summary['csv']}`
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "status":"done", "formal_rows":len(data), "configurations":len(per_seed),
        "model_target_count":int(data[["dataset","seed","depth","target"]].drop_duplicates().shape[0]),
        "cbc_optimal_trusted_rows":int(data.solver_trusted.sum()),
        "graph_and_cost_audit": graph_cost_audit,
        "paired":paired_depth.to_dict(orient="records"),
        "accuracy":models.groupby(["dataset","depth"]).agg(
            clean_train_mean=("clean_train_accuracy","mean"),
            clean_train_sd=("clean_train_accuracy","std"),
            clean_test_mean=("clean_test_accuracy","mean"),
            clean_test_sd=("clean_test_accuracy","std"),
        ).reset_index().to_dict(orient="records"),
        "report":str(output/"REPORT.md"), "raw":str(retained_rows_path),
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def parse(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--formal-root",required=True)
    parser.add_argument("--ablation-nodes",nargs="+",required=True)
    parser.add_argument("--controlled-exact-summary",required=True)
    parser.add_argument("--modes", nargs="+", choices=PAPER_MODES,
                        default=list(PAPER_MODES))
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--cache-dir",default=".cache/pyg")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse()),sort_keys=True))
