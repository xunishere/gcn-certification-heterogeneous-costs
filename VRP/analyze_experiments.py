"""Dataset-safe analysis for one-layer certification experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset_registry import dataset_spec
from .experiment_config import MODES, PAPER_MODES
from .graph_semantics import GRAPH_SEMANTICS_VERSION
from .risk_budget import (
    COST_BUDGET_POLICY,
    RISK_BUDGET_POLICY,
    allocate_fixed_mean_budgets,
)

COLORS = {
    "legacy": "#4c78a8",
    "heterogeneous-cost": "#f58518",
    "risk-budget": "#54a24b",
    "combined": "#e45756",
}
DATASET_COLORS = {
    "Cora": "#4c78a8",
    "CiteSeer": "#e45756",
    "Karate": "#54a24b",
}


def ordered_present_modes(frame):
    present = set(frame["mode"].unique())
    return [mode for mode in MODES if mode in present]


def parse_label_path(specification):
    if "=" not in specification:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    label, path = specification.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("label must not be empty")
    return label, Path(path)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def validate_formal_paper_matrix(frame):
    """Fail closed unless the full two-dataset paper matrix is present."""
    formal = frame[frame["scope"] == "formal_test1000"]
    if formal.empty:
        return {"checked": False}
    expected_datasets = {"Cora", "CiteSeer"}
    if set(formal["dataset"].unique()) != expected_datasets:
        raise ValueError("formal paper matrix must contain Cora and CiteSeer")
    expected = set()
    for dataset in sorted(expected_datasets):
        specification = dataset_spec(dataset)
        for seed in range(5):
            for mode in PAPER_MODES:
                for q0 in specification.sensitivity_q0:
                    expected.add((dataset, seed, mode,
                                  specification.main_q, q0))
    actual = {
        (str(row.dataset), int(row.seed), str(row.mode),
         int(row.q), int(row.Q0))
        for row in formal[["dataset", "seed", "mode", "q", "Q0"]]
        .drop_duplicates().itertuples(index=False)
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"incomplete formal paper grid; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    sizes = formal.groupby(["dataset", "seed", "mode", "q", "Q0"]).agg(
        rows=("target", "size"), targets=("target", "nunique")
    )
    if not ((sizes["rows"] == 1000) & (sizes["targets"] == 1000)).all():
        raise ValueError("every formal configuration must contain 1,000 unique targets")
    expected_rows = len(expected) * 1000
    if len(formal) != expected_rows:
        raise ValueError(
            f"formal paper matrix has {len(formal):,} rows; "
            f"expected {expected_rows:,}"
        )
    return {
        "checked": True,
        "datasets": sorted(expected_datasets),
        "configurations": len(expected),
        "rows": expected_rows,
        "targets_per_configuration": 1000,
    }


def _sample_std(values):
    values = pd.Series(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def per_seed_metrics(frame):
    frame = frame.copy()
    if "certified_and_correct" not in frame:
        frame["certified_and_correct"] = frame["certified"] * frame["correct"]
    rows = []
    keys = ["scope", "dataset", "seed", "mode", "q", "Q0"]
    for key, group in frame.groupby(keys, sort=True):
        top = group[group["key_node"].astype(bool)]
        risk_sum = float(group["risk_weight"].sum())
        budget_total = int(group["Q_t"].sum())
        expected_total = int(len(group) * int(group["Q0"].iloc[0]))
        rows.append(dict(zip(keys, key)) | {
            "certification_rate": float(group["certified"].mean()),
            "top10_certification_rate": (
                float(top["certified"].mean()) if len(top) else np.nan
            ),
            "risk_weighted_certification_rate": (
                float((group["certified"] * group["risk_weight"]).sum() / risk_sum)
                if risk_sum > 0 else np.nan
            ),
            "certified_and_correct_rate": float(
                group["certified_and_correct"].mean()
            ),
            "clean_target_accuracy": float(group["correct"].mean()),
            "clean_test_accuracy": float(group["clean_test_accuracy"].iloc[0]),
            "validation_seconds": float(group["seconds"].sum()),
            "mean_node_validation_seconds": float(group["seconds"].mean()),
            "targets": int(len(group)),
            "top10_targets": int(len(top)),
            "budget_total": budget_total,
            "budget_expected_total": expected_total,
            "budget_conserved": bool(budget_total == expected_total),
            "Q_t_mean": float(group["Q_t"].mean()),
            "Q_t_min": int(group["Q_t"].min()),
            "Q_t_max": int(group["Q_t"].max()),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(per_seed):
    metric_columns = [
        "certification_rate", "top10_certification_rate",
        "risk_weighted_certification_rate", "certified_and_correct_rate",
        "clean_target_accuracy", "clean_test_accuracy", "validation_seconds",
        "mean_node_validation_seconds", "Q_t_mean", "Q_t_min", "Q_t_max",
    ]
    keys = ["scope", "dataset", "mode", "q", "Q0"]
    rows = []
    for key, group in per_seed.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["seeds"] = int(group["seed"].nunique())
        row["targets_per_seed_min"] = int(group["targets"].min())
        row["targets_per_seed_max"] = int(group["targets"].max())
        row["budget_conserved_all_seeds"] = bool(group["budget_conserved"].all())
        for metric in metric_columns:
            values = group[metric].astype(float)
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = _sample_std(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _save_figure(figure, base, tight=True):
    if tight:
        figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_heatmaps(aggregate, figure_dir):
    for (dataset, scope), subset in aggregate.groupby(["dataset", "scope"], sort=True):
        modes = ordered_present_modes(subset)
        q_values = sorted(int(value) for value in subset["q"].unique())
        q0_values = sorted(int(value) for value in subset["Q0"].unique())
        figure, axes = plt.subplots(
            1, len(modes), figsize=(5 * len(modes), 4.5),
            sharex=True, sharey=True, squeeze=False,
        )
        image = None
        for axis, mode in zip(axes.flat, modes):
            mode_rows = subset[subset["mode"] == mode]
            pivot = mode_rows.pivot(index="q", columns="Q0",
                                    values="certification_rate_mean")
            matrix = pivot.reindex(index=q_values, columns=q0_values).to_numpy()
            image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0,
                                vmax=1, cmap="viridis")
            for row_index in range(len(q_values)):
                for column_index in range(len(q0_values)):
                    value = matrix[row_index, column_index]
                    if np.isfinite(value):
                        axis.text(column_index, row_index, f"{100*value:.1f}",
                                  ha="center", va="center", fontsize=8,
                                  color="white" if value < .55 else "black")
            axis.set_title(mode)
            axis.set_xticks(range(len(q0_values)), q0_values)
            axis.set_yticks(range(len(q_values)), q_values)
            axis.set_xlabel("Q0")
            axis.set_ylabel("local count q")
        figure.suptitle(f"q-Q certification: {dataset} / {scope}")
        figure.subplots_adjust(left=.08, right=.84, bottom=.08, top=.9,
                               wspace=.22, hspace=.28)
        color_axis = figure.add_axes([.875, .16, .025, .68])
        figure.colorbar(image, cax=color_axis, label="certification rate")
        _save_figure(
            figure,
            figure_dir / f"q-Q_{safe_name(dataset)}_{safe_name(scope)}",
            tight=False,
        )


def plot_main_curves(aggregate, figure_dir):
    panels = [
        ("certification_rate", "overall certification"),
        ("top10_certification_rate", "importance top-10%"),
        ("risk_weighted_certification_rate", "risk-weighted"),
        ("certified_and_correct_rate", "certified and correct"),
    ]
    for (dataset, scope), all_rows in aggregate.groupby(["dataset", "scope"], sort=True):
        modes = ordered_present_modes(all_rows)
        main_q = dataset_spec(dataset).main_q
        subset = all_rows[all_rows["q"] == main_q]
        if subset.empty:
            continue
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for axis, (metric, title) in zip(axes.flat, panels):
            for mode in modes:
                rows = subset[subset["mode"] == mode].sort_values("Q0")
                if rows.empty:
                    continue
                axis.errorbar(rows["Q0"], rows[metric + "_mean"],
                              yerr=rows[metric + "_std"], marker="o", capsize=3,
                              color=COLORS[mode], label=mode)
            axis.set_title(title)
            axis.set_xlabel("Q0")
            axis.set_ylim(-.03, 1.03)
            axis.grid(alpha=.25)
        axes[0, 0].set_ylabel("rate (mean ± sample SD)")
        axes[1, 0].set_ylabel("rate (mean ± sample SD)")
        axes[0, 1].legend(fontsize=8)
        figure.suptitle(f"Main q={main_q}: {dataset} / {scope}")
        _save_figure(
            figure,
            figure_dir / f"main-q{main_q}_{safe_name(dataset)}_{safe_name(scope)}",
        )


def plot_runtime(aggregate, figure_dir):
    for (dataset, scope), all_rows in aggregate.groupby(["dataset", "scope"], sort=True):
        modes = ordered_present_modes(all_rows)
        main_q = dataset_spec(dataset).main_q
        subset = all_rows[all_rows["q"] == main_q]
        if subset.empty:
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for mode in modes:
            rows = subset[subset["mode"] == mode].sort_values("Q0")
            if rows.empty:
                continue
            axis.errorbar(rows["Q0"], rows["validation_seconds_mean"],
                          yerr=rows["validation_seconds_std"], marker="o", capsize=3,
                          color=COLORS[mode], label=mode)
        axis.set_xlabel("Q0")
        axis.set_ylabel("exact validation seconds / seed / configuration")
        axis.set_title(f"Validation time at q={main_q}: {dataset} / {scope}")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        _save_figure(
            figure,
            figure_dir / f"runtime-q{main_q}_{safe_name(dataset)}_{safe_name(scope)}",
        )


def plot_budget_distributions(distribution, figure_dir):
    for (dataset, scope), subset in distribution.groupby(["dataset", "scope"], sort=True):
        q0_values = sorted(int(value) for value in subset["Q0"].unique())
        figure, axes = plt.subplots(len(q0_values), 1,
                                   figsize=(8, max(3, 2.5 * len(q0_values))),
                                   squeeze=False)
        for axis, q0 in zip(axes.flat, q0_values):
            rows = subset[subset["Q0"] == q0]
            pooled = rows.groupby("Q_t", sort=True)["count"].sum()
            proportions = pooled / pooled.sum()
            axis.bar(proportions.index.astype(int), proportions.values,
                     color="#54a24b", alpha=.85)
            axis.axvline(q0, color="black", linestyle="--", linewidth=1,
                         label=f"mean Q0={q0}")
            axis.set_ylabel("proportion")
            axis.set_title(f"Q0={q0}; pooled over {rows['seed'].nunique()} seeds")
            axis.legend(fontsize=8)
        axes[-1, 0].set_xlabel("allocated integer Q_t")
        figure.suptitle(f"Fixed-mean budget distributions: {dataset} / {scope}")
        _save_figure(
            figure,
            figure_dir / f"budget-distribution_{safe_name(dataset)}_{safe_name(scope)}",
        )


def plot_dataset_comparison(aggregate, figure_dir):
    for scope, scope_rows in aggregate.groupby("scope", sort=True):
        datasets = sorted(scope_rows["dataset"].unique())
        if len(datasets) < 2:
            continue
        modes = ordered_present_modes(scope_rows)
        figure, axes = plt.subplots(
            1, len(modes), figsize=(5.5 * len(modes), 4.5),
            sharex=True, sharey=True, squeeze=False,
        )
        plotted = False
        for axis, mode in zip(axes.flat, modes):
            for dataset in datasets:
                main_q = dataset_spec(dataset).main_q
                rows = scope_rows[(scope_rows["dataset"] == dataset)
                                  & (scope_rows["mode"] == mode)
                                  & (scope_rows["q"] == main_q)].sort_values("Q0")
                if rows.empty:
                    continue
                plotted = True
                axis.errorbar(rows["Q0"], rows["certified_and_correct_rate_mean"],
                              yerr=rows["certified_and_correct_rate_std"], marker="o",
                              capsize=3, color=DATASET_COLORS.get(dataset),
                              label=f"{dataset} (q={main_q})")
            axis.set_title(mode)
            axis.set_xlabel("Q0")
            axis.set_ylabel("certified-and-correct rate")
            axis.set_ylim(-.03, 1.03)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        if plotted:
            figure.suptitle(f"Dataset comparison at each dataset's main q: {scope}")
            _save_figure(figure, figure_dir / f"dataset-comparison_{safe_name(scope)}")
        else:
            plt.close(figure)


def _load_budget_manifests(run_specs):
    summaries, distributions = [], []
    seen_paths = set()
    for scope, csv_path in run_specs:
        mapping_root = csv_path.parent / "budget-mappings"
        if not mapping_root.exists():
            continue
        for path in sorted(mapping_root.glob("**/Q0-*.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload["records"]
            risks = {int(row["target"]): float(row["risk"]) for row in records}
            rebuilt = allocate_fixed_mean_budgets(risks, int(payload["Q0"]),
                                                   float(payload["beta"]))
            valid = (
                payload.get("policy") == RISK_BUDGET_POLICY
                and rebuilt.checksum == payload.get("mapping_checksum_sha256")
                and rebuilt.total == int(payload["budget_total"])
                and rebuilt.budgets == {int(row["target"]): int(row["Q_t"])
                                        for row in records}
            )
            summaries.append({
                "scope": scope,
                "dataset": payload["dataset"],
                "seed": int(payload["seed"]),
                "Q0": int(payload["Q0"]),
                "target_count": int(payload["target_count"]),
                "budget_total": int(payload["budget_total"]),
                "budget_mean": float(payload["budget_mean"]),
                "budget_min": int(payload["budget_min"]),
                "budget_max": int(payload["budget_max"]),
                "mapping_checksum_sha256": payload["mapping_checksum_sha256"],
                "mapping_valid": bool(valid),
                "path": str(path),
            })
            for q_t, count in payload["budget_distribution"].items():
                distributions.append({
                    "scope": scope,
                    "dataset": payload["dataset"],
                    "seed": int(payload["seed"]),
                    "Q0": int(payload["Q0"]),
                    "Q_t": int(q_t),
                    "count": int(count),
                })
    return pd.DataFrame(summaries), pd.DataFrame(distributions)


def cost_audit_checks(frame):
    columns = [
        "dataset", "seed", "target", "mode", "uses_heterogeneous_cost",
        "target_cost", "candidate_source_count", "candidate_sources",
        "candidate_source_costs", "graph_semantics_version",
        "canonical_graph_sha256",
    ]
    unique = frame[columns].drop_duplicates()
    valid_rows = 0
    errors = []
    for row in unique.itertuples(index=False):
        try:
            sources = json.loads(row.candidate_sources)
            costs = json.loads(row.candidate_source_costs)
            expected_cost_mode = row.mode in ("heterogeneous-cost", "combined")
            if bool(int(row.uses_heterogeneous_cost)) != expected_cost_mode:
                raise ValueError("cost-mode flag mismatch")
            if len(sources) != int(row.candidate_source_count) or len(costs) != len(sources):
                raise ValueError("candidate source/cost length mismatch")
            if int(row.target) not in sources:
                raise ValueError("target missing from candidate sources")
            target_index = sources.index(int(row.target))
            if not np.isclose(float(costs[target_index]), float(row.target_cost)):
                raise ValueError("target cost is not aligned with candidate vector")
            if any(not np.isfinite(float(cost)) or float(cost) <= 0 for cost in costs):
                raise ValueError("candidate costs must be finite and positive")
            valid_rows += 1
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            if len(errors) < 10:
                errors.append(
                    f"{row.dataset}/seed{row.seed}/target{row.target}/{row.mode}: {error}"
                )
    hashes = frame.groupby("dataset")["canonical_graph_sha256"].nunique()
    return {
        "unique_rows_checked": int(len(unique)),
        "valid_rows": int(valid_rows),
        "all_rows_valid": bool(valid_rows == len(unique)),
        "graph_semantics_version_valid": bool(
            set(frame["graph_semantics_version"].unique())
            == {GRAPH_SEMANTICS_VERSION}
        ),
        "one_canonical_graph_per_dataset": bool((hashes == 1).all()),
        "errors": errors,
    }


def integrity_checks(frame, budget_summary):
    keys = ["scope", "dataset", "seed", "mode", "q", "Q0", "target"]
    integer_qt = np.isclose(frame["Q_t"], np.round(frame["Q_t"]))
    nonnegative_qt = frame["Q_t"] >= 0
    budget_groups = []
    monotone = True
    for key, group in frame[frame["mode"].isin(("risk-budget", "combined"))].groupby(
            ["scope", "dataset", "seed", "mode", "q", "Q0"], sort=True):
        expected = len(group) * int(key[-1])
        budget_groups.append(int(group["Q_t"].sum()) == expected)
        ordered = group.sort_values(["risk_weight", "target"])
        distinct = ordered.groupby("risk_weight", sort=True)["Q_t"].agg(["min", "max"])
        if len(distinct) > 1:
            monotone = monotone and bool(
                (distinct["min"].iloc[1:].to_numpy()
                 >= np.maximum.accumulate(distinct["max"].iloc[:-1].to_numpy())).all()
            )

    risk_modes = frame[frame["mode"].isin(("risk-budget", "combined"))].copy()
    per_mode_target = (risk_modes.groupby(
        ["scope", "dataset", "seed", "Q0", "mode", "target"], sort=True
    ).agg(Q_t=("Q_t", "nunique"), Q_t_value=("Q_t", "first"),
          checksum_count=("risk_budget_mapping_checksum", "nunique"),
          checksum=("risk_budget_mapping_checksum", "first")).reset_index())
    q_invariant = bool((per_mode_target["Q_t"] == 1).all()
                       and (per_mode_target["checksum_count"] == 1).all())
    pivot_q = per_mode_target.pivot_table(
        index=["scope", "dataset", "seed", "Q0", "target"], columns="mode",
        values="Q_t_value", aggfunc="first"
    )
    common = pivot_q.dropna(subset=[column for column in ("risk-budget", "combined")
                                    if column in pivot_q.columns])
    mode_mapping_equal = (bool((common["risk-budget"] == common["combined"]).all())
                          if {"risk-budget", "combined"}.issubset(common.columns)
                          else False)
    checksum_pivot = per_mode_target.pivot_table(
        index=["scope", "dataset", "seed", "Q0", "target"], columns="mode",
        values="checksum", aggfunc="first"
    ).dropna()
    checksum_equal = (bool((checksum_pivot["risk-budget"]
                            == checksum_pivot["combined"]).all())
                      if {"risk-budget", "combined"}.issubset(checksum_pivot.columns)
                      else False)

    fixed_cost_identity = (frame["Q_cost"] - frame["Q_equiv"]).abs()
    heterogeneous = frame[frame["mode"] == "heterogeneous-cost"]
    combined = frame[frame["mode"] == "combined"]
    containment_rows = frame[frame["mode"].isin(
        ("legacy", "heterogeneous-cost")
    )]
    containment = containment_rows.pivot(
        index=["scope", "dataset", "seed", "q", "Q0", "target"],
        columns="mode", values=["certified", "certified_margin"],
    ).dropna()
    if len(containment):
        margin_delta = (
            containment[("certified_margin", "heterogeneous-cost")]
            - containment[("certified_margin", "legacy")]
        )
        certificate_violations = int((
            (containment[("certified", "legacy")] == 1)
            & (containment[("certified", "heterogeneous-cost")] == 0)
        ).sum())
        margin_violations = int((margin_delta < -1e-9).sum())
        minimum_margin_delta = float(margin_delta.min())
    else:
        certificate_violations = margin_violations = 0
        minimum_margin_delta = float("nan")
    return {
        "rows": int(len(frame)),
        "schema_versions": sorted(int(value) for value in frame["schema_version"].unique()),
        "duplicate_configuration_target_rows": int(frame.duplicated(keys).sum()),
        "Q_t_nonnegative_integer": bool(integer_qt.all() and nonnegative_qt.all()),
        "risk_budget_configurations_conserved": bool(all(budget_groups)),
        "risk_budget_monotone": bool(monotone),
        "risk_budget_q_invariant": q_invariant,
        "risk_and_combined_mapping_equal": mode_mapping_equal,
        "risk_and_combined_checksum_equal": checksum_equal,
        "max_fixed_cost_budget_error": float(fixed_cost_identity.max()),
        "heterogeneous_uses_Q0": bool((heterogeneous["Q_t"]
                                        == heterogeneous["Q0"]).all()),
        "legacy_hetero_target_pairs": int(len(containment)),
        "legacy_certified_but_hetero_not": certificate_violations,
        "hetero_margin_below_legacy": margin_violations,
        "minimum_hetero_minus_legacy_margin": minimum_margin_delta,
        "combined_uses_Q_t": bool((combined["Q_equiv"]
                                    == combined["Q_t"]).all()),
        "budget_mapping_manifests": int(len(budget_summary)),
        "budget_mapping_manifests_valid": (
            bool(budget_summary["mapping_valid"].all()) if len(budget_summary) else False
        ),
    }


def paper_mode_integrity(integrity):
    """Keep only checks relevant to the Legacy/Hetero paper protocol."""
    keys = (
        "rows", "schema_versions", "duplicate_configuration_target_rows",
        "max_fixed_cost_budget_error", "heterogeneous_uses_Q0",
        "legacy_hetero_target_pairs", "legacy_certified_but_hetero_not",
        "hetero_margin_below_legacy", "minimum_hetero_minus_legacy_margin",
        "graph_and_cost_audit",
    )
    return {key: integrity[key] for key in keys}


def _format_rate(mean, std):
    if not np.isfinite(std):
        return f"{100*mean:.2f}% (one seed)"
    return f"{100*mean:.2f}% ± {100*std:.2f} pp"


def _format_seconds(mean, std):
    if not np.isfinite(std):
        return f"{mean:.3f}s (one seed)"
    return f"{mean:.3f} ± {std:.3f}s"


def _markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def build_report(frame, per_seed, aggregate, budget_summary, distribution,
                 integrity, run_summaries):
    lines = [
        "# Fixed-mean risk-budget Cora/CiteSeer experiments",
        "",
        "## Method and fairness boundary",
        "",
        "For every actual `(dataset, seed, target_scope)` target set, all target "
        "risks are computed before allocation. Continuous scores "
        "`Q0 + beta*(r_t-mean(r))` are projected onto the non-negative simplex "
        "with total `N*Q0`, then deterministically integerized by largest "
        "remainder (target id breaks ties). Thus every Q_t is a non-negative "
        "integer, risk is weakly monotone with budget, and mean Q_t is exactly Q0. "
        "Risk-budget and combined consume the same persisted Q_t mapping.",
        "",
        "Heterogeneous-cost uses the fixed resource limit `Q_cost=Q0`; combined "
        "uses `Q_cost=Q_t`. Candidate mean cost is reported for audit but never "
        "increases either budget. Local q remains a feature-flip count.",
        "",
        "**Budget semantics:** count and cost modes receive the same numeric global "
        "resource limit. A high-cost source therefore consumes more of that fixed "
        "limit and can reduce the number of feasible flips. Fixed-mean risk allocation "
        "still redistributes Q_t across targets while preserving its target-set mean.",
        "",
        "## Evaluated scope",
        "",
    ]
    scope_rows = []
    group_keys = ["scope", "dataset"]
    for (scope, dataset), group in frame.groupby(group_keys, sort=True):
        config_sizes = group.groupby(["seed", "mode", "q", "Q0"]).size()
        scope_rows.append([
            scope, dataset, int(group["num_nodes"].iloc[0]),
            int(group["num_features"].iloc[0]), int(group["seed"].nunique()),
            f"{int(config_sizes.min())}–{int(config_sizes.max())}",
            ",".join(str(int(value)) for value in sorted(group["q"].unique())),
            f"{group.groupby(['seed','mode','q','Q0']).ngroups:,}",
            f"{len(group):,}", str(group["target_scope"].iloc[0]),
        ])
    lines.extend([
        _markdown_table(scope_rows, ["scope", "dataset", "nodes", "features",
                                     "seeds", "targets/config", "q", "configs",
                                     "rows", "target mask"]),
        "",
        "Planetoid formal rows use the standard test mask; the table gives its "
        "explicit target count per seed/configuration.",
        "",
        "## Fixed-local-q global-Q sweep",
        "",
        "Values are five-seed mean ± sample SD where five seeds are present. "
        "The top10% subset uses the global graph importance 90th-percentile "
        "threshold intersected with the evaluated mask; exact subset counts are "
        "recorded in per_seed.csv. `certified-and-correct` counts only targets "
        "whose clean prediction is correct and whose predicted class is certified "
        "stable. Validation time is exact optimizer time only; training, gain "
        "materialization, and I/O are excluded.",
        "",
    ])
    main_rows = []
    for (scope, dataset), group in aggregate.groupby(["scope", "dataset"], sort=True):
        main_q = dataset_spec(dataset).main_q
        selected = group[group["q"] == main_q]
        for _, row in selected.sort_values(["Q0", "mode"]).iterrows():
            main_rows.append([
                scope, dataset, main_q, int(row["Q0"]), row["mode"],
                _format_rate(row["certification_rate_mean"],
                             row["certification_rate_std"]),
                _format_rate(row["top10_certification_rate_mean"],
                             row["top10_certification_rate_std"]),
                _format_rate(row["risk_weighted_certification_rate_mean"],
                             row["risk_weighted_certification_rate_std"]),
                _format_rate(row["certified_and_correct_rate_mean"],
                             row["certified_and_correct_rate_std"]),
                _format_rate(row["clean_test_accuracy_mean"],
                             row["clean_test_accuracy_std"]),
                _format_seconds(row["validation_seconds_mean"],
                                row["validation_seconds_std"]),
                "yes" if row["budget_conserved_all_seeds"] else "NO",
            ])
    lines.extend([
        _markdown_table(main_rows, ["scope", "dataset", "q", "Q0", "mode",
                                    "overall", "top10", "risk-weighted",
                                    "certified-and-correct", "clean test acc.",
                                    "validation time", "budget conserved"]),
        "",
        "## Local-q sensitivity at each dataset's main Q0",
        "",
    ])
    sensitivity_rows = []
    sensitivity = aggregate[
        aggregate.apply(
            lambda row: int(row["Q0"]) == dataset_spec(row["dataset"]).main_q0,
            axis=1,
        )
    ]
    for _, row in sensitivity.sort_values(
            ["scope", "dataset", "q", "mode"]).iterrows():
        sensitivity_rows.append([
            row["scope"], row["dataset"], int(row["Q0"]), int(row["q"]),
            row["mode"],
            _format_rate(row["certification_rate_mean"],
                         row["certification_rate_std"]),
            _format_rate(row["certified_and_correct_rate_mean"],
                         row["certified_and_correct_rate_std"]),
        ])
    lines.extend([
        _markdown_table(sensitivity_rows, ["scope", "dataset", "Q0", "q",
                                           "mode", "overall",
                                           "certified-and-correct"]),
        "",
        "## Fixed-mean budget distributions",
        "",
    ])
    budget_rows = []
    if len(budget_summary):
        for key, group in budget_summary.groupby(["scope", "dataset", "Q0"], sort=True):
            scope, dataset, q0 = key
            pooled = distribution[(distribution["scope"] == scope)
                                  & (distribution["dataset"] == dataset)
                                  & (distribution["Q0"] == q0)].groupby("Q_t")["count"].sum()
            distribution_text = ", ".join(
                f"{int(q_t)}:{int(count)}" for q_t, count in pooled.items()
            )
            budget_rows.append([
                scope, dataset, int(q0), int(group["seed"].nunique()),
                f"{int(group['budget_min'].min())}–{int(group['budget_max'].max())}",
                f"{group['budget_mean'].mean():.6g}", distribution_text,
                "yes" if group["mapping_valid"].all() else "NO",
            ])
    lines.extend([
        _markdown_table(budget_rows, ["scope", "dataset", "Q0", "seeds",
                                      "min–max", "mean", "pooled Q_t:count",
                                      "checksum/rebuild valid"]),
        "",
        "Each per-seed mapping JSON contains target id, risk, continuous/projected "
        "score, integer Q_t, checksum, exact mean/min/max, and distribution.",
        "",
        "## Integrity and runtime",
        "",
        f"- Rows: {integrity['rows']:,}; duplicate configuration-target rows: "
        f"{integrity['duplicate_configuration_target_rows']}.",
        f"- Non-negative integer Q_t: {integrity['Q_t_nonnegative_integer']}; "
        f"strict sum conservation: {integrity['risk_budget_configurations_conserved']}; "
        f"risk monotonicity: {integrity['risk_budget_monotone']}.",
        f"- Risk/combined mapping and checksum equality: "
        f"{integrity['risk_and_combined_mapping_equal']} / "
        f"{integrity['risk_and_combined_checksum_equal']}.",
        f"- Mapping manifests rebuilt successfully: "
        f"{integrity['budget_mapping_manifests_valid']} "
        f"({integrity['budget_mapping_manifests']} files).",
        f"- Maximum `|Q_cost-Q_equiv|` under fixed-resource semantics: "
        f"{integrity['max_fixed_cost_budget_error']:.3g}.",
        "",
    ])
    if run_summaries:
        runtime_rows = [[
            label, payload.get("status"), f"{payload.get('rows', 0):,}",
            payload.get("expected_configurations"),
            f"{payload.get('first_completed_wall_seconds', payload.get('wall_seconds_this_invocation', float('nan'))):.3f}",
        ] for label, payload in run_summaries]
        lines.extend([
            _markdown_table(runtime_rows, ["run", "status", "rows", "configs",
                                           "first complete wall seconds"]),
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "The certificate is exact only for binary feature flips of the trained "
        "one-layer linear GCN. It does not establish robustness of deeper nonlinear "
        "GNNs or other perturbation types. Planetoid masks are fixed while model "
        "initialization varies across seeds. Overall certification includes stable "
        "but initially incorrect predictions; certified-and-correct should be used "
        "when correctness matters. Fixed average budget improves one fairness axis "
        "but does not make risk-stratified, count, and cost attack sets equivalent.",
        "",
    ])
    return "\n".join(lines)


def build_paper_mode_report(frame, per_seed, aggregate, integrity, run_summaries):
    lines = [
        "# Legacy versus heterogeneous-cost one-layer experiments",
        "",
        "## Threat configurations",
        "",
        "Only the two configurations that answer the paper's source-cost question "
        "are retained. Legacy uses the global count constraint `sum z <= Q0`; "
        "heterogeneous-cost uses the fixed weighted constraint `sum c_v z <= Q0`. "
        "Both use the same per-source local count limit `q`, and candidate mean "
        "cost never rescales the global budget.",
        "",
        "## Evaluated scope",
        "",
    ]
    scope_rows = []
    for (scope, dataset), group in frame.groupby(["scope", "dataset"], sort=True):
        config_sizes = group.groupby(["seed", "mode", "q", "Q0"]).size()
        scope_rows.append([
            scope, dataset, int(group["num_nodes"].iloc[0]),
            int(group["num_features"].iloc[0]), int(group["seed"].nunique()),
            f"{int(config_sizes.min())}–{int(config_sizes.max())}",
            ",".join(str(int(value)) for value in sorted(group["q"].unique())),
            f"{group.groupby(['seed','mode','q','Q0']).ngroups:,}",
            f"{len(group):,}", str(group["target_scope"].iloc[0]),
        ])
    lines.extend([
        _markdown_table(scope_rows, ["scope", "dataset", "nodes", "features",
                                     "seeds", "targets/config", "q", "configs",
                                     "rows", "target mask"]),
        "",
        "## Main-q results",
        "",
    ])
    main_rows = []
    for (scope, dataset), group in aggregate.groupby(["scope", "dataset"], sort=True):
        main_q = dataset_spec(dataset).main_q
        for _, row in group[group["q"] == main_q].sort_values(["Q0", "mode"]).iterrows():
            main_rows.append([
                scope, dataset, main_q, int(row["Q0"]), row["mode"],
                _format_rate(row["certification_rate_mean"],
                             row["certification_rate_std"]),
                _format_rate(row["top10_certification_rate_mean"],
                             row["top10_certification_rate_std"]),
                _format_rate(row["risk_weighted_certification_rate_mean"],
                             row["risk_weighted_certification_rate_std"]),
                _format_rate(row["certified_and_correct_rate_mean"],
                             row["certified_and_correct_rate_std"]),
                _format_seconds(row["validation_seconds_mean"],
                                row["validation_seconds_std"]),
            ])
    lines.extend([
        _markdown_table(main_rows, ["scope", "dataset", "q", "Q0", "mode",
                                    "stable certified", "top10",
                                    "risk-weighted", "certified-and-correct",
                                    "validation time"]),
        "",
        "## Integrity",
        "",
        f"- Filtered rows: {integrity['rows']:,}; duplicate configuration-target "
        f"rows: {integrity['duplicate_configuration_target_rows']}.",
        f"- Heterogeneous-cost always uses the fixed Q0 resource: "
        f"{integrity['heterogeneous_uses_Q0']}.",
        f"- Maximum `|Q_cost-Q_equiv|`: "
        f"{integrity['max_fixed_cost_budget_error']:.3g}.",
        f"- Legacy/Hetero paired targets: "
        f"{integrity['legacy_hetero_target_pairs']:,}; Legacy-certified but "
        f"Hetero-uncertified violations: "
        f"{integrity['legacy_certified_but_hetero_not']}; Hetero margin below "
        f"Legacy violations: {integrity['hetero_margin_below_legacy']}.",
        f"- Graph and source-cost audit passed: "
        f"{integrity['graph_and_cost_audit']['all_rows_valid']}.",
        "",
        "## Limitations",
        "",
        "The one-layer certificate is exact for binary feature flips of the "
        "trained linear GCN under the stated local and global constraints. "
        "Differences between Legacy and Hetero describe different threat sets; "
        "they are not changes to the trained model.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_label_path, required=True,
                        help="SCOPE_LABEL=nodes.csv; labels may repeat across datasets")
    parser.add_argument("--run-summary", action="append", type=parse_label_path,
                        default=[], help="LABEL=run_summary.json")
    parser.add_argument("--modes", nargs="+", choices=PAPER_MODES,
                        default=list(PAPER_MODES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    frames = []
    for label, path in args.run:
        frame = pd.read_csv(path)
        frame["scope"] = label
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    requested_modes = list(dict.fromkeys(args.modes))
    frame = frame[frame["mode"].isin(requested_modes)].copy()
    missing_modes = sorted(set(requested_modes) - set(frame["mode"].unique()))
    if missing_modes:
        raise ValueError("requested modes absent from result rows: "
                         + ", ".join(missing_modes))
    formal_protocol = (
        validate_formal_paper_matrix(frame)
        if tuple(requested_modes) == tuple(PAPER_MODES)
        else {"checked": False}
    )
    required = {
        "schema_version", "graph_semantics_version", "canonical_graph_sha256",
        "dataset", "seed", "mode", "q", "Q0", "Q_t",
        "Q_equiv", "Q_cost", "mean_cost", "global_budget", "target", "certified", "correct",
        "key_node", "risk_weight", "seconds", "clean_test_accuracy",
        "risk_budget_mapping_checksum", "cost_budget_policy",
        "uses_heterogeneous_cost", "target_cost", "candidate_source_count",
        "candidate_sources", "candidate_source_costs",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing required result fields: " + ", ".join(missing))
    if set(frame["budget_policy"].unique()) != {RISK_BUDGET_POLICY}:
        raise ValueError("incompatible risk-budget policy in input rows")
    if set(frame["cost_budget_policy"].unique()) != {COST_BUDGET_POLICY}:
        raise ValueError("incompatible cost-budget policy in input rows")
    cost_audit = cost_audit_checks(frame)
    if not (cost_audit["all_rows_valid"]
            and cost_audit["graph_semantics_version_valid"]
            and cost_audit["one_canonical_graph_per_dataset"]):
        raise ValueError("invalid graph/cost audit metadata: " + json.dumps(cost_audit))

    per_seed = per_seed_metrics(frame)
    aggregate = aggregate_metrics(per_seed)
    if set(requested_modes) & {"risk-budget", "combined"}:
        budget_summary, distribution = _load_budget_manifests(args.run)
    else:
        budget_summary, distribution = pd.DataFrame(), pd.DataFrame()
    integrity = integrity_checks(frame, budget_summary)
    integrity["graph_and_cost_audit"] = cost_audit

    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    budget_path = args.output_dir / "budget_mappings.csv"
    distribution_path = args.output_dir / "budget_distribution.csv"
    if len(budget_summary):
        budget_summary.to_csv(budget_path, index=False)
        distribution.to_csv(distribution_path, index=False)
    else:
        for stale_path in (budget_path, distribution_path):
            if stale_path.exists():
                stale_path.unlink()
    output_integrity = (
        paper_mode_integrity(integrity)
        if tuple(requested_modes) == tuple(PAPER_MODES)
        else integrity
    )
    output_integrity["formal_paper_matrix"] = formal_protocol
    (args.output_dir / "integrity.json").write_text(
        json.dumps(output_integrity, indent=2, sort_keys=True), encoding="utf-8"
    )

    plot_heatmaps(aggregate, figure_dir)
    plot_main_curves(aggregate, figure_dir)
    plot_runtime(aggregate, figure_dir)
    if len(distribution):
        plot_budget_distributions(distribution, figure_dir)
    plot_dataset_comparison(aggregate, figure_dir)

    run_summaries = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.run_summary
    ]
    if tuple(requested_modes) == tuple(PAPER_MODES):
        report = build_paper_mode_report(
            frame, per_seed, aggregate, output_integrity, run_summaries
        )
    else:
        report = build_report(frame, per_seed, aggregate, budget_summary,
                              distribution, integrity, run_summaries)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "rows": int(len(frame)),
        "aggregate_rows": int(len(aggregate)),
        "budget_mappings": int(len(budget_summary)),
        "integrity": output_integrity,
        "figures_png": len(list(figure_dir.glob("*.png"))),
        "report": str(args.output_dir / "REPORT.md"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
