"""Consolidate and integrity-check independent deep-experiment shards."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from .analyze_experiments import cost_audit_checks
from .dataset_registry import dataset_spec
from .deep_experiments import DEEP_EXPERIMENT_SCHEMA, METHODS
from .experiment_config import PAPER_MODES


def run(args):
    root=Path(args.shards_root); output=Path(args.output_root); output.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob("*/nodes.csv"))
    if not paths:
        raise RuntimeError(f"no shard nodes.csv under {root}")
    frames=[]; issues=[]; shards={}
    required_audit_fields = {
        "schema_version", "graph_semantics_version", "canonical_graph_sha256",
        "uses_heterogeneous_cost", "target_cost", "candidate_source_count",
        "candidate_sources", "candidate_source_costs",
        "solver_random_seed",
    }
    for path in paths:
        frame=pd.read_csv(path)
        missing = sorted(required_audit_fields - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"missing graph/cost audit fields in {path}: {', '.join(missing)}"
            )
        if set(frame.schema_version.unique()) != {DEEP_EXPERIMENT_SCHEMA}:
            raise RuntimeError(f"incompatible deep schema in {path}")
        audit = cost_audit_checks(frame)
        if not (audit["all_rows_valid"]
                and audit["graph_semantics_version_valid"]
                and audit["one_canonical_graph_per_dataset"]):
            raise RuntimeError(f"invalid graph/cost audit in {path}: {audit}")
        frames.append(frame)
        summary=json.loads((path.parent/"run_summary.json").read_text(encoding="utf-8"))
        run_config=json.loads((path.parent/"run_config.json").read_text(encoding="utf-8"))
        expected_done = sum(
            len(run_config["seeds"]) * len(run_config["depths"])
            * len(run_config["methods"]) * len(run_config["modes"])
            * len(run_config["q_values_by_dataset"][dataset])
            * len(run_config["Q0_values_by_dataset"][dataset])
            for dataset in run_config["datasets"]
        )
        shards[path.parent.name]={"rows":len(frame),"status":summary["status"],
                                  "expected_configurations":expected_done,
                                  "config_status":summary["configuration_status_counts"]}
        if (summary["status"]!="done"
                or summary["configuration_status_counts"]!={"done":expected_done}):
            issues.append(f"incomplete shard {path.parent.name}")
    data=pd.concat(frames,ignore_index=True); data.to_csv(output/"nodes.csv",index=False)
    cfg=["dataset","seed","depth","method","mode","q","Q0","samples"]
    grouped=data.groupby(cfg).agg(rows=("target","size"),targets=("target","nunique"),
                                   cohorts=("target_cohort_checksum","nunique"),
                                   models=("model_cache_key","nunique"))
    bad_configs=int(((grouped.rows!=args.targets_per_config)|
                     (grouped.targets!=args.targets_per_config)|
                     (grouped.cohorts!=1)|(grouped.models!=1)).sum())
    keys=["dataset","seed","depth","mode","q","Q0","target","model_cache_key",
          "target_cohort_checksum"]
    pair=data.pivot(index=keys,columns="method",
                    values=["lp_lower_bound","outer_bounds_sha256","certified"]).dropna().reset_index()
    pair.columns=["_".join(value).strip("_") for value in pair.columns]
    delta=pair.lp_lower_bound_dual-pair.lp_lower_bound_baseline
    graph_cost_audit = cost_audit_checks(data)
    expected_grid = {
        (dataset, seed, depth, method, mode,
         dataset_spec(dataset).main_q, q0, 128)
        for dataset in ("Cora", "CiteSeer")
        for seed in range(5)
        for depth in (2, 3)
        for method in METHODS
        for mode in PAPER_MODES
        for q0 in dataset_spec(dataset).sensitivity_q0
    }
    actual_grid = {
        (str(row.dataset), int(row.seed), int(row.depth), str(row.method),
         str(row.mode), int(row.q), int(row.Q0), int(row.samples))
        for row in data[cfg].drop_duplicates().itertuples(index=False)
    }
    if len(paths) != 20:
        issues.append(f"expected 20 shards, found {len(paths)}")
    if actual_grid != expected_grid:
        issues.append(
            f"formal grid mismatch: missing={len(expected_grid-actual_grid)}, "
            f"extra={len(actual_grid-expected_grid)}"
        )
    if len(data) != 16_000:
        issues.append(f"expected 16000 rows, found {len(data)}")
    if len(pair) != 8_000:
        issues.append(f"expected 8000 baseline/dual pairs, found {len(pair)}")
    if set(data.solver_random_seed) != {1}:
        issues.append(
            f"expected deterministic CBC seed 1, found "
            f"{sorted(set(data.solver_random_seed))}"
        )
    checks={
        "status":"passed", "rows":len(data), "shard_count":len(paths),
        "config_count":len(grouped), "bad_configs":bad_configs,
        "expected_formal_grid_count":len(expected_grid),
        "actual_formal_grid_count":len(actual_grid),
        "solver_random_seed_values":sorted(int(value) for value in
                                             data.solver_random_seed.unique()),
        "solver_status_counts":{str(k):int(v) for k,v in data.solver_status.value_counts().items()},
        "solver_trusted_counts":{str(k):int(v) for k,v in data.solver_trusted.value_counts().items()},
        "soundness_error_nonempty":int(data.soundness_error.notna().sum()),
        "pair_count":len(pair),
        "outer_mismatch":int((pair.outer_bounds_sha256_baseline!=pair.outer_bounds_sha256_dual).sum()),
        "delta_min":float(delta.min()), "delta_max":float(delta.max()),
        "delta_below_minus_1e-6":int((delta < -1e-6).sum()),
        "dual_set_gains":int((pair.certified_dual>pair.certified_baseline).sum()),
        "dual_set_losses":int((pair.certified_dual<pair.certified_baseline).sum()),
        "graph_and_cost_audit": graph_cost_audit,
        "shards":shards, "issues":issues,
    }
    if (bad_configs or set(data.solver_status)!={"Optimal"} or set(data.solver_trusted)!={1}
            or checks["soundness_error_nonempty"] or checks["outer_mismatch"]
            or checks["delta_below_minus_1e-6"]):
        checks["issues"].append("global integrity condition failed")
    checks["status"]="passed" if not checks["issues"] else "failed"
    (output/"integrity.json").write_text(json.dumps(checks,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(checks,sort_keys=True))
    if checks["status"]!="passed": raise SystemExit(1)
    return checks


def parse(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--shards-root",required=True)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--targets-per-config",type=int,default=50)
    return parser.parse_args(argv)

if __name__=="__main__": run(parse())
