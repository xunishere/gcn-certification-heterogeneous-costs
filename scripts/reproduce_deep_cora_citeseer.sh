#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv-risk/bin/python}
ARTIFACTS=${ARTIFACTS:-.pi-tasks/q1pct-relative-global-budget/artifacts/deep-deterministic-seed1}
CACHE=${CACHE:-.cache/pyg}
MODEL_CACHE=${MODEL_CACHE:-.pi-tasks/deep-canonical-graph-auditable-costs/artifacts/models/tuning-default}
FORMAL=${FORMAL:-$ARTIFACTS/raw/formal-q1pct-Qrelative-legacy-hetero-seed1}
ABLATION_ROOT=$ARTIFACTS/raw/legacy-hetero
VALIDATION_ROOT=$ARTIFACTS/validation/legacy-hetero
PARALLEL=${PARALLEL:-6}
if [[ ! -x "$PY" ]]; then echo "Python environment not found: $PY" >&2; exit 2; fi
mkdir -p "$ARTIFACTS"/{logs,models,raw,analysis-legacy-hetero,validation} \
  "$FORMAL/shards" "$MODEL_CACHE" "$ABLATION_ROOT" "$VALIDATION_ROOT"

# Repository and mandatory deep correctness gates.
"$PY" -m unittest discover -v 2>&1 | tee "$ARTIFACTS/logs/deep-tests-pre.log"
"$PY" -m compileall -q VRP

run_deep_shard() {
  local dataset="$1" seed="$2" depth="$3" phase="$4" limit="$5"
  local slug="${dataset}-s${seed}-d${depth}"
  local out="$FORMAL/shards/$slug"
  local limit_args=()
  [[ "$limit" == 0 ]] || limit_args=(--invocation-target-limit "$limit")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m VRP.deep_experiments \
    --datasets "$dataset" --seeds "$seed" --depths "$depth" --hidden-dims 16 --bias \
    --methods baseline dual --modes legacy heterogeneous-cost \
    --q0-sensitivity --target-scope test-stratified --sample-size 50 \
    "${limit_args[@]}" --samples 128 --input-abstraction budget-box --device cpu \
    --cache-dir "$CACHE" --model-cache-dir "$MODEL_CACHE" \
    --epochs 500 --learning-rate .01 --weight-decay .0005 --patience 100 \
    --solver-timeout 30 --solver-tolerance 1e-6 --config-timeout-seconds 3600 \
    --output-dir "$out" --gate-verbosity 0 > "$ARTIFACTS/logs/${phase}-${slug}.log" 2>&1
}
export -f run_deep_shard
export PY ARTIFACTS CACHE MODEL_CACHE FORMAL
JOBS="$FORMAL/formal-jobs.tsv"; : > "$JOBS"
for dataset in Cora CiteSeer; do for seed in 0 1 2 3 4; do for depth in 2 3; do
  printf '%s\t%s\t%s\n' "$dataset" "$seed" "$depth" >> "$JOBS"
done; done; done

# A fixed 50-target manifest is allocated first. The pilot processes its first
# 10 targets/config; the second invocation resumes the same files to all 50.
xargs -P "$PARALLEL" -n 3 bash -c 'run_deep_shard "$0" "$1" "$2" pilot 10' < "$JOBS"
xargs -P "$PARALLEL" -n 3 bash -c 'run_deep_shard "$0" "$1" "$2" formal 0' < "$JOBS"

"$PY" -m VRP.consolidate_deep_shards \
  --shards-root "$FORMAL/shards" --output-root "$FORMAL" --targets-per-config 50 \
  2>&1 | tee "$ARTIFACTS/logs/deep-consolidate.log"

# Samples ablation: seed 0, first 10 targets of the same 50-target cohorts,
# both datasets/depths, legacy q=floor(0.01*D) and the lambda=1 global Q.
# Main samples=128 comes from formal rows.
for samples in 32 256; do
  "$PY" -m VRP.deep_experiments \
    --datasets Cora CiteSeer --seeds 0 --depths 2 3 --hidden-dims 16 --bias \
    --methods baseline dual --modes legacy \
    --target-scope test-stratified --sample-size 50 --invocation-target-limit 10 \
    --samples "$samples" --input-abstraction budget-box --device cpu \
    --cache-dir "$CACHE" --model-cache-dir "$MODEL_CACHE" \
    --epochs 500 --learning-rate .01 --weight-decay .0005 --patience 100 \
    --solver-timeout 30 --solver-tolerance 1e-6 --config-timeout-seconds 3600 \
    --output-dir "$ABLATION_ROOT/ablation-s${samples}" --gate-verbosity 0 \
    > "$ARTIFACTS/logs/ablation-s${samples}.log" 2>&1
done

"$PY" -m VRP.validate_deep_real_controlled \
  --formal-root "$FORMAL" --model-cache-dir "$MODEL_CACHE" \
  --output-dir "$VALIDATION_ROOT" --cache-dir "$CACHE" \
  2>&1 | tee "$ARTIFACTS/logs/real-controlled-exact.log"

"$PY" -m VRP.analyze_deep_experiments \
  --formal-root "$FORMAL" \
  --ablation-nodes "$ABLATION_ROOT/ablation-s32/nodes.csv" "$ABLATION_ROOT/ablation-s256/nodes.csv" \
  --controlled-exact-summary "$VALIDATION_ROOT/real-controlled-exact-summary.json" \
  --output-dir "$ARTIFACTS/analysis-legacy-hetero" --cache-dir "$CACHE" \
  2>&1 | tee "$ARTIFACTS/logs/deep-analysis.log"

"$PY" -m unittest discover -v 2>&1 | tee "$ARTIFACTS/logs/deep-tests-final.log"
"$PY" -m compileall -q VRP
printf 'Deep report: %s\nRaw rows: %s\n' "$ARTIFACTS/analysis-legacy-hetero/REPORT.md" "$FORMAL/nodes.csv"
