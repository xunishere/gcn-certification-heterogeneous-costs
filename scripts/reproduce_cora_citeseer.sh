#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PY=${PY:-.venv-risk/bin/python}
ARTIFACTS=${ARTIFACTS:-.pi-tasks/q1pct-relative-global-budget/artifacts/one-layer}
CACHE=${CACHE:-.cache/pyg}
MODEL_CACHE=${MODEL_CACHE:-.pi-tasks/canonical-graph-auditable-costs/artifacts/model-cache}
TIMEOUT=${CONFIG_TIMEOUT_SECONDS:-600}
PROTOCOL=q1pct-Qrelative-legacy-hetero
SMOKE_ROOT=$ARTIFACTS/smoke/$PROTOCOL
RAW_ROOT=$ARTIFACTS/raw/$PROTOCOL

if [[ ! -x "$PY" ]]; then
  echo "Python environment not found: $PY" >&2
  echo "Create it and install requirements-lock.txt first." >&2
  exit 2
fi

mkdir -p "$ARTIFACTS"/{logs,raw,smoke,analysis-legacy-hetero} \
  "$MODEL_CACHE" "$SMOKE_ROOT" "$RAW_ROOT"
COMMON=(--device cpu --epochs 500 --cache-dir "$CACHE" \
        --model-cache-dir "$MODEL_CACHE" --config-timeout-seconds "$TIMEOUT")

# Required gate: all regression/unit tests before any real run.
"$PY" -m unittest discover -v \
  2>&1 | tee "$ARTIFACTS/logs/tests-pre-smoke.log"
"$PY" -m compileall -q VRP

# Small smoke runs. Seed 0 model caches are reused by the formal runs below.
"$PY" -m VRP.real_experiments \
  --datasets Karate --seeds 0 \
  --modes legacy heterogeneous-cost \
  --q 1 --q0 1 --target-scope all "${COMMON[@]}" \
  --output-dir "$SMOKE_ROOT/karate" \
  2>&1 | tee "$ARTIFACTS/logs/karate-smoke.log"

"$PY" -m VRP.real_experiments \
  --datasets Cora --seeds 0 \
  --modes legacy heterogeneous-cost \
  --q 14 --q0 14 --target-scope stratified --sample-size 7 "${COMMON[@]}" \
  --output-dir "$SMOKE_ROOT/cora-small" \
  2>&1 | tee "$ARTIFACTS/logs/cora-smoke.log"

# Both spellings are accepted; this deliberately exercises the Citeseer alias.
"$PY" -m VRP.real_experiments \
  --datasets Citeseer --seeds 0 \
  --modes legacy heterogeneous-cost \
  --q 37 --q0 37 --target-scope stratified --sample-size 7 "${COMMON[@]}" \
  --output-dir "$SMOKE_ROOT/citeseer-small" \
  2>&1 | tee "$ARTIFACTS/logs/citeseer-smoke.log"

# Priority 1: CiteSeer formal standard test mask, 1,000 targets per seed.
"$PY" -m VRP.real_experiments \
  --datasets CiteSeer --seeds 0 1 2 3 4 \
  --modes legacy heterogeneous-cost \
  --q0-sensitivity --target-scope test "${COMMON[@]}" \
  --output-dir "$RAW_ROOT/citeseer-formal-test1000" \
  2>&1 | tee "$ARTIFACTS/logs/citeseer-formal.log"

# Priority 2: Cora rerun under the same Legacy/Hetero protocol.
"$PY" -m VRP.real_experiments \
  --datasets Cora --seeds 0 1 2 3 4 \
  --modes legacy heterogeneous-cost \
  --q0-sensitivity --target-scope test "${COMMON[@]}" \
  --output-dir "$RAW_ROOT/cora-formal-test1000" \
  2>&1 | tee "$ARTIFACTS/logs/cora-formal.log"

"$PY" -m VRP.analyze_experiments \
  --run "formal_test1000=$RAW_ROOT/cora-formal-test1000/nodes.csv" \
  --run "formal_test1000=$RAW_ROOT/citeseer-formal-test1000/nodes.csv" \
  --run-summary "Cora-formal=$RAW_ROOT/cora-formal-test1000/run_summary.json" \
  --run-summary "CiteSeer-formal=$RAW_ROOT/citeseer-formal-test1000/run_summary.json" \
  --output-dir "$ARTIFACTS/analysis-legacy-hetero" \
  2>&1 | tee "$ARTIFACTS/logs/formal-analysis.log"

# Lock reproducibility and final test gate.
"$PY" -m pip freeze --all > "$ARTIFACTS/requirements-freeze-current.txt"
"$PY" -m unittest discover -v \
  2>&1 | tee "$ARTIFACTS/logs/tests-final.log"
"$PY" -m compileall -q VRP

printf 'Report: %s\n' "$ARTIFACTS/analysis-legacy-hetero/REPORT.md"
printf 'Cora formal raw: %s\n' "$RAW_ROOT/cora-formal-test1000/nodes.csv"
printf 'CiteSeer formal raw: %s\n' "$RAW_ROOT/citeseer-formal-test1000/nodes.csv"
