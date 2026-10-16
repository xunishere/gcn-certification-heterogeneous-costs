#!/usr/bin/env bash
# Backward-compatible entry point. The formal workflow now covers both Cora
# and CiteSeer under the Legacy/Hetero comparison protocol.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/reproduce_cora_citeseer.sh" "$@"
