#!/usr/bin/env bash
# Base-policy task screen (GPU; App. A, Table 3): every stage of every suite in [screen].
# Usage: scripts/screen.sh [--suite SUITE] [--stage N]
# Writes <outputs>/screen/<suite>_stage<N>.json. Finished tasks are skipped on a rerun.
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$REPO_ROOT"
"$PYTHON" -m ckpt_selection.screen --outputs "$OUTPUTS" "$@"
