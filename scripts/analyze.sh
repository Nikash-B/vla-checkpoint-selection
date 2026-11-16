#!/usr/bin/env bash
# Compute the results of every experiment from the run outputs (CPU only).
# Usage: scripts/analyze.sh [--outputs DIR]
# The outputs folder is --outputs DIR if given, else $OUTPUTS, else paths.outputs in the config.
# A relative DIR is read from the repository root; a relative OUTPUTS from the current folder.
# Writes <outputs>/results/<experiment>.md and <outputs>/results/results.json, and prints each
# Markdown file.
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$REPO_ROOT"
"$PYTHON" -m ckpt_selection.results --outputs "$OUTPUTS" "$@"
