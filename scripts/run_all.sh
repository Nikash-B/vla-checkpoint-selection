#!/usr/bin/env bash
# Run every experiment of the paper, then compute the results.
# Usage: scripts/run_all.sh
# Order: the task screen, training of every run in config/experiments.toml, evaluation of every
# run, then analyze.sh. screen.sh, train.sh and evaluate.sh need a GPU; analyze.sh runs on the CPU.
# Rerun: the GPU steps skip finished work, and analyze.sh recomputes every result.
# train.sh stops on a run whose training was interrupted. Move that run's <outputs>/<tag>/train/
# away before you rerun this script.
# Time on one RTX 3090 (App. A): a task-8 run takes about 6 h to train, 3 h to scan, 1 h for
# the held-out statistics and 2-3.5 h to confirm. The total grows with the number of runs and
# with the stages that each run lists in the config.
set -euo pipefail
source "$(dirname "$0")/common.sh"

scripts_dir="$REPO_ROOT/scripts"

tag_text="$(read_config tags)"
tags=()
while IFS= read -r tag; do
  if [ -n "$tag" ]; then
    tags+=("$tag")
  fi
done <<< "$tag_text"

bash "$scripts_dir/screen.sh"
bash "$scripts_dir/train.sh" "${tags[@]}"
bash "$scripts_dir/evaluate.sh" "${tags[@]}"
bash "$scripts_dir/analyze.sh"
