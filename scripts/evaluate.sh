#!/usr/bin/env bash
# Evaluate trained runs: every stage in the run's `stages` list of config/experiments.toml, in
# order.
# Usage: scripts/evaluate.sh TAG [TAG ...]
# Stages (all GPU except confirm_set and train_log):
#   scan         base policy and checkpoints on the scan pool  -> <tag>/scan/scan_<step>.json
#   heldout      held-out statistics at demonstration states   -> <tag>/heldout/
#   confirm_set  the confirm set, by the run's confirm rule    -> <tag>/confirm_set.json
#   confirm      the confirm set on the confirm pool           -> <tag>/confirm/confirm_<step>.json
#   softopc      collect the labeled pool, score every critic  -> <tag>/softopc/
#   train_log    training scalars from the offline W&B log     -> <tag>/train_log.json
# Finished steps are skipped, so the script can be rerun after an interruption.
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 TAG [TAG ...]" >&2
  exit 2
fi

cd "$REPO_ROOT"
for tag in "$@"; do
  stage_text="$(read_config stages "$tag")"
  stages=()
  while IFS= read -r stage; do
    stages+=("$stage")
  done <<< "$stage_text"

  for stage in "${stages[@]}"; do
    echo "$tag: $stage"
    case "$stage" in
      scan | confirm)
        "$PYTHON" -m ckpt_selection.evaluate --run "$tag" --pool "$stage" --outputs "$OUTPUTS"
        ;;
      heldout)
        "$PYTHON" -m ckpt_selection.heldout_stats --run "$tag" --outputs "$OUTPUTS"
        ;;
      confirm_set)
        "$PYTHON" -m ckpt_selection.confirm_set --run "$tag" --outputs "$OUTPUTS"
        ;;
      softopc)
        "$PYTHON" -m ckpt_selection.softopc_pool collect --run "$tag" --outputs "$OUTPUTS"
        "$PYTHON" -m ckpt_selection.softopc_pool score --run "$tag" --outputs "$OUTPUTS"
        ;;
      train_log)
        "$PYTHON" -m ckpt_selection.training_log --run "$tag" --outputs "$OUTPUTS"
        ;;
    esac
  done
done
