#!/usr/bin/env bash
# Train runs of config/experiments.toml with the DSRL code in dsrl_pi0/ (GPU).
# Usage: scripts/train.sh TAG [TAG ...]
# Writes <outputs>/<tag>/train/ (checkpoints, variant.json, practice_episodes.jsonl, W&B log)
# and <outputs>/<tag>/train.log. A run whose final checkpoint exists is skipped.
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 TAG [TAG ...]" >&2
  exit 2
fi

# Print the run's training folder, final checkpoint folder and training log, one per line.
# ckpt_selection.layout builds the paths, so the shell never spells them out.
paths_code='
import sys
from pathlib import Path
from ckpt_selection import config, layout
tag, outputs = sys.argv[1], Path(sys.argv[2])
final_step = max(config.checkpoint_steps(config.load_config(), tag))
print(layout.train_dir(outputs, tag))
print(layout.checkpoint_dir(outputs, tag, final_step))
print(layout.run_dir(outputs, tag) / "train.log")
'

cd "$REPO_ROOT"
for tag in "$@"; do
  flag_text="$(read_config train-flags "$tag" --outputs "$OUTPUTS")"
  flags=()
  while IFS= read -r flag; do
    flags+=("$flag")
  done <<< "$flag_text"

  path_text="$("$PYTHON" -c "$paths_code" "$tag" "$OUTPUTS")"
  {
    IFS= read -r train_dir
    IFS= read -r final_checkpoint
    IFS= read -r log
  } <<< "$path_text"

  if [ -e "$final_checkpoint" ]; then
    echo "$tag: already trained ($final_checkpoint exists)"
    continue
  fi
  if [ -e "$train_dir" ]; then
    echo "$tag: $train_dir holds an unfinished run; move it away to train again" >&2
    exit 1
  fi
  # Only the log's folder is made here. The training code makes train/ itself, so a start that
  # fails early leaves no train/ folder behind.
  mkdir -p "$(dirname "$log")"
  echo "$tag: training, log in $log"
  if ! "$PYTHON" "$DSRL_DIR/examples/launch_train_sim.py" "${flags[@]}" > "$log" 2>&1; then
    echo "$tag: training failed, see $log" >&2
    exit 1
  fi
  echo "$tag: trained"
done
