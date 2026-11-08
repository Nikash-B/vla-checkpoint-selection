# Can the Critic Select Checkpoints for Vision-Language-Action Models?

## Overview

This repository holds the code for the paper. It post-trains π₀ on LIBERO with DSRL, evaluates
every saved checkpoint on two disjoint episode pools, and scores rollout-free checkpoint
selectors by their regret against the best confirmed checkpoint.

The folders are:

```
config/          experiments.toml, the settings of every run
scripts/         the run scripts
ckpt_selection/  the Python package that the scripts call
env/             the Dockerfile for the GPU machine
dsrl_pi0/        the DSRL code that the training uses, with our edits
```

## Installation

Training and evaluation need a Linux machine with one NVIDIA GPU, CUDA 12 and EGL rendering. On
one RTX 3090, a run on LIBERO-10 task 8 takes about 6 h to train, 3 h to scan, 1 h for the held-out statistics
and 2-3.5 h to confirm. The analysis runs on CPU.

**GPU machine, with Docker.** Build the image, start one container with the repository mounted,
and run the setup inside it:

```bash
docker build -t ckpt-selection env/
docker run --gpus all -it --name ckpt-selection -v "$PWD":/workspace -w /workspace ckpt-selection bash
bash scripts/setup.sh
```

Without Docker, run `bash scripts/setup.sh` on a Linux host with CUDA 12 and an NVIDIA driver
that provides EGL. The script creates `.venv/`, clones the pinned openpi and LIBERO forks,
downloads the LIBERO demonstration files and checks that LIBERO renders on the GPU.

**Analysis only (CPU).** To compute the results from existing run outputs, install the CPU
requirements:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Reproducing paper results

Run every command from the repository root. Every script writes under `outputs/`.

```bash
# Every experiment, then the analysis
bash scripts/run_all.sh
```

Each run in the config has one of four roles: `earlier`, `replication`, `control` and
`drop_screen`. Each run also has a tag, which is its folder name under `outputs/`. To list the
tags of one role, run:

```bash
.venv/bin/python -m ckpt_selection.config tags --role replication
```

To run the pipeline step by step, use one script per stage:

1. **Task screen.** This runs the base policy on every task of each suite in the config.

   ```bash
   bash scripts/screen.sh
   bash scripts/screen.sh --suite libero_90 --stage 2    # one stage of one suite
   ```

   - The output goes to `outputs/screen/`.

2. **Training.** This trains one or more runs with the code in `dsrl_pi0/`.

   ```bash
   bash scripts/train.sh task8_seed01
   ```

   - Each run saves a checkpoint at every step in `checkpoint_steps` of the config.
   - The output goes to `outputs/<tag>/train/`.

3. **Evaluation.** This runs the stages in the run's `stages` list of the config.

   ```bash
   bash scripts/evaluate.sh task8_seed01
   ```

   - The stages are `scan`, `heldout`, `confirm_set`, `confirm`, `softopc` and `train_log`.
     The header of `scripts/evaluate.sh` describes each stage.
   - A rerun skips the checkpoints and stages whose output files exist.
   - The output goes to `outputs/<tag>/`.

4. **Analysis (CPU).** This computes the results of every experiment from the run outputs.

   ```bash
   bash scripts/analyze.sh
   ```

   - Each file in `outputs/results/` holds one experiment. `results.json` holds the same
     numbers at full precision, keyed by experiment. Each per-checkpoint or per-step series
     inside is keyed by run tag, then gradient step.
   - The docstring of `ckpt_selection/layout.py` documents every output file.

## Tips

- `config/experiments.toml` is the only file with run settings. Its comments name the place in
  the paper that states each setting.
- `train.sh` refuses to start over an unfinished run. Move that run's `outputs/<tag>/train/`
  away first.
- GPU nondeterminism changes training, so a rerun gives different checkpoints and success rates.
- Run one GPU job per GPU at a time.
- To write elsewhere, set `OUTPUTS=/path/to/dir` for any script, or pass `--outputs DIR` to
  `screen.sh` and `analyze.sh`. The flag wins over `OUTPUTS`.
- The `ckpt_selection` modules that `evaluate.sh`, `screen.sh` and `analyze.sh` run print their
  flags with `--help`.

## Acknowledgments

This codebase is built on top of [DSRL](https://github.com/nakamotoo/dsrl_pi0). The `dsrl_pi0/`
folder is taken from DSRL with its license, and our edits are marked in the code.
