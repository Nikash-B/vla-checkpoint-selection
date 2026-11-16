#!/usr/bin/env bash
# Install the GPU stack for training and evaluation into .venv/ (Linux, NVIDIA GPU, CUDA 12).
# Usage: bash scripts/setup.sh   (inside the env/Dockerfile container or on a CUDA 12 host)
# Steps: EGL system libraries; uv and Python 3.11.11; the openpi and LIBERO forks at the commits
# in [upstream] of config/experiments.toml; the pinned Python packages and the DSRL code in dsrl_pi0/;
# the CUDA 12 user-space wheels; the LIBERO demonstration files; a GPU render check.
# Rerun: every step skips work that is already done, so the script can be rerun after a failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO_ROOT/.venv-tools"
VENV="$REPO_ROOT/.venv"
UV="$TOOLS/bin/uv"
UV_VERSION="0.6.14"
PYTHON_VERSION="3.11.11"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_HTTP_TIMEOUT=180

# 1. EGL loader and GLES for headless LIBERO rendering.
egl_packages=(libegl1 libgles2)
if [ "$(id -u)" -eq 0 ]; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${egl_packages[@]}"
else
  echo "Not root: install the EGL libraries yourself if missing:"
  echo "  sudo apt-get install -y ${egl_packages[*]}"
fi

# 2. uv (a fast pip replacement) and the virtual environment with a pinned Python.
if [ ! -x "$UV" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" |
    env UV_INSTALL_DIR="$TOOLS/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv "$VENV" --python "$PYTHON_VERSION"
fi

# The shared settings need .venv/bin/python, so they load only after the venv exists.
# Setup always installs into .venv/, whatever PYTHON the caller set.
PYTHON="$VENV/bin/python"
source "$REPO_ROOT/scripts/common.sh"
cd "$REPO_ROOT"

# 3. The openpi and LIBERO forks that dsrl_pi0 pins, at the commits in [upstream].
# Submodule URLs of the form git@github.com: are fetched over HTTPS for this command only.
clone_at_commit() {
  local repo="$1" commit="$2" target="$3"
  if [ ! -d "$target/.git" ]; then
    git clone -q "$repo" "$target"
    git -C "$target" checkout -q "$commit"
  fi
  local have
  have="$(git -C "$target" rev-parse HEAD)"
  if [ "$have" != "$commit" ]; then
    echo "$target is at $have, but config/experiments.toml pins $commit" >&2
    exit 1
  fi
  git -C "$target" -c url."https://github.com/".insteadOf="git@github.com:" \
    submodule update -q --init --recursive
}
clone_at_commit "$(read_config get upstream.openpi_repo)" \
  "$(read_config get upstream.openpi_commit)" "$DSRL_DIR/openpi"
clone_at_commit "$(read_config get upstream.libero_repo)" \
  "$(read_config get upstream.libero_commit)" "$DSRL_DIR/LIBERO"

# 4. Python packages. dsrl_pi0/requirements.txt pins the GPU stack, including its own
# numpy and scipy, and the analysis runs on those versions too. requirements.txt
# (the CPU-only pins for analysis machines) is therefore not installed here, and this package is
# installed without its dependencies, so each library has one pin in this environment.
install() {
  "$UV" pip install --python "$PYTHON" -q "$@" ||
    { echo "retrying: uv pip install $*"; sleep 5; "$UV" pip install --python "$PYTHON" -q "$@"; }
}
install -e "$DSRL_DIR" -r "$DSRL_DIR/requirements.txt"
install "jax[cuda12]==0.5.0"
install -e "$DSRL_DIR/openpi" -e "$DSRL_DIR/openpi/packages/openpi-client"
# LIBERO needs the compat editable mode; the default editable install leaves `libero` unimportable.
install -e "$DSRL_DIR/LIBERO" --config-settings editable_mode=compat
install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # LIBERO imports torch
install mujoco==3.3.1   # upstream run_libero.sh upgrades mujoco over robosuite's pin
install --no-deps -e "$REPO_ROOT"

# 5. CUDA user-space repair. The resolver can pull CUDA 13 or newer CUDA 12 nvidia-* wheels, which
# shadow the cu12 ones and need newer drivers. Remove every nvidia-* wheel that is not cu12,
# then pin the cu12 wheels to CUDA 12.8, which also runs on newer drivers.
non_cu12="$(mktemp)"
"$UV" pip list --python "$PYTHON" 2>/dev/null |
  awk '/^nvidia-/ && $1 !~ /-cu12$/ {print $1}' > "$non_cu12"
if [ -s "$non_cu12" ]; then
  xargs "$UV" pip uninstall --python "$PYTHON" < "$non_cu12"
fi
rm -f "$non_cu12"
install \
  "nvidia-cublas-cu12==12.8.4.1" \
  "nvidia-cuda-cupti-cu12==12.8.90" \
  "nvidia-cuda-nvcc-cu12==12.8.93" \
  "nvidia-cuda-nvrtc-cu12==12.8.93" \
  "nvidia-cuda-runtime-cu12==12.8.90" \
  "nvidia-cudnn-cu12==9.8.0.87" \
  "nvidia-cufft-cu12==11.3.3.83" \
  "nvidia-cusolver-cu12==11.7.3.90" \
  "nvidia-cusparse-cu12==12.5.8.93" \
  "nvidia-nvjitlink-cu12==12.8.93"

# 6. LIBERO asks for a dataset folder on its first import. Answer "N" once to keep the defaults;
# it writes ~/.libero/config.yaml and does not ask again.
echo N | "$PYTHON" -c "import libero.libero" > /dev/null 2>&1 || true

# 7. The LIBERO demonstration files that the held-out statistics read (LIBERO-10 and LIBERO-90).
# Print the demonstration file of every run with the heldout stage that is missing.
missing_code='
from ckpt_selection import config, heldout_stats
settings = config.load_config()
for tag in config.run_tags(settings):
    entry = config.run_entry(settings, tag)
    if "heldout" in entry["stages"]:
        path, _ = heldout_stats.demo_file(entry["suite"], entry["task"])
        if not path.exists():
            print(path)
'
if [ -n "$("$PYTHON" -c "$missing_code")" ]; then
  "$PYTHON" "$DSRL_DIR/LIBERO/benchmark_scripts/download_libero_datasets.py" --datasets libero_100
fi
if [ -n "$("$PYTHON" -c "$missing_code")" ]; then
  echo "LIBERO demonstration files are still missing:" >&2
  "$PYTHON" -c "$missing_code" >&2
  exit 1
fi

# 8. Render one LIBERO frame on the GPU; stops if LIBERO renders in software.
"$PYTHON" -m ckpt_selection.agent
echo "Setup finished. Python: $PYTHON"
