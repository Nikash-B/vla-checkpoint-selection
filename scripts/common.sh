# Shared settings, sourced by every script in scripts/.
# Set PYTHON in the environment to use another interpreter. The scripts read OUTPUTS (the outputs
# folder); screen.sh and analyze.sh also take --outputs DIR, which wins over OUTPUTS.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# The DSRL code in dsrl_pi0/ (README, "Acknowledgments") and this package import from their roots.
DSRL_DIR="$REPO_ROOT/dsrl_pi0"
export PYTHONPATH="$DSRL_DIR:$REPO_ROOT"

# Read config/experiments.toml (see python -m ckpt_selection.config --help).
read_config() {
  "$PYTHON" -m ckpt_selection.config "$@"
}

# A relative OUTPUTS is relative to the caller's folder (the scripts later cd to REPO_ROOT).
# The default comes from config.outputs_root, so the shell and Python agree on it.
if [ -n "${OUTPUTS:-}" ]; then
  case "$OUTPUTS" in
    /*) ;;
    *) OUTPUTS="$PWD/$OUTPUTS" ;;
  esac
else
  outputs_code="from ckpt_selection import config; print(config.outputs_root(config.load_config()))"
  OUTPUTS="$("$PYTHON" -c "$outputs_code")"
fi

# Headless GPU rendering for LIBERO, lazy JAX GPU memory (pi0 and the agent share one GPU),
# the W&B log kept on disk, and the pi0 checkpoint cache inside the repo.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/.cache/openpi}"

# On the RTX 2080 Ti (compute capability 7.5), XLA's custom-kernel-fusion-rewriter pass replaces
# pi0's float32 x bfloat16 matmuls with a CUTLASS kernel that returns wrong values, and pi0 fails
# its tasks. We disable the pass when any GPU has compute capability 7.5 and keep XLA's default
# otherwise. Without nvidia-smi (CPU analysis), nothing changes.
gpu_capabilities="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
if printf "%s\n" "$gpu_capabilities" | grep -qx "7.5"; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
