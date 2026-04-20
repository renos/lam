#!/usr/bin/env bash
# setup_cluster.sh — populate a fresh clone of lam on a SLURM cluster
# (e.g. GH200 Delta-AI) so `uv run python train.py --multirun hydra/launcher=slurm`
# works end-to-end.
#
# Recommended invocation (pins uv to the project venv even inside a conda
# shell, so we never accidentally install into the surrounding env):
#
#     uv run bash scripts/setup_cluster.sh
#
# Plain `bash scripts/setup_cluster.sh` also works; the CONDA_PREFIX unset
# below is a belt-and-suspenders fallback.
#
# Idempotent; re-running skips anything already in place.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== lam cluster setup @ $REPO_ROOT ==="

# Force uv to use the project-local .venv/ even if a conda env is active.
# Without this, uv sees CONDA_PREFIX and installs into the conda env.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE \
      CONDA_EXE CONDA_SHLVL VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"

# ---------------------------------------------------------------------------
# 1. uv + pyproject deps
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] uv sync (creates .venv/ and installs from uv.lock)"
uv sync

# ---------------------------------------------------------------------------
# 2. External clones — molmospaces + mujoco_menagerie
# ---------------------------------------------------------------------------
mkdir -p external
if [ ! -d external/molmospaces ]; then
    echo "[setup] cloning allenai/molmospaces into external/"
    git clone --depth 1 https://github.com/allenai/molmospaces.git external/molmospaces
else
    echo "[setup] external/molmospaces already present"
fi

if [ ! -d external/mujoco_menagerie ]; then
    echo "[setup] cloning google-deepmind/mujoco_menagerie into external/"
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git external/mujoco_menagerie
else
    echo "[setup] external/mujoco_menagerie already present"
fi

# ---------------------------------------------------------------------------
# 3. Install molmospaces editable (deps-less; we only need base_scene.xml and
#    the resource manager — the full molmospaces dep graph is heavy and not
#    required for the tracking env).
#
# Note: `uv run ...` implicitly re-syncs the venv from uv.lock, which would
# wipe this editable install.  Anywhere below we need molmo_spaces we use
# the venv's python directly (VENV_PY) to avoid the re-sync.
# ---------------------------------------------------------------------------
VENV_PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: $VENV_PY missing — uv sync failed to create the project venv."
    echo "       Check for a surrounding conda/virtualenv activation and retry."
    exit 1
fi
echo "[setup] installing molmospaces into .venv (editable, no deps)"
uv pip install --python "$VENV_PY" -e ./external/molmospaces --no-deps

# ---------------------------------------------------------------------------
# 4. franka_droid robot assets via molmospaces resource manager
# ---------------------------------------------------------------------------
FRANKA_XML="storage/mlspaces_assets/robots/franka_droid/model.xml"
if [ ! -f "$FRANKA_XML" ]; then
    echo "[setup] downloading franka_droid assets via molmospaces resource manager"
    "$VENV_PY" - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_resource_manager
m = get_resource_manager()
m.install_packages(["franka_droid"])
print("franka_droid install OK")
PY
else
    echo "[setup] $FRANKA_XML already present"
fi

# ---------------------------------------------------------------------------
# 5. molmobot-data sample H5 (reference trajectory pool)
# ---------------------------------------------------------------------------
# The training entry defaults to:
#   storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5
# Point --reference-h5 at whatever you have on the cluster.
#
# Automatic download from HF is non-trivial (shards are nested .tar / .tar.zst).
# Easiest: rsync from your dev box OR run the manual HF download block below.
SAMPLE_H5="storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5"
if [ ! -f "$SAMPLE_H5" ]; then
    echo "[setup] NOTE: $SAMPLE_H5 missing"
    echo "         Populate it via ONE of:"
    echo "         a) rsync from dev box:"
    echo "              rsync -av <dev>:/home/renos/lam/storage/molmobot_data_sample/ \\"
    echo "                         $REPO_ROOT/storage/molmobot_data_sample/"
    echo "         b) huggingface-cli download (one shard ≈ 2 GB):"
    echo "              uv run huggingface-cli download allenai/molmobot-data \\"
    echo "                FrankaPickOmniCamConfig/train_shards/00000.tar \\"
    echo "                --repo-type dataset \\"
    echo "                --local-dir storage/molmobot_data_sample/hf_raw"
    echo "              (then untar the outer, zstd-decode the inner house tar,"
    echo "               untar into storage/molmobot_data_sample/extracted/)"
else
    echo "[setup] $SAMPLE_H5 already present"
fi

# ---------------------------------------------------------------------------
# 6. iThor scenes (only needed if you train on base_scene with objects; the
#    default tracking env uses the scene shell so pull them just in case).
# ---------------------------------------------------------------------------
ITHOR_DIR="storage/mlspaces_assets/scenes/ithor"
if [ ! -d "$ITHOR_DIR" ]; then
    echo "[setup] downloading iThor scenes (~324 MB) via molmospaces resource manager"
    "$VENV_PY" - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_resource_manager
m = get_resource_manager()
m.install_packages(["ithor"])
print("ithor install OK")
PY
else
    echo "[setup] $ITHOR_DIR already present"
fi

echo ""
echo "=== setup complete ==="
echo "Smoke-check:"
echo "  uv run python train.py --cfg job                                    # validate config"
echo "  uv run python train.py --multirun hydra/launcher=slurm exp_name=v1  # submit"
