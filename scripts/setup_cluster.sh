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

# Tell the molmospaces resource manager to symlink into the path that
# latent_mj.envs.molmobot_manipulation.constants expects (storage/mlspaces_assets/).
# The same export needs to be in the SLURM job's setup if you ever override
# this default — otherwise the env defaults match.
export MLSPACES_ASSETS_DIR="$REPO_ROOT/storage/mlspaces_assets"
mkdir -p "$MLSPACES_ASSETS_DIR"

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

# Curated subset of molmospaces's transitive deps — enough for the resource
# manager (R2 download + decompress) and for `import molmo_spaces`. We skip
# the conflicting heavyweights (jax~=0.6.2, mujoco-mjx==3.4.0, torch,
# tensorflow, open_clip_torch) because lam pins newer/different versions.
echo "[setup] installing molmospaces resource-manager transitive deps"
uv pip install --python "$VENV_PY" \
    compress-json boto3 zstandard requests tqdm \
    omegaconf 'pydantic>=2' lxml attrs pyyaml pandas \
    shapely msgpack msgpack-numpy future stringcase shortuuid \
    nltk pynvml transforms3d numpy-quaternion jaxlie prior \
    molmospaces-resources

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
m.install_all_for_source(data_type="robots", source="franka_droid")
print("franka_droid install OK")
PY
else
    echo "[setup] $FRANKA_XML already present"
fi

# ---------------------------------------------------------------------------
# 5. molmobot-data sample H5 (reference trajectory pool)
# ---------------------------------------------------------------------------
# Downloads one HF shard (~2 GB), extracts only the house_0 .tar.zst member,
# zstd-decodes, untars into storage/molmobot_data_sample/extracted/house_0/.
# All in one Python block — no manual rsync needed.
# Downloads one HF shard (~2 GB) and extracts EVERY per-house tar.zst into
# storage/molmobot_data_sample/extracted/house_<N>/. Set MOLMOBOT_MAX_HOUSES to
# limit (e.g. 5) for a quick test; unset or 0 = unlimited. All in one Python
# block — no manual rsync needed.
EXTRACTED_DIR="storage/molmobot_data_sample/extracted"
H5_COUNT=$(find "$EXTRACTED_DIR" -maxdepth 2 -name "*.h5" 2>/dev/null | wc -l)
if [ "$H5_COUNT" -lt "2" ]; then
    echo "[setup] downloading molmobot-data shard from HuggingFace (~2 GB) and extracting all houses"
    MOLMOBOT_MAX_HOUSES="${MOLMOBOT_MAX_HOUSES:-0}" "$VENV_PY" - <<'PY'
import io, os, tarfile
from pathlib import Path
import zstandard as zstd
from huggingface_hub import hf_hub_download

REPO = "allenai/molmobot-data"
SHARD = "FrankaPickOmniCamConfig/train_shards/00000.tar"
EXTRACT_DIR = Path("storage/molmobot_data_sample/extracted")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

max_houses = int(os.environ.get("MOLMOBOT_MAX_HOUSES", "0"))

print(f"  hf_hub_download {REPO}:{SHARD}")
shard_path = hf_hub_download(repo_id=REPO, filename=SHARD, repo_type="dataset")

with tarfile.open(shard_path, "r") as outer:
    house_members = sorted(
        (m for m in outer.getmembers() if m.name.startswith("FrankaPickOmniCamConfig_house_")
         and m.name.endswith(".tar.zst")),
        key=lambda m: int(m.name.split("_house_")[1].split(".")[0]),
    )
    if max_houses > 0:
        house_members = house_members[:max_houses]
    print(f"  extracting {len(house_members)} per-house tar.zst archives")

    dctx = zstd.ZstdDecompressor()
    for i, m in enumerate(house_members):
        with outer.extractfile(m) as zst_stream:
            zst_bytes = zst_stream.read()
        with dctx.stream_reader(io.BytesIO(zst_bytes)) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as inner:
                inner.extractall(EXTRACT_DIR)
        if (i + 1) % 10 == 0 or (i + 1) == len(house_members):
            print(f"    {i + 1}/{len(house_members)} houses extracted")

n_h5 = sum(1 for _ in EXTRACT_DIR.rglob("*.h5"))
print(f"  done. {n_h5} H5 files under {EXTRACT_DIR}")
PY
else
    echo "[setup] already extracted ($H5_COUNT H5 files under $EXTRACTED_DIR)"
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
m.install_all_for_source(data_type="scenes", source="ithor")
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
