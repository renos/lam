"""Path constants for MolmoBot manipulation environments.

Two asset roots:
  * external/molmospaces — source clone, gives us base_scene.xml
  * storage/mlspaces_assets/robots/franka_droid — MolmoSpaces' real
    franka_droid composite (Franka FR3 + Robotiq 2F-85 v4), pulled by
    molmo_spaces.molmo_spaces_constants.get_resource_manager() via R2.

Fallback: external/mujoco_menagerie ships the raw fr3.xml + grippers
in case the MolmoSpaces resource pack isn't installed.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../lam/

# MolmoSpaces source clone (provides base_scene.xml + Python task code)
MOLMOSPACES_ROOT = REPO_ROOT / "external" / "molmospaces"
MOLMOSPACES_BASE_SCENE_XML = (
    MOLMOSPACES_ROOT / "molmo_spaces" / "resources" / "base_scene.xml"
)

# MolmoSpaces resource pack (downloaded assets — must be installed via
# molmo_spaces_constants.get_resource_manager().install_all_for_source).
MLSPACES_ASSETS_DIR = Path(
    os.environ.get("MLSPACES_ASSETS_DIR", REPO_ROOT / "storage" / "mlspaces_assets")
)
FRANKA_DROID_DIR = MLSPACES_ASSETS_DIR / "robots" / "franka_droid"
FRANKA_DROID_XML = FRANKA_DROID_DIR / "model.xml"  # Franka FR3 + Robotiq 2F-85 v4

# Raw menagerie fallbacks (if MolmoSpaces resource pack isn't installed)
MENAGERIE_ROOT = REPO_ROOT / "external" / "mujoco_menagerie"
FRANKA_FR3_XML = MENAGERIE_ROOT / "franka_fr3" / "fr3.xml"
FRANKA_FR3_SCENE_XML = MENAGERIE_ROOT / "franka_fr3" / "scene.xml"

# Backend selection: "warp" uses NVIDIA mujoco_warp; "jax" uses standard mjx.
DEFAULT_IMPL = "warp"
