"""Load MolmoBot training-data shards from huggingface.co/datasets/allenai/molmobot-data.

Each shard is a tar of per-house tar.zst archives, each containing one or more
H5 files with multiple trajectories. Each trajectory has an `obs_scene` field
that is a JSON envelope with a base64-pickled SavedEpisode. This module
converts those into the same episode_dict format that
`MolmoBotPickEnv.from_episode` already accepts.
"""
import base64
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py


# Known data-gen config -> scene_dataset mapping.
# Source: molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py
DATAGEN_CONFIG_TO_SCENE_DATASET = {
    "FrankaPickOmniCamConfig": "procthor-10k",
    "FrankaPickAndPlaceOmniCamConfig": "procthor-10k",
    "FrankaPickAndPlaceColorOmniCamConfig": "procthor-10k",
    "FrankaPickAndPlaceNextToOmniCamConfig": "procthor-10k",
    "DoorOpeningDataGenConfig": "procthor-10k",
    "RBY1OpenDataGenConfig": "procthor-10k",
    "RBY1PickAndPlaceDataGenConfig": "procthor-10k",
    "RBY1PickDataGenConfig": "procthor-10k",
}


def _make_unpickler(stream):
    """Custom pickle.Unpickler that rewrites legacy mujoco_thor namespace
    to molmo_spaces (the package was renamed)."""
    import molmo_spaces  # noqa: F401  (ensure importable)
    sys.modules.setdefault("mujoco_thor", __import__("molmo_spaces"))

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("mujoco_thor"):
                module = module.replace("mujoco_thor", "molmo_spaces", 1)
            return super().find_class(module, name)

    return _Unpickler(stream)


def decode_obs_scene(obs_scene_bytes: bytes) -> tuple[dict, Any]:
    """Decode a trajectory's `obs_scene` blob.

    Returns:
        (json_envelope, saved_episode)
            json_envelope: top-level dict (task_type, task_description, ...)
            saved_episode: deserialised SavedEpisode pickle
    """
    txt = obs_scene_bytes.decode("utf-8")
    envelope = json.loads(txt)
    fc = base64.b64decode(envelope["frozen_config"])
    saved_episode = _make_unpickler(io.BytesIO(fc)).load()
    return envelope, saved_episode


def saved_episode_to_episode_dict(
    saved_episode,
    house_index: int,
    scene_dataset: str,
    data_split: str = "train",
    task_description: str | None = None,
) -> dict:
    """Convert a SavedEpisode + house metadata to the episode_dict schema
    consumed by MolmoBotPickEnv.from_episode()."""
    rc = saved_episode.robot_config
    tc = saved_episode.task_config

    # Split object_poses into:
    #   - objaverse objects (need to be added via added_objects + object_poses)
    #   - other objects (scene-baked iThor objects, just re-position via object_poses)
    #
    # Path convention from locate_uid_package / add_install_prefixes:
    #   ASSETS_DIR / "objects" / "objaverse" / uid / uid + ".xml"
    # Relative to ASSETS_DIR: "objects/objaverse/{uid}/{uid}.xml"
    added_objects: dict[str, str] = {}
    object_poses: dict[str, list] = dict(tc.object_poses)

    for obj_name in tc.object_poses:
        if obj_name.startswith("obja"):
            # Name convention: "obja{type}_{uid}_{instance}_{copy}_{idx}"
            # uid is the 32-char hex hash — second token after first '_'
            parts = obj_name.split("_")
            # Find the 32-char uid segment
            uid = None
            for part in parts[1:]:
                if len(part) == 32 and all(c in "0123456789abcdef" for c in part):
                    uid = part
                    break
            if uid is not None:
                added_objects[obj_name] = f"objects/objaverse/{uid}/{uid}.xml"

    return {
        "house_index": house_index,
        "scene_dataset": scene_dataset,
        "data_split": data_split,
        "robot": {
            "robot_name": rc.name,
            "init_qpos": dict(rc.init_qpos),
        },
        "scene_modifications": {
            "added_objects": added_objects,
            "object_poses": object_poses,
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            "robot_base_pose": list(tc.robot_base_pose),
            "pickup_obj_name": tc.pickup_obj_name,
            "pickup_obj_start_pose": list(tc.pickup_obj_start_pose),
            "pickup_obj_goal_pose": list(tc.pickup_obj_goal_pose),
        },
        "language": {
            "task_description": task_description or "(no description)",
            "referral_expressions": {},
        },
    }


def load_h5_trajectory(
    h5_path: Path,
    traj_key: str = "traj_0",
    datagen_config_name: str | None = None,
    scene_dataset_override: str | None = None,
) -> dict:
    """Load one trajectory from a molmobot-data H5 and return an episode_dict.

    Args:
        h5_path: path to trajectories_batch_*.h5
        traj_key: which trajectory inside the file (traj_0..traj_4)
        datagen_config_name: e.g. 'FrankaPickOmniCamConfig'. Used to look up
            scene_dataset via DATAGEN_CONFIG_TO_SCENE_DATASET.
        scene_dataset_override: if provided, use this instead of the registry lookup.
    """
    h5_path = Path(h5_path)

    # house_index from the directory containing the H5 (must look like 'house_<N>')
    house_dir = h5_path.parent.name
    if not house_dir.startswith("house_"):
        raise ValueError(f"Expected H5 to live under 'house_<N>/', got {h5_path}")
    house_index = int(house_dir.split("_", 1)[1])

    # data_split: assume 'train' unless the path contains 'val_shards' or 'val/'
    path_parts = {p.lower() for p in h5_path.parts}
    if "val_shards" in path_parts or "val" in path_parts:
        data_split = "val"
    elif "test_shards" in path_parts or "test" in path_parts:
        data_split = "test"
    else:
        data_split = "train"

    # scene_dataset
    if scene_dataset_override is not None:
        scene_dataset = scene_dataset_override
    elif datagen_config_name is not None:
        scene_dataset = DATAGEN_CONFIG_TO_SCENE_DATASET.get(
            datagen_config_name, "procthor-10k"
        )
    else:
        # try to infer from path by matching known config names
        scene_dataset = "procthor-10k"  # safe default
        for cfg_name, ds in DATAGEN_CONFIG_TO_SCENE_DATASET.items():
            if cfg_name in str(h5_path):
                scene_dataset = ds
                break

    with h5py.File(h5_path, "r") as f:
        if traj_key not in f:
            raise KeyError(f"{traj_key} not in {h5_path}; have: {list(f.keys())}")
        envelope, saved_episode = decode_obs_scene(f[traj_key]["obs_scene"][()])

    return saved_episode_to_episode_dict(
        saved_episode,
        house_index=house_index,
        scene_dataset=scene_dataset,
        data_split=data_split,
        task_description=envelope.get("task_description"),
    )
