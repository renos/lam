"""JSON-episode loader for the MolmoBot pick benchmark.

Public API:
    load_benchmark(benchmark_dir) -> list[dict]
    episode_to_mj_model(episode, impl, install_missing_objects) -> (MjModel, metadata)
"""
import json
import os
from pathlib import Path
from typing import Any

import mujoco

from latent_mj.envs.molmobot_manipulation.train.scene_loader import (
    patch_robot_into_scene_at_pose,
    strip_warp_unsupported_options,
)


def _assets_dir() -> Path:
    val = os.environ.get("MLSPACES_ASSETS_DIR", "")
    if not val:
        raise EnvironmentError(
            "MLSPACES_ASSETS_DIR environment variable is not set. "
            "Point it to the mlspaces_assets directory, e.g.: "
            "export MLSPACES_ASSETS_DIR=/home/renos/lam/storage/mlspaces_assets"
        )
    return Path(val)


def load_benchmark(benchmark_dir: Path) -> list[dict]:
    """Read benchmark.json from *benchmark_dir* and return it as a Python list."""
    benchmark_dir = Path(benchmark_dir)
    # Accept either a directory containing benchmark.json or the file itself
    if benchmark_dir.is_dir():
        benchmark_file = benchmark_dir / "benchmark.json"
    else:
        benchmark_file = benchmark_dir
    with open(benchmark_file) as f:
        return json.load(f)


def episode_to_mj_model(
    episode: dict,
    impl: str = "warp",
    install_missing_objects: bool = True,
) -> tuple[mujoco.MjModel, dict[str, Any]]:
    """Build an MjModel from a benchmark episode dict.

    Returns (mj_model, metadata) where metadata contains at minimum:
      house_index, scene_dataset, scene_xml_path,
      robot_name, robot_xml_path, robot_prefix,
      init_qpos, pickup_obj_name, pickup_obj_start_pose,
      robot_base_pose, task_description, added_objects.
    """
    from molmo_spaces.molmo_spaces_constants import get_scenes, get_robot_path
    from molmo_spaces.utils.constants.simulation_constants import (
        OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
    )

    assets_dir = _assets_dir()

    # --- Resolve scene XML ---
    house_index = episode["house_index"]
    scene_dataset = episode["scene_dataset"]
    data_split = episode.get("data_split", "val")

    scenes = get_scenes(scene_dataset, data_split)
    # get_scenes returns {split: {int_index: path_or_None}}
    split_map: dict = scenes.get(data_split, scenes)
    scene_xml = split_map.get(house_index)
    if scene_xml is None:
        raise FileNotFoundError(
            f"Scene XML not found for {scene_dataset} house {house_index} "
            f"(split={data_split}). "
            f"Make sure MLSPACES_ASSETS_DIR contains the scenes."
        )
    # get_scenes may return a dict {'base': path, 'ceiling': path, 'map': ...}
    # for some datasets (e.g. procthor-10k); extract the 'base' XML in that case.
    if isinstance(scene_xml, dict):
        scene_xml = scene_xml.get("base") or next(
            v for v in scene_xml.values() if v is not None
        )
    scene_xml = Path(scene_xml)

    # --- Resolve robot XML ---
    robot_name = episode["robot"]["robot_name"]
    robot_xml = get_robot_path(robot_name) / "model.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(
            f"Robot XML not found: {robot_xml}. "
            f"Make sure the robot assets are installed under MLSPACES_ASSETS_DIR."
        )

    # --- Robot base pose (xyz + quat wxyz) ---
    robot_base_pose: list = episode["task"]["robot_base_pose"]
    robot_pos = robot_base_pose[0:3]
    robot_quat = robot_base_pose[3:7]  # already [w, x, y, z]

    # --- Patch robot into scene at the given base pose ---
    robot_prefix = "robot/"
    spec = patch_robot_into_scene_at_pose(
        scene_xml_path=scene_xml,
        robot_xml_path=robot_xml,
        robot_prefix=robot_prefix,
        pos=robot_pos,
        quat=robot_quat,
    )

    # --- Add auxiliary objects ---
    added_objects: dict = episode["scene_modifications"].get("added_objects", {})
    object_poses: dict = episode["scene_modifications"].get("object_poses", {})

    actual_added: dict = {}
    # Track how many times each XML file has been attached so we can give each
    # instance a unique suffix.  MuJoCo's attach_body prefixes/suffixes ALL
    # element names (including mesh assets), so a non-empty suffix prevents
    # duplicate-name errors when the same object XML appears more than once
    # (e.g. two instances of the same objaverse object in a scene).
    xml_attach_count: dict[str, int] = {}
    for object_name, object_xml_rel in added_objects.items():
        object_xml = assets_dir / object_xml_rel
        object_uid = Path(object_xml_rel).stem

        if not object_xml.is_file():
            if install_missing_objects:
                from molmo_spaces.utils.lazy_loading_utils import install_uid
                install_uid(object_uid)
            if not object_xml.is_file():
                raise FileNotFoundError(
                    f"Object asset not found and could not be installed: {object_xml}"
                )

        object_spec = mujoco.MjSpec.from_file(str(object_xml))
        if not object_spec.worldbody.bodies:
            raise ValueError(f"Object XML has no bodies: {object_xml}")
        obj_body: mujoco.MjsBody = object_spec.worldbody.bodies[0]

        # Normalize body name to match expected key
        name_parts = object_name.split("/")
        expected_body_name = name_parts[-1]
        if not obj_body.name or obj_body.name.strip() == "":
            obj_body.name = expected_body_name
        elif obj_body.name != expected_body_name:
            obj_body.name = expected_body_name

        # Add free joint if absent.  Use the body name (== object_name leaf) to
        # guarantee uniqueness across all attached objects.
        if not obj_body.first_joint():
            obj_body.add_joint(
                name=f"{expected_body_name}_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
            )

        # Pose for this object
        if object_name in object_poses:
            pose = object_poses[object_name]
            pos = pose[0:3]
            quat = pose[3:7]  # [w, x, y, z]
        else:
            pos = [0.0, 0.0, 0.0]
            quat = [1.0, 0.0, 0.0, 0.0]

        attach_frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        prefix = "/".join(name_parts[:-1]) + "/" if len(name_parts) > 1 else ""

        # Build a unique asset suffix so that duplicate XML attachments (two
        # instances of the same objaverse object) don't produce repeated mesh
        # names.  First attachment gets suffix "", subsequent ones get "_2",
        # "_3", … to stay backward-compatible with single-instance scenes.
        count = xml_attach_count.get(object_xml_rel, 0) + 1
        xml_attach_count[object_xml_rel] = count
        asset_suffix = "" if count == 1 else f"_{count}"

        attach_frame.attach_body(obj_body, prefix, asset_suffix)
        actual_added[object_name] = object_xml_rel

    # --- Strip warp-incompatible options and compile ---
    strip_warp_unsupported_options(spec)
    mj_model = spec.compile()

    # --- Build metadata ---
    init_qpos = episode["robot"].get("init_qpos", {})
    metadata: dict[str, Any] = {
        "house_index": house_index,
        "scene_dataset": scene_dataset,
        "scene_xml_path": scene_xml,
        "robot_name": robot_name,
        "robot_xml_path": robot_xml,
        "robot_prefix": robot_prefix,
        "init_qpos": {
            "arm": init_qpos.get("arm", []),
            "gripper": init_qpos.get("gripper", []),
        },
        "pickup_obj_name": episode["task"].get("pickup_obj_name", ""),
        "pickup_obj_start_pose": episode["task"].get("pickup_obj_start_pose", []),
        "robot_base_pose": robot_base_pose,
        "task_description": episode.get("language", {}).get("task_description", ""),
        "added_objects": actual_added,
    }

    return mj_model, metadata
