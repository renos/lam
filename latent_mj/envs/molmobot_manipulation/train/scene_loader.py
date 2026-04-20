"""Scene loading utilities for MolmoBot manipulation environments.

Two public functions:
  patch_robot_into_scene  – inject a robot MJCF via <include/> string-patch
  strip_warp_unsupported_options – zero out options mujoco_warp ignores
"""
import os
from pathlib import Path

import mujoco


def patch_robot_into_scene_at_pose(
    scene_xml_path: Path,
    robot_xml_path: Path,
    robot_prefix: str = "robot/",
    pos: list | None = None,
    quat: list | None = None,
) -> mujoco.MjSpec:
    """Like patch_robot_into_scene but places the robot at *pos* / *quat*.

    Args:
        pos:  [x, y, z] position for the robot base frame. Defaults to origin.
        quat: [w, x, y, z] quaternion for the robot base frame. Defaults to identity.

    Returns:
        Combined ``mujoco.MjSpec`` (not yet compiled).
    """
    scene_xml_path = Path(scene_xml_path)
    robot_xml_path = Path(robot_xml_path)

    scene_spec = mujoco.MjSpec.from_file(str(scene_xml_path))
    robot_spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    frame_kwargs: dict = {}
    if pos is not None:
        frame_kwargs["pos"] = pos
    if quat is not None:
        frame_kwargs["quat"] = quat

    parent_frame = scene_spec.worldbody.add_frame(**frame_kwargs)
    scene_spec.attach(robot_spec, prefix=robot_prefix, frame=parent_frame)

    return scene_spec


def patch_robot_into_scene(
    scene_xml_path: Path,
    robot_xml_path: Path,
    robot_prefix: str = "robot/",
) -> mujoco.MjSpec:
    """Return an MjSpec combining *scene_xml_path* and *robot_xml_path*.

    Uses ``MjSpec.attach`` (MJCF 3.x) so each model is parsed in its OWN
    directory context — meshdir resolves relative to each XML's location
    independently. This is the only sane way to combine MuJoCo MJCFs with
    differing ``<compiler meshdir="..."/>`` directives, which is the case for
    iThor scenes (no meshdir, scene-relative paths) + MolmoBot's franka_droid
    (``meshdir="assets"``).

    A naive ``<include file="..."/>`` flattens both into one compiler context
    and one of the meshdirs always loses — see commit history for the meshdir
    war we fought before switching to attach.

    Args:
        scene_xml_path: Absolute path to the scene MJCF (loaded as PARENT).
        robot_xml_path: Absolute path to the robot MJCF (loaded as CHILD,
            attached to scene's worldbody).
        robot_prefix: Prefix prepended to all robot body/joint/actuator names
            (default ``"robot/"``). Matches MolmoBot's namespace convention.

    Returns:
        Combined ``mujoco.MjSpec`` (NOT yet compiled — caller must call
        ``.compile()`` after any post-processing).
    """
    scene_xml_path = Path(scene_xml_path)
    robot_xml_path = Path(robot_xml_path)

    # Each spec carries its own meshdir context.
    scene_spec = mujoco.MjSpec.from_file(str(scene_xml_path))
    robot_spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    # MjSpec.attach requires a parent frame or site. Add a frame on the scene's
    # worldbody — caller can override its pos/quat to place the robot base by
    # editing scene_spec.worldbody.first_frame() (or equivalent) after attach.
    parent_frame = scene_spec.worldbody.add_frame()
    scene_spec.attach(robot_spec, prefix=robot_prefix, frame=parent_frame)

    return scene_spec


def strip_warp_unsupported_options(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove / zero option flags that mujoco_warp silently ignores or errors on.

    mujoco_warp (as of 3.7.0.1) does NOT support:
      - multiccd  (multi-convex-contact-detection) — controlled via enableflags
      - noslip_iterations > 0 — warp has no projected-Gauss-Seidel noslip solver

    Both are set in MolmoSpaces' base_scene.xml and must be cleared before
    calling ``mjx.put_model(..., impl="warp")``.

    The ``MjSpec.option`` object exposes:
      - ``enableflags``  (int bitmask, see mujoco.mjtEnableBit)
      - ``noslip_iterations`` (int)

    There is no ``spec.option.flag`` sub-object in mujoco 3.7.0; flags live
    directly on ``spec.option`` as integer bitmasks.

    Args:
        spec: An ``MjSpec`` (not yet compiled) whose options may contain
              warp-incompatible settings.

    Returns:
        The same *spec* object, mutated in place, for chaining convenience.
    """
    # mujoco_warp's _put_option only handles a subset of mjtEnableBit. Any
    # bit it doesn't recognize raises NotImplementedError. Clear the ones the
    # MolmoSpaces base scene sets that warp rejects:
    #   - mjENBL_MULTICCD (16) — multi-convex-contact-detection (CPU-only)
    #   - mjENBL_ENERGY   (2)  — energy computation (CPU-only)
    # Leave other enable bits (FWDINV, ISLAND, etc.) untouched in case future
    # scenes use ones warp does support.
    for bit in (mujoco.mjtEnableBit.mjENBL_MULTICCD, mujoco.mjtEnableBit.mjENBL_ENERGY):
        spec.option.enableflags &= ~int(bit)

    # Zero noslip iterations: warp has no noslip solver.
    spec.option.noslip_iterations = 0

    return spec
