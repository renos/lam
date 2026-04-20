"""
molmobot_traj_loader.py

Loads molmobot-data H5 files into the Trajectory / TrajectoryData /
TrajectoryInfo / TrajectoryModel types used by the rest of latent_mj.

Arm joints (7) are HINGE (mjJNT_HINGE = 3).
Finger joints (2) are SLIDE (mjJNT_SLIDE = 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import jax.numpy as jnp
import mujoco

from latent_mj.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)

# mjJNT_HINGE = 3  (verified via mujoco.mjtJoint)
_JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)

_DEFAULT_JOINT_NAMES: List[str] = [
    f"panda_joint{i}" for i in range(1, 8)
] + ["panda_finger_joint1", "panda_finger_joint2"]

# jnt_type vector: the franka_droid model the tracking env compiles is
# 13 HINGE joints (7 FR3 + 6 Robotiq driver/spring/follower). For
# arm-only tracking the consumer only reads qpos[:, :7] so this field is
# informational; we default to all-HINGE to avoid misleading callers
# that walk jnt_type. Loader callers that need an exact match against a
# different model layout should pass ``joint_names`` explicitly.
_DEFAULT_JNT_TYPE = np.array([_JNT_HINGE] * 9, dtype=np.int32)


def _decode_qpos(raw_row: np.ndarray) -> np.ndarray:
    """Decode one uint8 null-padded JSON row → 9-dim float32 qpos.

    JSON schema: {"arm": [7 floats], "base": [], "gripper": [2 floats]}
    Result: arm (7) + base (0, always empty) + gripper (2) = 9 dims.
    """
    blob = bytes(raw_row).rstrip(b"\x00")
    d = json.loads(blob.decode())
    vec = d["arm"] + d.get("base", []) + d["gripper"]
    return np.array(vec, dtype=np.float32)


def _decode_qvel(raw_row: np.ndarray) -> np.ndarray:
    """Decode one uint8 null-padded JSON row → 9-dim float32 qvel.

    Same schema as qpos.
    """
    blob = bytes(raw_row).rstrip(b"\x00")
    d = json.loads(blob.decode())
    vec = d["arm"] + d.get("base", []) + d["gripper"]
    return np.array(vec, dtype=np.float32)


def _load_single_traj(
    grp: h5py.Group,
    trim: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode qpos and qvel arrays for one trajectory group.

    Args:
        grp: HDF5 group for a single trajectory (e.g. f['traj_0']).
        trim: If True, trim to first timestep where terminated|truncated is True.

    Returns:
        qpos: (T, 9) float32
        qvel: (T, 9) float32
        T: number of timesteps kept
    """
    raw_qpos = np.array(grp["obs/agent/qpos"])  # (T_full, 2000) uint8
    raw_qvel = np.array(grp["obs/agent/qvel"])  # (T_full, 2000) uint8
    T_full = raw_qpos.shape[0]

    if trim:
        terminated = np.array(grp["terminated"], dtype=bool)
        truncated = np.array(grp["truncated"], dtype=bool)
        done = terminated | truncated
        first_done = np.where(done)[0]
        T = int(first_done[0]) + 1 if len(first_done) > 0 else T_full
    else:
        T = T_full

    qpos_list = [_decode_qpos(raw_qpos[t]) for t in range(T)]
    qvel_list = [_decode_qvel(raw_qvel[t]) for t in range(T)]

    qpos = np.stack(qpos_list, axis=0)  # (T, 9)
    qvel = np.stack(qvel_list, axis=0)  # (T, 9)
    return qpos, qvel, T


def load_h5_reference_dataset(
    h5_path: str | Path,
    traj_keys: Optional[List[str]] = None,
    control_dt: float = 0.02,
    joint_names: Optional[List[str]] = None,
    trim: bool = True,
) -> Trajectory:
    """Load a molmobot H5 file into a Trajectory.

    Args:
        h5_path: Path to the H5 file.
        traj_keys: Which traj groups to load (default: all keys starting with
            'traj_', in sorted order).
        control_dt: Control timestep in seconds. Sets info.frequency = 1/control_dt.
        joint_names: 9 joint name strings. If None, uses the default Panda names
            ["panda_joint1".."panda_joint7", "panda_finger_joint1",
            "panda_finger_joint2"]. The caller may need to remap to
            'robot/'-prefixed names via TrajectoryHandler.filter_and_extend.
        trim: Whether to trim each trajectory at the first terminated|truncated
            timestep (+1). Default True.

    Returns:
        Trajectory with:
          data.qpos:         (sum_k T_k, 9) float32
          data.qvel:         (sum_k T_k, 9) float32
          data.split_points: (n_trajs+1,) int64
          info.joint_names:  9 joint name strings
          info.model.jnt_type: (9,) int32 array
          info.frequency:    1 / control_dt
          info.body_names, info.site_names: None
          info.metadata:     {'source_h5': str(path), 'traj_keys': [...]}
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        all_keys = sorted(f.keys())
        if traj_keys is None:
            traj_keys = [k for k in all_keys if k.startswith("traj_")]
        traj_keys = sorted(traj_keys)

        all_qpos: List[np.ndarray] = []
        all_qvel: List[np.ndarray] = []
        traj_lengths: List[int] = []

        for key in traj_keys:
            grp = f[key]
            qpos, qvel, T = _load_single_traj(grp, trim=trim)
            all_qpos.append(qpos)
            all_qvel.append(qvel)
            traj_lengths.append(T)

    # Stack across all trajectories: (sum_k T_k, 9)
    qpos_all = np.concatenate(all_qpos, axis=0).astype(np.float32)
    qvel_all = np.concatenate(all_qvel, axis=0).astype(np.float32)

    # split_points: cumulative starts, length n_trajs+1
    split_points = np.concatenate(
        [[0], np.cumsum(traj_lengths)]
    ).astype(np.int64)

    # Build TrajectoryData with empty optional arrays
    data = TrajectoryData(
        qpos=jnp.array(qpos_all),
        qvel=jnp.array(qvel_all),
        xpos=jnp.empty(0),
        xquat=jnp.empty(0),
        cvel=jnp.empty(0),
        subtree_com=jnp.empty(0),
        site_xpos=jnp.empty(0),
        site_xmat=jnp.empty(0),
        split_points=jnp.array(split_points),
    )

    # Build TrajectoryModel: only joint info, no bodies/sites
    jnt_type = jnp.array(_DEFAULT_JNT_TYPE, dtype=jnp.int32)
    model = TrajectoryModel(
        njnt=9,
        jnt_type=jnt_type,
        nbody=None,
        nsite=None,
    )

    # Build TrajectoryInfo
    if joint_names is None:
        joint_names = list(_DEFAULT_JOINT_NAMES)

    info = TrajectoryInfo(
        joint_names=joint_names,
        model=model,
        frequency=1.0 / control_dt,
        body_names=None,
        site_names=None,
        metadata={
            "source_h5": str(h5_path),
            "traj_keys": list(traj_keys),
        },
    )

    return Trajectory(info=info, data=data)


if __name__ == "__main__":
    import os

    # Set required env var if missing (only for standalone testing)
    if "GLI_PATH" not in os.environ:
        os.environ["GLI_PATH"] = "/tmp"

    H5_PATH = (
        "/home/renos/lam/storage/molmobot_data_sample/extracted/"
        "house_0/trajectories_batch_4_of_20.h5"
    )

    print(f"Loading: {H5_PATH}")
    traj = load_h5_reference_dataset(H5_PATH)

    n_trajs = traj.data.n_trajectories
    total_steps = int(traj.data.split_points[-1])
    print(f"n_trajectories: {n_trajs}")
    print(f"total_timesteps: {total_steps}")
    print(f"qpos.shape:      {traj.data.qpos.shape}")
    print(f"qvel.shape:      {traj.data.qvel.shape}")
    print(f"split_points:    {traj.data.split_points}")
    print(f"info.joint_names: {traj.info.joint_names}")
    print(f"info.model.jnt_type: {traj.info.model.jnt_type}")
    print(f"info.frequency:  {traj.info.frequency} Hz")
