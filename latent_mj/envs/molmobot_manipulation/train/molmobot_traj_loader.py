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


def _decode_action(raw_row: np.ndarray) -> np.ndarray:
    """Decode one uint8 null-padded JSON row → 8-dim float32 action.

    JSON schema: {"arm": [7 floats], "gripper": [1 float]}
    Result: arm (7) + gripper (1) = 8 dims, matching the franka_droid
    actuator order. Empty `{}` rows fall back to NaN so downstream code can
    handle "no command this step" (data-gen sometimes emits sparse cmds).
    """
    blob = bytes(raw_row).rstrip(b"\x00")
    if not blob:
        return np.full(8, np.nan, dtype=np.float32)
    d = json.loads(blob.decode())
    if not d:
        return np.full(8, np.nan, dtype=np.float32)
    vec = d.get("arm", []) + d.get("gripper", [])
    if len(vec) != 8:
        return np.full(8, np.nan, dtype=np.float32)
    return np.array(vec, dtype=np.float32)


def _load_single_traj(
    grp: h5py.Group,
    trim: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode qpos, qvel, and recorded actions for one trajectory group.

    Args:
        grp: HDF5 group for a single trajectory (e.g. f['traj_0']).
        trim: If True, trim to first timestep where terminated|truncated is True.

    Returns:
        qpos:    (T, 9) float32 — observed joint positions per step
        qvel:    (T, 9) float32 — observed joint velocities per step
        actions: (T, 8) float32 — recorded ctrl commands per step (NaN
                 where no command was issued; forward-fill before use)
        T: number of timesteps kept
    """
    raw_qpos = np.array(grp["obs/agent/qpos"])      # (T_full, 2000) uint8
    raw_qvel = np.array(grp["obs/agent/qvel"])      # (T_full, 2000) uint8
    raw_act = np.array(grp["actions/joint_pos"])    # (T_full, 2000) uint8
    T_full = raw_qpos.shape[0]

    if trim:
        terminated = np.array(grp["terminated"], dtype=bool)
        truncated = np.array(grp["truncated"], dtype=bool)
        done = terminated | truncated
        first_done = np.where(done)[0]
        T = int(first_done[0]) + 1 if len(first_done) > 0 else T_full
    else:
        T = T_full

    qpos = np.stack([_decode_qpos(raw_qpos[t]) for t in range(T)], axis=0)
    qvel = np.stack([_decode_qvel(raw_qvel[t]) for t in range(T)], axis=0)
    actions = np.stack([_decode_action(raw_act[t]) for t in range(T)], axis=0)

    # Forward-fill NaN action rows with the previous step's command (sparse
    # cmd convention in data-gen).  Bias the first row to the next non-NaN.
    if np.isnan(actions).any():
        valid = ~np.isnan(actions).any(axis=1)
        if not valid.any():
            actions[:] = 0.0
        else:
            first_valid = int(np.argmax(valid))
            actions[:first_valid] = actions[first_valid]
            for i in range(first_valid + 1, T):
                if not valid[i]:
                    actions[i] = actions[i - 1]

    return qpos, qvel, actions, T


def load_h5_reference_dataset(
    h5_path: str | Path,
    traj_keys: Optional[List[str]] = None,
    control_dt: float = 0.02,
    joint_names: Optional[List[str]] = None,
    trim: bool = True,
    glob_pattern: str = "**/*.h5",
    max_h5_files: Optional[int] = None,
) -> Trajectory:
    """Load one or many molmobot H5 files into a single Trajectory.

    Args:
        h5_path: Path to a single H5 file OR a directory. If a directory, we
            recursively glob for files matching ``glob_pattern`` and concat
            every ``traj_K`` group across all files.
        traj_keys: Single-file only — which traj groups to load. Ignored for
            directory inputs (we always take every ``traj_*`` group).
        control_dt: Control timestep in seconds.
        joint_names: 9 joint name strings (default: panda_joint1..7 +
            panda_finger_joint{1,2}).
        trim: Trim each trajectory at first terminated|truncated+1.
        glob_pattern: Pattern for directory-mode file discovery.
        max_h5_files: If set, stop after this many files (handy for debug /
            curriculum-sized runs without rewriting the reference pool layout).

    Returns:
        Trajectory with:
          data.qpos:         (sum_k T_k, 9) float32 — stacked across ALL trajs
          data.qvel:         (sum_k T_k, 9) float32
          data.split_points: (n_trajs+1,) int64
          info.metadata:     {
              'source_h5':       str(path or dir),
              'source_h5_files': [path, ...],  # one entry in file mode
              'traj_keys':       [(file_idx, key), ...],
              'recorded_actions': (sum_k T_k, 8) numpy float32,
          }
    """
    h5_path = Path(h5_path)

    # Discover H5 files (single-file or directory mode).
    if h5_path.is_dir():
        h5_files = sorted(h5_path.glob(glob_pattern))
        if max_h5_files is not None:
            h5_files = h5_files[:max_h5_files]
        assert h5_files, f"no .h5 files found under {h5_path} matching {glob_pattern}"
        print(f"load_h5_reference_dataset: discovered {len(h5_files)} H5 files "
              f"under {h5_path}")
    else:
        h5_files = [h5_path]

    all_qpos: List[np.ndarray] = []
    all_qvel: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    traj_lengths: List[int] = []
    source_traj_keys: List[tuple[int, str]] = []

    for file_idx, f_path in enumerate(h5_files):
        with h5py.File(f_path, "r") as f:
            all_keys = sorted(f.keys())
            if h5_path.is_dir() or traj_keys is None:
                keys_to_load = [k for k in all_keys if k.startswith("traj_")]
            else:
                keys_to_load = sorted(traj_keys)

            for key in keys_to_load:
                qpos, qvel, actions, T = _load_single_traj(f[key], trim=trim)
                all_qpos.append(qpos)
                all_qvel.append(qvel)
                all_act.append(actions)
                traj_lengths.append(T)
                source_traj_keys.append((file_idx, key))

    print(f"load_h5_reference_dataset: {len(traj_lengths)} trajectories, "
          f"{sum(traj_lengths)} total timesteps")

    # Stack across all trajectories: (sum_k T_k, 9)
    qpos_all = np.concatenate(all_qpos, axis=0).astype(np.float32)
    qvel_all = np.concatenate(all_qvel, axis=0).astype(np.float32)
    actions_all = np.concatenate(all_act, axis=0).astype(np.float32)

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
            "source_h5_files": [str(p) for p in h5_files],
            "traj_keys": source_traj_keys,
            # 8-dim recorded ctrl per step (arm 7 + gripper 1), stacked the
            # same way as data.qpos/qvel; aligned to data.split_points.
            # Stash as numpy (not jax) since this is metadata, not part of
            # the staged TrajectoryData payload.
            "recorded_actions": actions_all,
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
