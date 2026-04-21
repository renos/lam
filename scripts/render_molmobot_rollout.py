"""Render a MolmoBotTrackingEnv rollout to mp4 + report per-step tracking error.

Two modes:

  --mode residual-zero   (default)
      Step the env with action=0 under our residual parameterization
      ⇒ ctrl = ref_q each step. Shows what our env's "perfect feedforward
      from policy" baseline achieves.

  --mode replay-actions
      Bypass the env's residual mapping and write the recorded
      ``actions/joint_pos[t]`` directly as ctrl. This is the dynamics-feasible
      ORACLE — the same ctrl signal that produced the recorded reference qpos
      in MolmoSpaces' data-gen sim. R_qpos in this mode is the upper bound
      ANY policy can reach in our env on this data. If oracle ≈ recorded
      qpos → physics is faithful, residual-policy is undertrained. If oracle
      diverges → sim-to-data dynamics gap, the ceiling is real.

Usage:
    uv run python scripts/render_molmobot_rollout.py \\
        --reference-h5 storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5 \\
        --traj-no 0 --mode replay-actions --out oracle.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("GLI_PATH", "/tmp")

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from tqdm import tqdm

from latent_mj.envs.molmobot_manipulation.train.molmobot_tracking_env import (
    MolmoBotTrackingEnv,
    get_default_tracking_config,
)
from mujoco_playground._src import mjx_env

_N_ARM = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", required=True)
    parser.add_argument("--traj-no", type=int, default=0)
    parser.add_argument("--mode", choices=["residual-zero", "replay-actions"],
                        default="residual-zero")
    parser.add_argument("--steps", type=int, default=None,
                        help="Default = full trajectory length.")
    parser.add_argument("--out", default="rollout.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--camera", default="-1",
                        help="Camera name or numeric id (-1 = free cam).")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip mp4 output, just print per-step error.")
    args = parser.parse_args()

    cfg = get_default_tracking_config()
    cfg.random_start = False

    env = MolmoBotTrackingEnv(config=cfg)
    traj = env.prepare_trajectory(args.reference_h5)
    n_trajs = int(traj.data.split_points.shape[0]) - 1
    print(f"loaded {n_trajs} trajs from {args.reference_h5}")
    assert 0 <= args.traj_no < n_trajs, f"traj-no must be in [0, {n_trajs})"

    sp = np.asarray(env._ref_split_points)
    start, end = int(sp[args.traj_no]), int(sp[args.traj_no + 1])
    traj_len = end - start
    print(f"trajectory {args.traj_no}: {traj_len} steps "
          f"(@ {1/env.dt:.1f} Hz = {traj_len * env.dt:.2f}s)  mode={args.mode}")

    # Slice the chosen trajectory's references (numpy for the oracle path).
    ref_qpos = np.asarray(env._ref_qpos[start:end])      # (T, 7)
    ref_qvel = np.asarray(env._ref_qvel[start:end])      # (T, 7)
    ref_actions = np.asarray(env._ref_actions[start:end])  # (T, 8)

    n_steps = args.steps or traj_len - 1
    n_steps = min(n_steps, traj_len - 1)

    # Initial state: qpos = ref_q[0], qvel = ref_v[0], rest at model defaults.
    init_qpos = jp.asarray(env._mj_model.qpos0).at[:_N_ARM].set(ref_qpos[0])
    init_qvel = jp.zeros(env._mjx_model.nv).at[:_N_ARM].set(ref_qvel[0])

    if args.mode == "residual-zero":
        init_ctrl = env._action_to_ctrl(jp.zeros(env.action_size),
                                        jp.asarray(ref_qpos[0]))
    else:  # replay-actions
        init_ctrl = jp.asarray(ref_actions[0])

    data = mjx_env.make_data(
        env.mj_model,
        qpos=init_qpos, qvel=init_qvel, ctrl=init_ctrl,
        impl=env._impl, naconmax=int(env._config.naconmax),
    )

    n_substeps = int(round(env._config.ctrl_dt / env._config.sim_dt))
    print(f"physics: ctrl_dt={env._config.ctrl_dt}s sim_dt={env._config.sim_dt}s "
          f"⇒ {n_substeps} sub-steps per env.step")

    @jax.jit
    def _step(data, ctrl):
        d = data.replace(ctrl=ctrl)
        d, _ = jax.lax.scan(
            lambda d, _: (mjx.step(env.mjx_model, d), None),
            d, None, length=n_substeps,
        )
        return d

    # Optional renderer setup.
    writer = renderer = mj_render_data = None
    cam_id = int(args.camera) if args.camera.lstrip("-").isdigit() else (
        mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    )
    if not args.no_render:
        renderer = mujoco.Renderer(env.mj_model, height=args.height, width=args.width)
        mj_render_data = mujoco.MjData(env.mj_model)
        writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=1)

    print("step | max|qpos-ref|  mean|qpos-ref|  max|qvel-ref|  R_qpos(σ=0.5)")
    sigma_qpos2 = 0.25
    err_log = []

    for t in tqdm(range(n_steps)):
        # Compare CURRENT data (post step t-1) to ref[t].
        arm_q = np.asarray(data.qpos[:_N_ARM])
        arm_v = np.asarray(data.qvel[:_N_ARM])
        target_q = ref_qpos[t]
        target_v = ref_qvel[t]
        dq = np.abs(arm_q - target_q)
        dv = np.abs(arm_v - target_v)
        r_qpos = float(np.exp(-np.sum((arm_q - target_q) ** 2) / sigma_qpos2))
        err_log.append((t, dq.max(), dq.mean(), dv.max(), r_qpos))

        if writer is not None:
            mj_render_data.qpos[:] = arm_q.tolist() + np.asarray(data.qpos[_N_ARM:]).tolist()
            mj_render_data.qpos[:] = np.asarray(data.qpos)
            mj_render_data.qvel[:] = np.asarray(data.qvel)
            mujoco.mj_forward(env.mj_model, mj_render_data)
            if cam_id >= 0:
                renderer.update_scene(mj_render_data, camera=cam_id)
            else:
                renderer.update_scene(mj_render_data)
            writer.append_data(renderer.render())

        # Pick ctrl for this step.
        if args.mode == "residual-zero":
            ctrl = env._action_to_ctrl(jp.zeros(env.action_size), jp.asarray(target_q))
        else:
            ctrl = jp.asarray(ref_actions[t])
        data = _step(data, ctrl)

    if writer is not None:
        writer.close()
        print(f"wrote {Path(args.out).resolve()}")

    # Summary.
    arr = np.array(err_log)
    print(f"\nover {len(arr)} steps:")
    print(f"  qpos-err  max:  mean={arr[:,1].mean():.4f}  median={np.median(arr[:,1]):.4f}  p95={np.percentile(arr[:,1],95):.4f}")
    print(f"  qpos-err  mean: mean={arr[:,2].mean():.4f}  median={np.median(arr[:,2]):.4f}")
    print(f"  qvel-err  max:  mean={arr[:,3].mean():.4f}  median={np.median(arr[:,3]):.4f}")
    print(f"  R_qpos σ=0.5:   mean={arr[:,4].mean():.4f}  final={arr[-1,4]:.4f}  min={arr[:,4].min():.4f}")


if __name__ == "__main__":
    main()
