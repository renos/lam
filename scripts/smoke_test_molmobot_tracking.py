"""Smoke test: vmap-wrapped MolmoBotTrackingEnv over N envs.

Verifies the env can be wrapped, jit-compiled, and stepped for short rollouts
with random actions.

Usage:
    uv run python scripts/smoke_test_molmobot_tracking.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("GLI_PATH", "/tmp")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jp

import latent_mj as lmj
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn

H5_PATH = (
    "/home/renos/lam/storage/molmobot_data_sample/extracted/"
    "house_0/trajectories_batch_4_of_20.h5"
)
N_ENVS = 4
N_STEPS = 20
EPISODE_LENGTH = 200


def main() -> None:
    EnvCls = lmj.registry.get("MolmoBotTracking", "tracking_train_env_class")
    cfg = lmj.registry.get("MolmoBotTracking", "tracking_config")

    env = EnvCls(config=cfg.env_config)
    traj = env.prepare_trajectory(H5_PATH)
    print(f"reference: n_trajs={int(traj.data.split_points.shape[0]) - 1}, "
          f"total_steps={int(traj.data.split_points[-1])}")
    print(f"obs_size: {env.observation_size}, action_size: {env.action_size}")

    wrapped = wrap_fn(env, episode_length=EPISODE_LENGTH, action_repeat=1)

    rng = jax.random.split(jax.random.PRNGKey(0), N_ENVS)

    jreset = jax.jit(wrapped.reset)
    jstep = jax.jit(wrapped.step)

    state = jreset(rng, traj.data)
    print(f"reset OK; state['state'].shape={state.obs['state'].shape} "
          f"reward.shape={state.reward.shape}")

    rewards = []
    dones = []
    for i in range(N_STEPS):
        rng_step, _ = jax.random.split(jax.random.PRNGKey(i + 1), 2)
        action = jax.random.uniform(rng_step, (N_ENVS, env.action_size), minval=-0.05, maxval=0.05)
        state = jstep(state, action, traj.data)
        rewards.append(float(state.reward.mean()))
        dones.append(float(state.done.mean()))

    print(f"after {N_STEPS} steps:")
    print(f"  mean reward over time: {[f'{r:.3f}' for r in rewards]}")
    print(f"  mean done   over time: {[f'{d:.2f}' for d in dones]}")
    print(f"  any NaN in qpos: {bool(jp.isnan(state.data.qpos).any())}")
    print(f"  any NaN in obs:  {bool(jp.isnan(state.obs['state']).any())}")
    print("MOLMOBOT_TRACKING_SMOKE_OK")


if __name__ == "__main__":
    main()
