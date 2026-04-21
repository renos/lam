"""Minimal Brax PPO entry-point for ``MolmoBotTracking``.

Mirrors ``train_ppo_track_tennis.py`` but takes an H5 reference path (or
directory of H5s) rather than a named MoCap dataset. Designed to be the
smoke-test driver until per-trajectory dataset packaging is built out.

Usage:
    python -m latent_mj.learning.train.train_ppo_track_molmobot \\
        --reference-h5 storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5 \\
        --num-timesteps 100000 --exp-name debug
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import numpy as np
import tyro
from absl import logging
from brax.training.agents.ppo.networks import make_ppo_networks

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo


@dataclass
class Args:
    reference_h5: str
    """Path to a molmobot-data H5 file (or directory containing one)."""

    task: str = "MolmoBotTracking"
    exp_name: str = "debug"
    num_timesteps: int = 100_000
    num_envs: Optional[int] = None
    """Override policy_config.num_envs (handy for small-GPU smoke tests)."""

    wandb: bool = False
    """If True, wandb.init + log per-progress-tick metrics."""
    wandb_project: str = "lam-molmobot-tracking"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online | offline | disabled


def _setup_paths(exp_name: str) -> Path:
    logdir = Path(WANDB_PATH_LOG) / "track_molmobot" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return ckpt_path


def _apply_debug_overrides(cfg) -> None:
    cfg.training_metrics_steps = 1000
    cfg.num_evals = 0
    cfg.batch_size = 8
    cfg.num_minibatches = 2
    cfg.num_envs = cfg.batch_size * cfg.num_minibatches
    cfg.episode_length = 200
    cfg.unroll_length = 10
    cfg.num_updates_per_batch = 1
    cfg.num_resets_per_eval = 1


def _prepare_training_params(cfg, ckpt_path: Path) -> dict:
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_fn
    params["network_factory"] = functools.partial(
        make_ppo_networks, **cfg.network_factory
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def train(args: Args) -> None:
    env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    timestamp = datetime.now().strftime("%m%d%H%M")
    exp_name = f"{timestamp}_{args.task}_{args.exp_name}"
    debug_mode = "debug" in exp_name

    ckpt_path = _setup_paths(exp_name)
    logging.info(f"Checkpoint path: {ckpt_path}")

    policy_cfg.num_timesteps = args.num_timesteps
    if args.num_envs is not None:
        policy_cfg.num_envs = args.num_envs
    if debug_mode:
        _apply_debug_overrides(policy_cfg)

    env = env_class(config=env_cfg)
    traj = env.prepare_trajectory(args.reference_h5)
    n_trajs = int(traj.data.split_points.shape[0]) - 1
    total_steps = int(traj.data.split_points[-1])
    print(
        f"Loaded {n_trajs} reference trajectories, {total_steps} total timesteps "
        f"@ {1.0 / env.dt:.1f} Hz from {args.reference_h5}"
    )
    print(f"Obs size: {env.observation_size}, action size: {env.action_size}")

    policy_params = _prepare_training_params(policy_cfg, ckpt_path)
    train_fn = functools.partial(ppo.train, **policy_params)

    times = [time.monotonic()]

    last_steps = [0]

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=exp_name,
            config={
                "task": args.task,
                "num_envs": args.num_envs or policy_cfg.num_envs,
                "num_timesteps": args.num_timesteps,
                "reference_h5": args.reference_h5,
                "n_reference_trajs": int(traj.data.split_points.shape[0]) - 1,
                "n_reference_steps": int(traj.data.split_points[-1]),
                "obs_size": dict(env.observation_size),
                "action_size": int(env.action_size),
            },
        )

    def _progress(num_steps: int, metrics: dict) -> None:
        now = time.monotonic()
        times.append(now)
        if len(times) > 1 and num_steps > 0:
            dt = times[-1] - times[-2]
            sps = (num_steps - last_steps[0]) / max(dt, 1e-6)
            r = lambda k: float(metrics.get(k, float("nan")))
            print(
                f"step={num_steps:>10d}  dt={dt:6.2f}s  sps={sps:>9.0f}  "
                f"R_sum={r('average/sum_reward'):+7.3f}  "
                f"R_qpos={r('average/reward/arm_qpos_tracking'):+6.3f}  "
                f"R_term={r('average/reward/termination'):+6.3f}  "
                f"ep_len={r('episode/length'):5.1f}  "
                f"pi_loss={r('training/policy_loss'):+7.4f}  "
                f"v_loss={r('training/v_loss'):+7.4f}  "
                f"std={r('training/policy_dist_mean_std'):5.3f}"
            )
            if args.wandb:
                import wandb
                log = {k: float(v) for k, v in metrics.items()
                       if hasattr(v, "__float__") or isinstance(v, (int, float))}
                log["wallclock/sps"] = sps
                wandb.log(log, step=num_steps)
        last_steps[0] = num_steps

    make_inference_fn, params, _ = train_fn(
        environment=env,
        trajectory_data=traj.data,
        progress_fn=_progress,
        policy_params_fn=lambda *args: None,
    )

    if len(times) > 1:
        print(f"JIT compile: {times[1] - times[0]:.2f}s")
        print(f"Train wall-time: {times[-1] - times[1]:.2f}s")

    print(f"Run {exp_name} done. Checkpoint: {ckpt_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train(tyro.cli(Args))
