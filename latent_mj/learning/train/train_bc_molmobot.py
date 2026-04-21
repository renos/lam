"""Goal-conditioned behavior cloning on the molmobot-data dataset.

Trains π(s_t, target_s_{t+1}) → a_t as pure supervised learning — no sim,
no physics, no PPO. Complements the RL tracker in train_ppo_track_molmobot:
use this to warm-start or, with enough data, to replace the tracker entirely.

State currently = arm qpos (7) + arm qvel (7) = 14-dim proprio.
Goal  = next-step arm qpos = 7-dim.
Action = recorded actions/joint_pos (arm 7 + gripper 1) = 8-dim.

Extend `_triple` below to include object poses / gripper state / language
tokens / camera embeddings for richer conditioning.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

_N_ARM = 7
_ACTION_DIM = 8  # arm 7 + gripper 1


@dataclass
class Args:
    reference_h5: str = ""
    """Path to a single H5 or a directory of H5s (directory = recursive glob).
    Required when ``stream_from_hf=False``; ignored when streaming."""

    stream_from_hf: bool = False
    """Pull shards on demand from HuggingFace instead of a local directory.
    Uses allenai/molmobot-data by default."""
    hf_repo_id: str = "allenai/molmobot-data"
    hf_datagen_config: str = "FrankaPickOmniCamConfig"
    hf_split: str = "train"
    hf_max_shards: Optional[int] = None
    """If streaming, cap the number of outer tar shards per epoch."""
    shuffle_buffer: int = 50_000
    """Reservoir buffer size for cross-shard shuffling (streaming mode only)."""
    num_train_batches: Optional[int] = None
    """If streaming, stop the epoch after this many gradient steps. None =
    iterate until the shard stream is exhausted."""

    exp_name: str = "bc_debug"
    num_epochs: int = 50
    batch_size: int = 8192
    learning_rate: float = 3e-4
    hidden_sizes: tuple[int, ...] = (256, 256, 128)

    train_frac: float = 0.95
    """Fraction of trajectories (not samples) used for training; rest = eval.
    Only applies to the local-dir loader; streaming uses a held-out shard list."""
    seed: int = 0

    log_every: int = 50
    """Print + wandb-log every N gradient steps."""
    eval_every: int = 500

    wandb: bool = False
    wandb_project: str = "lam-molmobot-tracking"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


class GoalConditionedPolicy(nn.Module):
    hidden_sizes: tuple[int, ...]
    action_dim: int = _ACTION_DIM

    @nn.compact
    def __call__(self, s_t: jnp.ndarray, s_goal: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s_t, s_goal], axis=-1)
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.swish(x)
        return nn.Dense(self.action_dim)(x)


def _triples_from_trajectory(traj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (s_t, s_{t+1}, a_t) numpy arrays from a loaded Trajectory.

    We do NOT pair samples across trajectory boundaries — for each trajectory
    of length T we emit T-1 triples.
    """
    qpos = np.asarray(traj.data.qpos[:, :_N_ARM])
    qvel = np.asarray(traj.data.qvel[:, :_N_ARM])
    actions = np.asarray(traj.info.metadata["recorded_actions"])
    split = np.asarray(traj.data.split_points)

    s_t_list, s_tp1_list, a_t_list, traj_of_sample = [], [], [], []
    for traj_idx in range(len(split) - 1):
        start, end = int(split[traj_idx]), int(split[traj_idx + 1])
        if end - start < 2:
            continue
        s = np.concatenate([qpos[start:end], qvel[start:end]], axis=-1)  # (T, 14)
        g = qpos[start:end]                                              # (T, 7)
        a = actions[start:end]                                           # (T, 8)
        s_t_list.append(s[:-1])
        s_tp1_list.append(g[1:])
        a_t_list.append(a[:-1])
        traj_of_sample.append(np.full(end - start - 1, traj_idx, dtype=np.int32))

    s_t = np.concatenate(s_t_list, axis=0).astype(np.float32)
    s_goal = np.concatenate(s_tp1_list, axis=0).astype(np.float32)
    a_t = np.concatenate(a_t_list, axis=0).astype(np.float32)
    traj_of_sample = np.concatenate(traj_of_sample, axis=0)
    return s_t, s_goal, a_t, traj_of_sample


def _split_train_eval(
    s_t: np.ndarray,
    s_goal: np.ndarray,
    a_t: np.ndarray,
    traj_idx: np.ndarray,
    train_frac: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_trajs = int(traj_idx.max()) + 1
    shuffled = rng.permutation(n_trajs)
    n_train = int(round(train_frac * n_trajs))
    train_trajs = set(shuffled[:n_train].tolist())
    mask = np.array([int(t) in train_trajs for t in traj_idx])
    return {
        "train": (s_t[mask], s_goal[mask], a_t[mask]),
        "eval":  (s_t[~mask], s_goal[~mask], a_t[~mask]),
        "n_train_trajs": n_train,
        "n_eval_trajs": n_trajs - n_train,
    }


def _batch_iter(arrays: tuple[np.ndarray, ...], batch_size: int, rng: np.random.Generator):
    n = arrays[0].shape[0]
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield tuple(a[sel] for a in arrays)


def _loss_fn(params, apply_fn, s_t, s_goal, a_t):
    pred = apply_fn(params, s_t, s_goal)
    return jnp.mean((pred - a_t) ** 2)


def train(args: Args) -> None:
    if args.stream_from_hf:
        _train_streaming(args)
        return
    _train_local_dir(args)


def _train_local_dir(args: Args) -> None:
    from latent_mj.envs.molmobot_manipulation.train.molmobot_traj_loader import (
        load_h5_reference_dataset,
    )

    assert args.reference_h5, "reference_h5 required when stream_from_hf=False"
    print(f"loading dataset from {args.reference_h5}")
    traj = load_h5_reference_dataset(args.reference_h5)
    s_t, s_goal, a_t, traj_idx = _triples_from_trajectory(traj)
    print(f"  total (s, s', a) triples: {len(s_t):,}  "
          f"(s_t dim={s_t.shape[1]}, goal dim={s_goal.shape[1]}, a dim={a_t.shape[1]})")

    splits = _split_train_eval(s_t, s_goal, a_t, traj_idx, args.train_frac, args.seed)
    train_arrays = splits["train"]
    eval_arrays = splits["eval"]
    print(f"  split: {splits['n_train_trajs']} train trajs ({len(train_arrays[0]):,} samples),  "
          f"{splits['n_eval_trajs']} eval trajs ({len(eval_arrays[0]):,} samples)")

    # Normalize inputs (zero-mean unit-std) using training stats.
    s_t_mean = train_arrays[0].mean(axis=0)
    s_t_std = train_arrays[0].std(axis=0) + 1e-6
    g_mean = train_arrays[1].mean(axis=0)
    g_std = train_arrays[1].std(axis=0) + 1e-6
    def _norm_inputs(s, g):
        return (s - s_t_mean) / s_t_std, (g - g_mean) / g_std

    policy = GoalConditionedPolicy(hidden_sizes=tuple(args.hidden_sizes))
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = policy.init(init_rng, jnp.zeros((1, s_t.shape[1])),
                         jnp.zeros((1, s_goal.shape[1])))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate))
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, s, g, a):
        loss, grads = jax.value_and_grad(_loss_fn)(state.params, state.apply_fn, s, g, a)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(params, s, g, a):
        return _loss_fn(params, policy.apply, s, g, a)

    # Experiment setup (dirs, wandb).
    timestamp = datetime.now().strftime("%m%d%H%M")
    exp_name = f"{timestamp}_BC_{args.exp_name}"
    logdir = Path(os.environ.get("WANDB_DIR", "/tmp")) / "track_molmobot_bc" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"logdir: {logdir}")

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            name=exp_name,
            config={
                "reference_h5": args.reference_h5,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_sizes": list(args.hidden_sizes),
                "n_train_samples": len(train_arrays[0]),
                "n_eval_samples": len(eval_arrays[0]),
                "n_train_trajs": splits["n_train_trajs"],
                "n_eval_trajs": splits["n_eval_trajs"],
                "s_t_dim": s_t.shape[1],
                "goal_dim": s_goal.shape[1],
                "action_dim": a_t.shape[1],
            },
        )

    # Pre-normalize entire arrays once to keep the inner loop cheap.
    s_t_train_n = (train_arrays[0] - s_t_mean) / s_t_std
    g_train_n = (train_arrays[1] - g_mean) / g_std
    a_train = train_arrays[2]
    s_t_eval_n = (eval_arrays[0] - s_t_mean) / s_t_std
    g_eval_n = (eval_arrays[1] - g_mean) / g_std
    a_eval = eval_arrays[2]

    step = 0
    best_eval = float("inf")
    t0 = time.monotonic()
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(args.num_epochs):
        for batch in _batch_iter((s_t_train_n, g_train_n, a_train), args.batch_size, np_rng):
            state, loss = train_step(state, *batch)
            step += 1

            if step % args.log_every == 0:
                l = float(loss)
                sps = step * args.batch_size / max(time.monotonic() - t0, 1e-6)
                print(f"epoch={epoch:>3d}  step={step:>7d}  train_mse={l:.5f}  samples/s={sps:>9,.0f}")
                if args.wandb:
                    import wandb
                    wandb.log({"train/mse": l, "train/samples_per_sec": sps,
                               "train/epoch": epoch}, step=step)

            if step % args.eval_every == 0:
                n_eval = s_t_eval_n.shape[0]
                if n_eval > 0:
                    eval_bs = min(args.batch_size, n_eval)
                    eval_losses = []
                    for i in range(0, n_eval, eval_bs):
                        el = eval_step(state.params,
                                       s_t_eval_n[i:i+eval_bs],
                                       g_eval_n[i:i+eval_bs],
                                       a_eval[i:i+eval_bs])
                        eval_losses.append(float(el))
                    eval_mse = float(np.mean(eval_losses))
                    if eval_mse < best_eval:
                        best_eval = eval_mse
                    print(f"  EVAL step={step}  eval_mse={eval_mse:.5f}  "
                          f"best={best_eval:.5f}")
                    if args.wandb:
                        import wandb
                        wandb.log({"eval/mse": eval_mse, "eval/best_mse": best_eval},
                                  step=step)

    # Final checkpoint
    ckpt = {
        "params": state.params,
        "normalizer": {"s_t_mean": s_t_mean, "s_t_std": s_t_std,
                        "g_mean": g_mean, "g_std": g_std},
        "hidden_sizes": list(args.hidden_sizes),
    }
    ckpt_path = logdir / "final_ckpt"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpointer = PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(str(ckpt_path.resolve()), ckpt, save_args=save_args, force=True)
    print(f"\nsaved final checkpoint to {ckpt_path}")

    if args.wandb:
        import wandb
        wandb.finish()


def _train_streaming(args: Args) -> None:
    """Train BC by streaming shards from HuggingFace one at a time.

    First shard is used to compute input normalization stats; all
    subsequent shards feed the training loop via a shuffle-buffer.
    """
    from latent_mj.utils.dataset.molmobot_hf_stream import (
        StreamingMolmobotDataset,
    )

    print(f"streaming {args.hf_repo_id} / {args.hf_datagen_config} / {args.hf_split}  "
          f"(max_shards={args.hf_max_shards})")
    ds = StreamingMolmobotDataset(
        repo_id=args.hf_repo_id,
        datagen_config=args.hf_datagen_config,
        split=args.hf_split,
        max_shards=args.hf_max_shards,
    )

    # Compute normalizer from the first shard, then restart the stream.
    print("pre-pass: computing normalizer from first shard...")
    first_shard_triples: Optional[tuple[np.ndarray, ...]] = None
    for s, g, a in ds.iter_triples(seed=args.seed):
        first_shard_triples = (s, g, a)
        break
    if first_shard_triples is None:
        raise RuntimeError("streaming produced no triples on the first shard")
    fs, fg, _ = first_shard_triples
    s_t_mean = fs.mean(axis=0)
    s_t_std = fs.std(axis=0) + 1e-6
    g_mean = fg.mean(axis=0)
    g_std = fg.std(axis=0) + 1e-6
    print(f"  normalizer from {fs.shape[0]:,} samples")

    policy = GoalConditionedPolicy(hidden_sizes=tuple(args.hidden_sizes))
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = policy.init(init_rng, jnp.zeros((1, fs.shape[1])),
                         jnp.zeros((1, fg.shape[1])))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate))
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, s, g, a):
        loss, grads = jax.value_and_grad(_loss_fn)(state.params, state.apply_fn, s, g, a)
        return state.apply_gradients(grads=grads), loss

    # Experiment setup (dirs, wandb).
    timestamp = datetime.now().strftime("%m%d%H%M")
    exp_name = f"{timestamp}_BC_stream_{args.exp_name}"
    logdir = Path(os.environ.get("WANDB_DIR", "/tmp")) / "track_molmobot_bc" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"logdir: {logdir}")

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            name=exp_name,
            config={
                "stream_from_hf": True,
                "hf_repo_id": args.hf_repo_id,
                "hf_datagen_config": args.hf_datagen_config,
                "hf_split": args.hf_split,
                "hf_max_shards": args.hf_max_shards,
                "shuffle_buffer": args.shuffle_buffer,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_sizes": list(args.hidden_sizes),
                "s_t_dim": fs.shape[1],
                "goal_dim": fg.shape[1],
            },
        )

    step = 0
    t0 = time.monotonic()
    samples_seen = 0
    for epoch in range(args.num_epochs):
        batch_iter = ds.iter_batches(
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed + epoch,
        )
        for s, g, a in batch_iter:
            s = (s - s_t_mean) / s_t_std
            g = (g - g_mean) / g_std
            state, loss = train_step(state, s, g, a)
            step += 1
            samples_seen += s.shape[0]

            if step % args.log_every == 0:
                l = float(loss)
                sps = samples_seen / max(time.monotonic() - t0, 1e-6)
                print(f"epoch={epoch:>3d}  step={step:>7d}  train_mse={l:.5f}  samples/s={sps:>9,.0f}")
                if args.wandb:
                    import wandb
                    wandb.log({"train/mse": l, "train/samples_per_sec": sps,
                               "train/epoch": epoch}, step=step)
            if args.num_train_batches is not None and step >= args.num_train_batches:
                break
        if args.num_train_batches is not None and step >= args.num_train_batches:
            break

    # Final checkpoint
    ckpt = {
        "params": state.params,
        "normalizer": {"s_t_mean": s_t_mean, "s_t_std": s_t_std,
                        "g_mean": g_mean, "g_std": g_std},
        "hidden_sizes": list(args.hidden_sizes),
    }
    ckpt_path = logdir / "final_ckpt"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpointer = PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(str(ckpt_path.resolve()), ckpt, save_args=save_args, force=True)
    print(f"\nsaved final checkpoint to {ckpt_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train(tyro.cli(Args))
