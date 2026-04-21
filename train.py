"""Hydra + submitit entry point for MolmoBot motion-tracking PPO.

Local smoke run (1 GPU):
    python train.py exp_name=local_smoke num_envs=256 num_timesteps=5_000_000

SLURM single-GH200 run:
    python train.py --multirun hydra/launcher=slurm exp_name=v1

SLURM sweep over seeds:
    python train.py --multirun hydra/launcher=slurm seed=0,1,2 exp_name=seed_sweep

SLURM hyperparam grid:
    python train.py --multirun hydra/launcher=slurm \
        num_envs=512,1024,2048 num_timesteps=100_000_000

The hydra launcher config lives at conf/hydra/launcher/slurm.yaml and is set
up for a GH200 chip (partition=ghx4, gpus_per_node=1). Flip gpus_per_node
there (or override via +hydra.launcher.gpus_per_node=4) to take a full node.
"""

from __future__ import annotations

import os

# mujoco_warp + jax need these set BEFORE any jax/mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("GLI_PATH", "/tmp")
# molmo_spaces' episode_to_mj_model reads this to find robot/object/scene
# assets. Default to <repo>/storage/mlspaces_assets/ so scene_from_h5 works
# without extra shell exports.
_default_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "storage", "mlspaces_assets")
os.environ.setdefault("MLSPACES_ASSETS_DIR", _default_assets)

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve now so ${oc.env:HOME} etc. become concrete paths.
    OmegaConf.resolve(cfg)

    # Late import so the os.environ defaults above take effect before jax
    # touches the CUDA runtime.
    from latent_mj.learning.train.train_ppo_track_molmobot import Args, train

    def _opt(name, cast=None):
        v = cfg.get(name)
        if v is None:
            return None
        return cast(v) if cast is not None else v

    args = Args(
        reference_h5=cfg.reference_h5,
        task=cfg.task,
        exp_name=cfg.exp_name,
        num_timesteps=int(cfg.num_timesteps),
        num_envs=int(cfg.num_envs) if cfg.num_envs is not None else None,
        wandb=bool(cfg.get("wandb", False)),
        wandb_project=cfg.get("wandb_project", "lam-molmobot-tracking"),
        wandb_entity=cfg.get("wandb_entity"),
        wandb_mode=cfg.get("wandb_mode", "online"),
        # Scene-from-H5 (optional — populates env with one trajectory's objects)
        scene_from_h5=cfg.get("scene_from_h5"),
        scene_from_h5_traj_key=cfg.get("scene_from_h5_traj_key", "traj_0"),
        # PPO overrides (None → keep registered default)
        entropy_cost=_opt("entropy_cost", float),
        learning_rate=_opt("learning_rate", float),
        num_updates_per_batch=_opt("num_updates_per_batch", int),
        num_minibatches=_opt("num_minibatches", int),
        batch_size=_opt("batch_size", int),
        unroll_length=_opt("unroll_length", int),
        discounting=_opt("discounting", float),
        clipping_epsilon=_opt("clipping_epsilon", float),
    )
    print(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Resolved Args: {args}")
    train(args)


if __name__ == "__main__":
    main()
