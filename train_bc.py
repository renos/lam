"""Hydra + submitit entry point for molmobot goal-conditioned BC.

Local single-GPU smoke run:
    python train_bc.py exp_name=local_smoke num_epochs=2 batch_size=512

SLURM single-GH200 run:
    python train_bc.py --multirun hydra/launcher=slurm exp_name=bc_v1

LR / batch sweep on 4 GH200 chips:
    python train_bc.py --multirun hydra/launcher=slurm \\
        exp_name=bc_lr_sweep \\
        learning_rate=1e-4,3e-4,1e-3 batch_size=4096,8192

Complements train.py (PPO motion tracker) — uses the same H5 loader,
wandb project, and SLURM launcher config.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("GLI_PATH", "/tmp")
_default_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "storage", "mlspaces_assets")
os.environ.setdefault("MLSPACES_ASSETS_DIR", _default_assets)

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="bc_config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    from latent_mj.learning.train.train_bc_molmobot import Args, train

    args = Args(
        reference_h5=cfg.get("reference_h5", "") or "",
        stream_from_hf=bool(cfg.get("stream_from_hf", False)),
        hf_repo_id=cfg.get("hf_repo_id", "allenai/molmobot-data"),
        hf_datagen_config=cfg.get("hf_datagen_config", "FrankaPickOmniCamConfig"),
        hf_split=cfg.get("hf_split", "train"),
        hf_max_shards=(int(cfg.hf_max_shards) if cfg.get("hf_max_shards") is not None else None),
        shuffle_buffer=int(cfg.get("shuffle_buffer", 50_000)),
        num_train_batches=(int(cfg.num_train_batches) if cfg.get("num_train_batches") is not None else None),
        exp_name=cfg.exp_name,
        num_epochs=int(cfg.num_epochs),
        batch_size=int(cfg.batch_size),
        learning_rate=float(cfg.learning_rate),
        hidden_sizes=tuple(int(h) for h in cfg.hidden_sizes),
        train_frac=float(cfg.train_frac),
        seed=int(cfg.seed),
        log_every=int(cfg.log_every),
        eval_every=int(cfg.eval_every),
        wandb=bool(cfg.get("wandb", False)),
        wandb_project=cfg.get("wandb_project", "lam-molmobot-tracking"),
        wandb_entity=cfg.get("wandb_entity"),
        wandb_mode=cfg.get("wandb_mode", "online"),
    )
    print(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Resolved Args: {args}")
    train(args)


if __name__ == "__main__":
    main()
