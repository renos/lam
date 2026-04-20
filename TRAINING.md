# MolmoBot Tracking — Training Quickstart

Two ways to launch: direct CLI (fast iteration, uses whatever `CUDA_VISIBLE_DEVICES`
points at) and Hydra + submitit on SLURM (one job per GH200 chip, supports
`--multirun` sweeps).

## A. Hydra + SLURM (recommended for real runs)

### Single GH200 chip, one run

```bash
cd /home/renos/lam
python train.py --multirun hydra/launcher=slurm exp_name=v1
```

This submits one SLURM job to `partition=ghx4` with `gpus_per_node=1`. The
launcher config is at `conf/hydra/launcher/slurm.yaml` and already activates
`lam/.venv` and sets `MUJOCO_GL=egl`.

### Single GH200, shorter shakeout

```bash
cd /home/renos/lam
python train.py --multirun hydra/launcher=slurm \
    exp_name=warmup num_envs=512 num_timesteps=10_000_000
```

### Multi-seed sweep (3 jobs, 3 GH200 chips in parallel)

```bash
cd /home/renos/lam
python train.py --multirun hydra/launcher=slurm \
    exp_name=seed_sweep seed=0,1,2
```

### Hyperparameter grid (9 jobs)

```bash
cd /home/renos/lam
python train.py --multirun hydra/launcher=slurm \
    num_envs=512,1024,2048 num_timesteps=100_000_000 seed=0,1,2
```

### Full GH200 node (4 chips)

Override on command line:

```bash
cd /home/renos/lam
python train.py --multirun hydra/launcher=slurm \
    +hydra.launcher.gpus_per_node=4 exp_name=4gpu_run
```

Or edit `conf/hydra/launcher/slurm.yaml` and set `gpus_per_node: 4`.

### Local dry-run (no SLURM submission)

```bash
cd /home/renos/lam
python train.py exp_name=local_smoke num_envs=256 num_timesteps=5_000_000
```

## B. Direct CLI (fast iteration on current machine)

Useful for debugging on the interactive box with specific GPUs.

```bash
# Sanity check (2 min) — 16 envs, confirms stack boots
cd /home/renos/lam
MUJOCO_GL=egl GLI_PATH=/tmp CUDA_VISIBLE_DEVICES=0,1,6,7 \
  uv run python -u -m latent_mj.learning.train.train_ppo_track_molmobot \
    --reference-h5 storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5 \
    --num-timesteps 100000 \
    --exp-name debug
```

```bash
# Short full-scale shakeout (~5 min, 4 RTX 6000 Ada, 1024 envs)
cd /home/renos/lam
MUJOCO_GL=egl GLI_PATH=/tmp CUDA_VISIBLE_DEVICES=0,1,6,7 \
  uv run python -u -m latent_mj.learning.train.train_ppo_track_molmobot \
    --reference-h5 storage/molmobot_data_sample/extracted/house_0/trajectories_batch_4_of_20.h5 \
    --num-timesteps 5000000 \
    --num-envs 1024 \
    --exp-name warmup
```

## Monitor GPUs (separate terminal)

```bash
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

## Outputs

- stdout: `step=N dt=s sps=... R_sum=... R_qpos=... pi_loss=... std=...` every ~0.25s
- checkpoints: `/tmp/storage/logs/track_molmobot/<timestamp>_MolmoBotTracking_<exp-name>/checkpoints/<step>/`
- SLURM logs: `slurm_out/<job_id>/`

## Hydra overrides cheatsheet

Any field in `conf/config.yaml` is overridable on the CLI:

| Override | Effect |
| --- | --- |
| `num_envs=2048` | Scale parallel envs |
| `num_timesteps=1_000_000_000` | Longer run |
| `exp_name=my_run` | Name output dir |
| `seed=42` | Fix RNG |
| `reference_h5=/path/to/other.h5` | Different H5 |
| `+hydra.launcher.gpus_per_node=4` | Full GH200 node |
| `+hydra.launcher.timeout_min=1440` | 24-hour job |
| `+hydra.launcher.partition=debug` | Different partition |

## Gotchas

- Default `naconmax=8192` fits 256 envs/device. Bump in `latent_mj/envs/molmobot_manipulation/train/molmobot_tracking_env.py:get_default_tracking_config()` if you push past 512 envs/device.
- No wandb hooked up yet (progress is stdout-only).
- Reference pool is a single H5 (5 trajectories). Extend the loader to a directory glob for real training.
- Hydra launcher activates `/home/renos/lam/.venv`. If running on a different cluster, edit the `setup:` block in `conf/hydra/launcher/slurm.yaml`.
