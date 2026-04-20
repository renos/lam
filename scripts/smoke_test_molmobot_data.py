"""Smoke test: load a molmobot-data H5 trajectory and step the env on warp."""
from pathlib import Path
import jax
from mujoco import mjx

from latent_mj.envs.molmobot_manipulation.train.molmobot_data_loader import (
    load_h5_trajectory,
)
from latent_mj.envs.molmobot_manipulation.train.pick_env import MolmoBotPickEnv

H5 = Path(
    "/home/renos/lam/storage/molmobot_data_sample/extracted/house_0/"
    "trajectories_batch_4_of_20.h5"
)

ep = load_h5_trajectory(H5, traj_key="traj_0", datagen_config_name="FrankaPickOmniCamConfig")
print(f"house={ep['house_index']} scene_dataset={ep['scene_dataset']} split={ep['data_split']}")
print(f"task: {ep['language']['task_description']}")
print(f"pickup: {ep['task']['pickup_obj_name']}")
print(f"added_objects: {len(ep['scene_modifications']['added_objects'])}")
print(f"object_poses: {len(ep['scene_modifications']['object_poses'])}")

env = MolmoBotPickEnv.from_episode(ep, impl="warp")
print(f"  nq={env.mj_model.nq} nv={env.mj_model.nv} nu={env.mj_model.nu}")

data = mjx.make_data(env.mj_model, impl="warp", njmax=2048, nconmax=1024, naconmax=1024)
data = mjx.step(env.mjx_model, data)
print("eager step OK")

@jax.jit
def jit_step(model, d):
    return mjx.step(model, d)
data = jit_step(env.mjx_model, data)
jax.block_until_ready(data.qpos)
print("jit step OK")
print("MOLMOBOT_DATA_LOADER_OK")
