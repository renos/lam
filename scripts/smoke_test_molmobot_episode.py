"""Smoke test: load episode 0 from FrankaPickHardBench (iThor split) and step it."""
import json
from pathlib import Path
import jax
from mujoco import mjx

from latent_mj.envs.molmobot_manipulation.train.pick_env import MolmoBotPickEnv
from latent_mj.envs.molmobot_manipulation.train.episode_loader import load_benchmark

bench_path = Path(
    "/home/renos/lam/storage/mlspaces_assets/benchmarks/molmospaces-bench-v1/"
    "ithor/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark/benchmark.json"
)
episodes = load_benchmark(bench_path)
print(f"loaded {len(episodes)} episodes")

ep = episodes[0]
print(f"episode 0: house={ep['house_index']} task='{ep['language']['task_description']}'")

env = MolmoBotPickEnv.from_episode(ep, impl="warp")
print(f"  nq={env.mj_model.nq} nv={env.mj_model.nv} nu={env.mj_model.nu} "
      f"ngeom={env.mj_model.ngeom} nbody={env.mj_model.nbody}")
print(f"  solver={env.mj_model.opt.solver} cone={env.mj_model.opt.cone}")
print(f"  metadata: pickup={env._episode_metadata['pickup_obj_name']}")

# Bump njmax/nconmax to handle the larger scene
data = mjx.make_data(env.mj_model, impl="warp", njmax=1024, nconmax=512, naconmax=512)
data = mjx.step(env.mjx_model, data)
print("  eager step OK")

@jax.jit
def jit_step(model, d):
    return mjx.step(model, d)

data = jit_step(env.mjx_model, data)
jax.block_until_ready(data.qpos)
print("  jit step OK")

print("MOLMOBOT_EPISODE_LOADER_OK")
