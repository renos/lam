#!/usr/bin/env python
"""Smoke test for the MolmoBotPickEnv warp scaffold.

Run with:
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
        uv run python scripts/smoke_test_molmobot.py

Expected output ends with: MOLMOBOT_WARP_SCAFFOLD_OK
"""

import jax
import jax.numpy as jnp
from mujoco import mjx

from latent_mj.envs.molmobot_manipulation.train.pick_env import (
    MolmoBotPickEnv,
    get_default_config,
)


def main():
    print("=== MolmoBot warp scaffold smoke test ===")

    config = get_default_config()
    env = MolmoBotPickEnv(config=config, impl="warp")

    mjx_model = env.mjx_model
    assert mjx_model is not None, "mjx_model is None"

    print(f"  nq      = {env.mj_model.nq}")
    print(f"  nv      = {env.mj_model.nv}")
    print(f"  nu      = {env.mj_model.nu}")
    print(f"  solver  = {env.mj_model.opt.solver}")
    print(f"  cone    = {env.mj_model.opt.cone}")
    print(f"  impl    = {env.impl}")

    # Eager step
    data = mjx.make_data(env.mj_model, impl="warp")
    data = mjx.step(env.mjx_model, data)
    print("  eager mjx.step OK")

    # JIT step
    @jax.jit
    def jit_step(model, d):
        return mjx.step(model, d)

    data = jit_step(env.mjx_model, data)
    jax.block_until_ready(data.qpos)
    print("  jit mjx.step OK")

    print("\nMOLMOBOT_WARP_SCAFFOLD_OK")


if __name__ == "__main__":
    main()
