"""MolmoBotPickEnv — stub for pick-and-place task.

Reward, observation, and termination logic will be filled in by task #8.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from latent_mj.envs.molmobot_manipulation import constants as C
from latent_mj.envs.molmobot_manipulation.train.base_env import MolmoBotEnv
from latent_mj.envs.molmobot_manipulation.train.scene_loader import (
    patch_robot_into_scene,
    strip_warp_unsupported_options,
)


def get_default_config() -> config_dict.ConfigDict:
    """Minimal config required by MjxEnv base class."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
    )


class MolmoBotPickEnv(MolmoBotEnv):
    """Pick-and-place environment using MolmoSpaces base scene + Franka FR3.

    Scene is loaded by:
      1. Patching the Franka FR3 MJCF into MolmoSpaces' base_scene.xml via
         a temporary <include file="..."/> injection (preserves relative paths).
      2. Stripping mujoco_warp-incompatible options (multiccd, noslip).
      3. Compiling the MjSpec and handing the resulting XML path to
         MolmoBotEnv.__init__, which calls mjx.put_model(..., impl=impl).

    Reward / observation / termination: NOT YET IMPLEMENTED (task #8).
    """

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        impl: str = C.DEFAULT_IMPL,
        scene_xml_path: Optional[Path] = None,
        robot_xml_path: Optional[Path] = None,
    ) -> None:
        if config is None:
            config = get_default_config()

        scene_xml_path = Path(scene_xml_path or C.MOLMOSPACES_BASE_SCENE_XML)
        robot_xml_path = Path(robot_xml_path or C.FRANKA_DROID_XML)

        # Step 1: patch robot into scene and strip warp-incompatible options.
        spec = patch_robot_into_scene(scene_xml_path, robot_xml_path)
        strip_warp_unsupported_options(spec)

        # Step 2: compile spec → MjModel and pass it directly to base init.
        mj_model = spec.compile()
        super().__init__(
            mj_model=mj_model,
            config=config,
            config_overrides=config_overrides,
            impl=impl,
        )

    # ------------------------------------------------------------------
    # Abstract methods required by MjxEnv — stubs only (task #8 fills in)
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        raise NotImplementedError(
            "MolmoBotPickEnv.reset() is a stub. "
            "Implement in task #8 (obs, reward, termination)."
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        raise NotImplementedError(
            "MolmoBotPickEnv.step() is a stub. "
            "Implement in task #8 (obs, reward, termination)."
        )

    def _get_obs(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        raise NotImplementedError(
            "MolmoBotPickEnv._get_obs() is a stub. "
            "Implement in task #8."
        )

    def _get_reward(
        self, data: mjx.Data, action: jax.Array
    ) -> tuple[jax.Array, dict]:
        raise NotImplementedError(
            "MolmoBotPickEnv._get_reward() is a stub. "
            "Implement in task #8."
        )

    @classmethod
    def from_episode(
        cls,
        episode: dict,
        impl: str = C.DEFAULT_IMPL,
    ) -> "MolmoBotPickEnv":
        """Construct a MolmoBotPickEnv from a benchmark episode dict.

        Calls episode_to_mj_model to build the MjModel, then initialises the
        env via MolmoBotEnv.__init__ (passing mj_model directly so no XML
        round-trip is needed).  The raw episode dict and parsed metadata are
        stored as ``env._episode`` and ``env._episode_metadata``.
        """
        from latent_mj.envs.molmobot_manipulation.train.episode_loader import (
            episode_to_mj_model,
        )

        mj_model, metadata = episode_to_mj_model(episode, impl=impl)
        env = cls.__new__(cls)
        MolmoBotEnv.__init__(
            env,
            mj_model=mj_model,
            config=get_default_config(),
            impl=impl,
        )
        env._episode = episode
        env._episode_metadata = metadata
        return env

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        raise NotImplementedError(
            "MolmoBotPickEnv.observation_size is a stub. "
            "Implement in task #8."
        )
