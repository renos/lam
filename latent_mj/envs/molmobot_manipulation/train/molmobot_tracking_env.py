"""MolmoBotTrackingEnv — joint-space reference tracking for Franka FR3.

Reference signal: per-step arm qpos (7 dims) extracted from molmobot-data H5
trajectories via molmobot_traj_loader. The environment is fixed-base, so we
skip TrajectoryHandler's name-matching and store reference arrays directly.

Action: 8-dim (7 arm + 1 gripper) in [-1, 1], rescaled to actuator ctrl_range.
Reward: arm tracking + small action smoothness penalty.
Termination: trajectory exhausted, large qpos deviation, or NaNs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

import latent_mj as lmj
from latent_mj.envs.molmobot_manipulation import constants as C
from latent_mj.envs.molmobot_manipulation.train.base_env import MolmoBotEnv
from latent_mj.envs.molmobot_manipulation.train.scene_loader import (
    patch_robot_into_scene,
    strip_warp_unsupported_options,
)
from latent_mj.envs.molmobot_manipulation.train.molmobot_traj_loader import (
    load_h5_reference_dataset,
)
from latent_mj.utils.dataset.traj_class import Trajectory


# Number of arm joints we track (7-DOF Franka FR3).
_N_ARM = 7
EPISODE_LENGTH = 200


def molmobot_tracking_task_config() -> config_dict.ConfigDict:
    """Unified env + PPO task config registered as ``MolmoBotTracking``."""
    env_config = get_default_tracking_config()
    env_config.episode_length = EPISODE_LENGTH

    policy_config = config_dict.create(
        num_timesteps=200_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        # Heavy scene physics → smaller batch than tennis. 1024 envs is a safe
        # starting point for a single RTX 6000 Ada (47 GB) with the iThor
        # base scene + franka_droid (nq=13).
        num_envs=1024,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=None,
        # PPO
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=512,
        num_minibatches=16,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 256, 128),
            value_hidden_layer_sizes=(256, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        num_evals=0,
        log_training_metrics=True,
        training_metrics_steps=int(1e5),
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )

    return config_dict.create(env_config=env_config, policy_config=policy_config)


def get_default_tracking_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=200,
        action_repeat=1,
        action_scale=1.0,
        random_start=True,
        termination_qpos_threshold=1.5,  # rad — kill if any arm joint > this far from ref
        # Residual-action parameterization:
        #   arm_ctrl[i] = ref_q[i] + arm_action_scale * tanh(action[i])
        # so policy output a=0 ⇒ ctrl = ref_q (no transient at reset).
        arm_action_scale=0.5,  # rad max residual from reference per arm joint
        # Gripper polarity: Robotiq fingers_actuator ctrl_range is [0, 255],
        # with 0 = open, 255 = closed. Map asymmetrically so action=0 ⇒ open
        # (keeps the policy from random-walk-closing through objects with no
        # gripper tracking signal).
        gripper_ctrl_max=255.0,
        # mujoco_warp pre-allocates a global contact buffer per device with this
        # capacity. For 256 envs/device the broadphase needs ~1300; 8192 leaves
        # plenty of headroom for larger batches and richer scenes. Increase if
        # warp logs "broadphase overflow - please increase ... naconmax".
        naconmax=8192,
        reward_config=config_dict.create(
            scales=config_dict.create(
                arm_qpos_tracking=1.0,
                arm_qvel_tracking=0.1,
                action_rate=-0.01,
                termination=-50.0,
            ),
            sigmas=config_dict.create(
                arm_qpos=0.5,
                arm_qvel=2.0,
            ),
        ),
        obs_keys=[
            "arm_qpos",
            "arm_qvel",
            "ref_arm_qpos",
            "ref_arm_qvel",
            "dif_arm_qpos",
            "dif_arm_qvel",
            "last_action",
        ],
        privileged_obs_keys=[
            "arm_qpos",
            "arm_qvel",
            "ref_arm_qpos",
            "ref_arm_qvel",
            "dif_arm_qpos",
            "dif_arm_qvel",
            "last_action",
        ],
    )


@lmj.registry.register("MolmoBotTracking", "tracking_train_env_class")
class MolmoBotTrackingEnv(MolmoBotEnv):
    """Joint-space reference tracking for Franka FR3 on a MolmoBot scene.

    Constructed with the MolmoSpaces base scene + franka_droid robot (no
    objects) for fast-iteration tracking training. For full-scene tracking
    (with all the household objects) construct via ``MolmoBotPickEnv``-style
    scene patching and pass ``mj_model``/``xml_path`` overrides.

    Reference data is loaded via ``prepare_trajectory(h5_path)`` which calls
    ``load_h5_reference_dataset``. Only the arm 7-DOF qpos/qvel are used for
    tracking; the gripper is driven by the policy's 8th action dim.
    """

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        impl: str = C.DEFAULT_IMPL,
        scene_xml_path: Optional[Path] = None,
        robot_xml_path: Optional[Path] = None,
        scene_from_h5: Optional[Path] = None,
        scene_from_h5_traj_key: str = "traj_0",
    ) -> None:
        if config is None:
            config = get_default_tracking_config()

        scene_xml_path = Path(scene_xml_path or C.MOLMOSPACES_BASE_SCENE_XML)
        robot_xml_path = Path(robot_xml_path or C.FRANKA_DROID_XML)

        mj_model = self._build_scene_mj_model(
            scene_xml_path=scene_xml_path,
            robot_xml_path=robot_xml_path,
            scene_from_h5=scene_from_h5,
            scene_from_h5_traj_key=scene_from_h5_traj_key,
        )

        super().__init__(
            mj_model=mj_model,
            config=config,
            config_overrides=config_overrides,
            impl=impl,
        )

        # Resolve actuator ctrl ranges for action rescaling.
        self._ctrl_lo = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_hi = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

        # Default qpos/qvel — used for non-arm joints at reset.
        self._default_qpos = jp.array(self._mj_model.qpos0)

        # Arm qpos/qvel indices: convention is the first 7 model joints are the
        # arm (verified for franka_droid attached at scene root). Hard-coded
        # rather than name-matched because the molmobot-data trajectory uses
        # 'panda_*' naming while the model uses 'robot/fr3_*'.
        self._arm_qpos_idx = jp.arange(_N_ARM)
        self._arm_qvel_idx = jp.arange(_N_ARM)

        # Reference data placeholders — populated by prepare_trajectory().
        self._ref_qpos: Optional[jp.ndarray] = None     # (total_T, 7)
        self._ref_qvel: Optional[jp.ndarray] = None     # (total_T, 7)
        self._ref_split_points: Optional[jp.ndarray] = None  # (n_trajs+1,)

    @staticmethod
    def _build_scene_mj_model(
        scene_xml_path: Path,
        robot_xml_path: Path,
        scene_from_h5: Optional[Path],
        scene_from_h5_traj_key: str,
    ):
        """Compile the env's MjModel.

        If ``scene_from_h5`` is set, load that trajectory's recorded
        scene_modifications (added_objects + object_poses) and compile a full
        populated scene via ``episode_to_mj_model`` — single compile used for
        ALL envs, so tracking cost is unchanged. Falls back to the empty
        base_scene + robot if loading fails (e.g. missing objaverse assets).
        """
        if scene_from_h5 is not None:
            try:
                from latent_mj.envs.molmobot_manipulation.train.molmobot_data_loader import (
                    load_h5_trajectory,
                )
                from latent_mj.envs.molmobot_manipulation.train.episode_loader import (
                    episode_to_mj_model,
                )
                print(
                    f"MolmoBotTrackingEnv: loading scene from "
                    f"{scene_from_h5}:{scene_from_h5_traj_key}"
                )
                episode = load_h5_trajectory(
                    Path(scene_from_h5), traj_key=scene_from_h5_traj_key
                )
                mj_model, metadata = episode_to_mj_model(episode)
                print(
                    f"  populated scene: {len(metadata.get('added_objects', {}))} "
                    f"added objects, house_index={metadata.get('house_index')}"
                )
                return mj_model
            except Exception as exc:  # noqa: BLE001
                print(
                    f"MolmoBotTrackingEnv: scene_from_h5 failed ({type(exc).__name__}: {exc}) — "
                    f"falling back to empty base_scene + franka_droid."
                )

        spec = patch_robot_into_scene(scene_xml_path, robot_xml_path)
        strip_warp_unsupported_options(spec)
        return spec.compile()

    # ------------------------------------------------------------------
    # Reference trajectory loading
    # ------------------------------------------------------------------

    def prepare_trajectory(self, h5_path: Union[str, Path]) -> Trajectory:
        """Load reference trajectories from a molmobot-data H5 file.

        Stores per-step arm qpos/qvel + recorded ctrl actions and the
        trajectory split_points as JAX arrays on self. Returns the
        underlying Trajectory for inspection.
        """
        traj = load_h5_reference_dataset(
            h5_path, control_dt=float(self._config.ctrl_dt)
        )
        # Trim to arm joints (first 7 dims).
        self._ref_qpos = jp.array(traj.data.qpos[:, :_N_ARM], dtype=jp.float32)
        self._ref_qvel = jp.array(traj.data.qvel[:, :_N_ARM], dtype=jp.float32)
        # Recorded 8-dim ctrl actions per step (arm 7 + gripper 1).  Used by
        # the oracle replay path in scripts/render_molmobot_rollout.py and
        # available for any future "feedforward" reward terms.
        self._ref_actions = jp.array(
            traj.info.metadata["recorded_actions"], dtype=jp.float32
        )
        self._ref_split_points = jp.array(
            traj.data.split_points, dtype=jp.int32
        )
        return traj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_reference_loaded(self) -> None:
        if self._ref_qpos is None:
            raise RuntimeError(
                "MolmoBotTrackingEnv: call prepare_trajectory(h5_path) before "
                "reset()/step()."
            )

    def _ref_at(self, traj_no: jax.Array, sub_t: jax.Array):
        """Look up reference (qpos, qvel) at trajectory ``traj_no`` step ``sub_t``."""
        start = self._ref_split_points[traj_no]
        idx = start + sub_t
        return self._ref_qpos[idx], self._ref_qvel[idx]

    def _ref_length(self, traj_no: jax.Array) -> jax.Array:
        return self._ref_split_points[traj_no + 1] - self._ref_split_points[traj_no]

    def _action_to_ctrl(self, action: jax.Array, ref_q: jax.Array) -> jax.Array:
        """Residual-action → actuator ctrl.

        Arm (first 7): ``ctrl = ref_q + arm_action_scale * tanh(action)`` so
        that ``action=0`` at reset produces no transient (ctrl == ref_q).

        Gripper (index 7): asymmetric map ``ctrl = max(0, action) * ctrl_hi``
        so that ``action=0`` keeps the gripper open. Clipped to actuator
        ctrl_range to be safe.
        """
        arm = ref_q + self._config.arm_action_scale * jp.tanh(action[:_N_ARM])
        gripper = jp.maximum(action[_N_ARM], 0.0) * self._config.gripper_ctrl_max
        ctrl = jp.concatenate([arm, gripper[None]])
        return jp.clip(ctrl, self._ctrl_lo, self._ctrl_hi)

    # ------------------------------------------------------------------
    # MjxEnv API
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array, trajectory_data: Any = None) -> mjx_env.State:
        # trajectory_data arg is accepted for compatibility with the
        # tennis-env-style VmapWrapper (which passes it through). We store the
        # reference on self in prepare_trajectory(), so trajectory_data is
        # ignored here.
        del trajectory_data
        self._ensure_reference_loaded()

        rng, traj_rng, sub_rng = jax.random.split(rng, 3)

        n_trajs = self._ref_split_points.shape[0] - 1
        traj_no = jax.random.randint(traj_rng, (), 0, n_trajs)
        traj_len = self._ref_length(traj_no)

        # Random subtraj start (always at least 1 step of room before episode end).
        if self._config.random_start:
            sub_t0 = jax.random.randint(
                sub_rng, (), 0, jp.maximum(traj_len - 1, 1)
            )
        else:
            sub_t0 = jp.zeros((), dtype=jp.int32)

        ref_q, ref_v = self._ref_at(traj_no, sub_t0)

        # Build initial qpos/qvel: arm from reference, rest at model defaults.
        init_qpos = self._default_qpos.at[: _N_ARM].set(ref_q)
        init_qvel = jp.zeros(self._mjx_model.nv).at[: _N_ARM].set(ref_v)

        # Initial ctrl matches the residual-action mapping at action=0:
        # arm ⇒ ref_q, gripper ⇒ 0 (open).
        init_ctrl = self._action_to_ctrl(jp.zeros(self.action_size), ref_q)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=init_qpos,
            qvel=init_qvel,
            ctrl=init_ctrl,
            impl=self._impl,
            naconmax=int(self._config.naconmax),
        )

        info = {
            "rng": rng,
            "step": jp.zeros((), dtype=jp.int32),
            "traj_no": traj_no,
            "sub_t0": sub_t0,
            "traj_len": traj_len,
            "last_action": jp.zeros(self.action_size),
        }

        ref_q_next, ref_v_next = self._ref_at(traj_no, sub_t0)
        obs = self._get_obs(data, ref_q_next, ref_v_next, info)

        metrics = {
            f"reward/{k}": jp.zeros(())
            for k in self._config.reward_config.scales.keys()
        }

        reward = jp.zeros(())
        done = jp.zeros(())
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
        trajectory_data: Any = None,
    ) -> mjx_env.State:
        del trajectory_data

        # Look up the reference BEFORE advancing physics so residual ctrl
        # is built against the current target.
        traj_no = state.info["traj_no"]
        sub_t0 = state.info["sub_t0"]
        cur_step = state.info["step"]
        sub_t_cur = jp.minimum(sub_t0 + cur_step, state.info["traj_len"] - 1)
        ref_q_cur, _ = self._ref_at(traj_no, sub_t_cur)

        ctrl = self._action_to_ctrl(action, ref_q_cur)
        data = state.data.replace(ctrl=ctrl)
        # Advance physics by ctrl_dt (= n_substeps * sim_dt). With ctrl_dt=0.02
        # and sim_dt=0.002 this is 10 sub-steps per env.step. Without the loop
        # we'd only advance 2 ms per step and the reference (recorded @ 50 Hz)
        # would be 10x out of phase with sim time.
        n_substeps = int(round(self._config.ctrl_dt / self._config.sim_dt))
        def _one_substep(d, _):
            return mjx.step(self.mjx_model, d), None
        data, _ = jax.lax.scan(_one_substep, data, None, length=n_substeps)

        # Advance reference index for reward / obs.
        next_step = cur_step + 1
        sub_t = jp.minimum(sub_t0 + next_step, state.info["traj_len"] - 1)
        ref_q, ref_v = self._ref_at(traj_no, sub_t)

        obs = self._get_obs(data, ref_q, ref_v, state.info)
        reward, reward_components = self._get_reward(
            data, ref_q, ref_v, action, state.info
        )

        # Termination: arm qpos divergence or NaN.
        arm_qpos = data.qpos[: _N_ARM]
        max_dev = jp.max(jp.abs(arm_qpos - ref_q))
        diverged = max_dev > self._config.termination_qpos_threshold
        is_nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        ref_exhausted = (sub_t0 + next_step) >= state.info["traj_len"]

        done = (diverged | is_nan | ref_exhausted).astype(jp.float32)

        # Apply termination penalty to reward.
        reward = reward + self._config.reward_config.scales.termination * (
            diverged | is_nan
        ).astype(jp.float32)

        info = dict(state.info)
        info["step"] = next_step
        info["last_action"] = action

        metrics = {f"reward/{k}": v for k, v in reward_components.items()}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------
    # Observation / reward
    # ------------------------------------------------------------------

    def _get_obs(
        self,
        data: mjx.Data,
        ref_q: jax.Array,
        ref_v: jax.Array,
        info: Dict[str, Any],
    ) -> Dict[str, jax.Array]:
        arm_qpos = data.qpos[: _N_ARM]
        arm_qvel = data.qvel[: _N_ARM]

        comp = {
            "arm_qpos": arm_qpos,
            "arm_qvel": arm_qvel,
            "ref_arm_qpos": ref_q,
            "ref_arm_qvel": ref_v,
            "dif_arm_qpos": ref_q - arm_qpos,
            "dif_arm_qvel": ref_v - arm_qvel,
            "last_action": info["last_action"],
        }

        state = jp.concatenate([comp[k] for k in self._config.obs_keys])
        privileged = jp.concatenate(
            [comp[k] for k in self._config.privileged_obs_keys]
        )
        state = jp.nan_to_num(state)
        privileged = jp.nan_to_num(privileged)
        return {"state": state, "privileged_state": privileged}

    def _get_reward(
        self,
        data: mjx.Data,
        ref_q: jax.Array,
        ref_v: jax.Array,
        action: jax.Array,
        info: Dict[str, Any],
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        arm_qpos = data.qpos[: _N_ARM]
        arm_qvel = data.qvel[: _N_ARM]

        scales = self._config.reward_config.scales
        sigmas = self._config.reward_config.sigmas

        qpos_err2 = jp.sum((arm_qpos - ref_q) ** 2)
        qvel_err2 = jp.sum((arm_qvel - ref_v) ** 2)
        action_rate2 = jp.sum((action - info["last_action"]) ** 2)

        rewards = {
            "arm_qpos_tracking": jp.exp(-qpos_err2 / (sigmas.arm_qpos ** 2)),
            "arm_qvel_tracking": jp.exp(-qvel_err2 / (sigmas.arm_qvel ** 2)),
            "action_rate": action_rate2,
            "termination": jp.zeros(()),  # filled in step()
        }

        total = sum(scales[k] * rewards[k] for k in rewards.keys())
        return total, rewards

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        # Compute lazily based on obs_keys composition.
        comp_dims = {
            "arm_qpos": _N_ARM,
            "arm_qvel": _N_ARM,
            "ref_arm_qpos": _N_ARM,
            "ref_arm_qvel": _N_ARM,
            "dif_arm_qpos": _N_ARM,
            "dif_arm_qvel": _N_ARM,
            "last_action": int(self.action_size),
        }
        state = sum(comp_dims[k] for k in self._config.obs_keys)
        privileged = sum(comp_dims[k] for k in self._config.privileged_obs_keys)
        return {"state": (state,), "privileged_state": (privileged,)}


lmj.registry.register("MolmoBotTracking", "tracking_config")(
    molmobot_tracking_task_config()
)
