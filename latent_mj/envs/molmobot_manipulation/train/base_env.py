"""Base class for MolmoBot manipulation environments (mujoco_warp backend)."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


class MolmoBotEnv(mjx_env.MjxEnv):
    """Base class for MolmoBot manipulation environments.

    Uses NVIDIA's mujoco_warp backend by default (``impl="warp"``), which
    provides GPU-accelerated physics via ``mjx.put_model(m, impl="warp")``.

    Sensor helpers are intentionally omitted here — Franka FR3 has different
    sensors than G1 and those will be added per-task in subclasses.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        xml_path: Optional[str] = None,
        mj_model: Optional[mujoco.MjModel] = None,
        impl: str = "warp",
    ) -> None:
        if (xml_path is None) == (mj_model is None):
            raise ValueError("Exactly one of xml_path or mj_model must be set")

        super().__init__(config, config_overrides)

        self._mj_model = mj_model if mj_model is not None else mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._impl = impl
        # The warp backend requires JAX to operate in float32 mode.  Some
        # upstream imports (e.g. molmo_spaces unpickling via h5py) silently
        # enable jax_enable_x64, which causes mjx.put_model to produce float64
        # arrays and then fails inside the warp FFI layer with a dtype error on
        # opt__density.  Restore float32 mode immediately before put_model.
        import jax as _jax
        _jax.config.update("jax_enable_x64", False)
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._xml_path = xml_path  # may be None when constructed from mj_model

    # Accessors.

    @property
    def xml_path(self) -> Optional[str]:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def impl(self) -> str:
        return self._impl
