"""jax 0.10 compat shims for code that still calls removed APIs.

jax 0.10 dropped ``jax.device_put_replicated``; brax 0.14.2 (and our forked
PPO trainer in ``latent_mj.learning.policy.ppo.train_tracking``) still call
it. We restore a working drop-in by stacking each leaf along a new device
axis and putting it on a 1-D mesh of the local devices via
``NamedSharding``. ``jax.pmap`` consumes NamedSharding-replicated arrays
in jax 0.10, so this preserves multi-GPU pmap-based training (1, 4, 8 ...
devices all behave identically to the old API).

Importing this module is idempotent and a no-op once the upstream API is
restored or once brax migrates to ``shard_map``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _install_device_put_replicated() -> None:
    if hasattr(jax, "device_put_replicated"):
        return

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    def device_put_replicated(value, devices):
        n = len(devices)
        mesh = Mesh(np.asarray(devices), ("d",))
        leaves, treedef = jax.tree_util.tree_flatten(value)
        out = []
        for leaf in leaves:
            arr = jnp.asarray(leaf)
            stacked = jnp.broadcast_to(jnp.expand_dims(arr, 0), (n,) + arr.shape)
            sharding = NamedSharding(mesh, PartitionSpec("d", *([None] * arr.ndim)))
            out.append(jax.device_put(stacked, sharding))
        return jax.tree_util.tree_unflatten(treedef, out)

    jax.device_put_replicated = device_put_replicated  # type: ignore[attr-defined]


_install_device_put_replicated()
