"""Streaming loader for allenai/molmobot-data on HuggingFace.

Pulls one outer-tar shard at a time from HF, extracts its per-house
tar.zst archives → H5s → (s_t, s_{t+1}, a_t) triples, all with a bounded
disk footprint (~2 GB/shard + optional H5 temp dir).

Design goals:
  - Bounded disk: each shard is downloaded to HF cache, untarred to a temp
    dir, consumed, then temp dir is wiped. HF cache cleanup is left to the
    user (set HF_HUB_CACHE= if you need to cap it).
  - Bounded memory: decompress one house at a time, not the full shard.
  - Integrate with the existing loader: after per-shard extraction we call
    the same `load_h5_reference_dataset(tmpdir)` used by the rest of the
    codebase, so H5 decode stays in one place.

Usage:
    ds = StreamingMolmobotDataset(
        repo_id="allenai/molmobot-data",
        datagen_config="FrankaPickOmniCamConfig",
        split="train",
        max_shards=None,      # None = all shards
    )
    for batch in ds.iter_batches(batch_size=8192, shuffle_buffer=50000, seed=0):
        s_t, s_goal, a_t = batch  # numpy float32 arrays
        ...
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download


_N_ARM = 7


@dataclass
class StreamingMolmobotDataset:
    repo_id: str = "allenai/molmobot-data"
    datagen_config: str = "FrankaPickOmniCamConfig"
    split: str = "train"  # train | val | test (maps to <config>/<split>_shards/)
    max_shards: Optional[int] = None
    """If set, stop after this many outer tar shards (for debug / small runs)."""
    shuffle_shards: bool = True
    """Randomize shard iteration order each epoch."""
    hf_cache_dir: Optional[str] = None
    """If set, override HF hub cache location."""

    def list_shards(self) -> list[str]:
        """Return the HF-relative paths of all outer shard tars for this split."""
        api = HfApi()
        all_files = api.list_repo_files(self.repo_id, repo_type="dataset")
        prefix = f"{self.datagen_config}/{self.split}_shards/"
        shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
        if not shards:
            raise ValueError(
                f"No shards found on {self.repo_id} matching {prefix}*.tar"
            )
        if self.max_shards is not None:
            shards = shards[: self.max_shards]
        return shards

    def iter_shard_extract_dirs(self, seed: int = 0) -> Iterator[Path]:
        """Generator over locally-extracted shard directories.

        Yields a Path to a temp dir containing `house_<N>/*.h5`. Cleans up
        the temp dir after the consumer is done with it (i.e. on `next()`).
        """
        shards = self.list_shards()
        if self.shuffle_shards:
            rng = np.random.default_rng(seed)
            shards = [shards[i] for i in rng.permutation(len(shards))]

        dctx = zstd.ZstdDecompressor()
        for shard in shards:
            print(f"  [stream] hf_hub_download {shard}")
            shard_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=shard,
                repo_type="dataset",
                cache_dir=self.hf_cache_dir,
            )
            tmpdir = Path(tempfile.mkdtemp(prefix="molmobot_shard_"))
            try:
                with tarfile.open(shard_path, "r") as outer:
                    members = [
                        m for m in outer.getmembers()
                        if m.name.startswith(f"{self.datagen_config}_house_")
                        and m.name.endswith(".tar.zst")
                    ]
                    print(f"  [stream] extracting {len(members)} houses -> {tmpdir}")
                    for m in members:
                        with outer.extractfile(m) as zst_stream:
                            zst_bytes = zst_stream.read()
                        with dctx.stream_reader(io.BytesIO(zst_bytes)) as reader:
                            with tarfile.open(fileobj=reader, mode="r|") as inner:
                                inner.extractall(tmpdir)
                yield tmpdir
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def iter_triples(self, seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Stream (s_t, s_goal, a_t) sample arrays per shard.

        Each yield is (s_t, s_goal, a_t) for ONE entire shard's worth of
        triples (order-preserving within the shard, no cross-shard
        shuffling). Consumer is expected to shuffle+batch downstream.
        """
        from latent_mj.envs.molmobot_manipulation.train.molmobot_traj_loader import (
            load_h5_reference_dataset,
        )

        for shard_dir in self.iter_shard_extract_dirs(seed=seed):
            traj = load_h5_reference_dataset(shard_dir)
            qpos = np.asarray(traj.data.qpos[:, :_N_ARM])
            qvel = np.asarray(traj.data.qvel[:, :_N_ARM])
            actions = np.asarray(traj.info.metadata["recorded_actions"])
            split_points = np.asarray(traj.data.split_points)

            s_t_chunks: list[np.ndarray] = []
            s_goal_chunks: list[np.ndarray] = []
            a_t_chunks: list[np.ndarray] = []
            for i in range(len(split_points) - 1):
                start, end = int(split_points[i]), int(split_points[i + 1])
                if end - start < 2:
                    continue
                s = np.concatenate([qpos[start:end], qvel[start:end]], axis=-1)
                g = qpos[start:end]
                a = actions[start:end]
                s_t_chunks.append(s[:-1])
                s_goal_chunks.append(g[1:])
                a_t_chunks.append(a[:-1])

            if not s_t_chunks:
                continue
            yield (
                np.concatenate(s_t_chunks, axis=0).astype(np.float32),
                np.concatenate(s_goal_chunks, axis=0).astype(np.float32),
                np.concatenate(a_t_chunks, axis=0).astype(np.float32),
            )

    def iter_batches(
        self,
        batch_size: int,
        shuffle_buffer: int = 50_000,
        seed: int = 0,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield random-order (s_t, s_goal, a_t) batches using a reservoir buffer.

        Buffer is refilled from the shard stream as it drains, so the order
        is approximately uniform but not perfectly global-random.
        """
        rng = np.random.default_rng(seed)

        def _take(buf_s, buf_g, buf_a, k):
            n = buf_s.shape[0]
            idx = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            return (buf_s[idx], buf_g[idx], buf_a[idx],
                    buf_s[keep], buf_g[keep], buf_a[keep])

        buf_s = np.zeros((0, 14), dtype=np.float32)
        buf_g = np.zeros((0, _N_ARM), dtype=np.float32)
        buf_a = np.zeros((0, 8), dtype=np.float32)

        shard_iter = self.iter_triples(seed=seed)
        exhausted = False

        while True:
            # Top up buffer until it passes shuffle_buffer or shards run out.
            while buf_s.shape[0] < shuffle_buffer and not exhausted:
                try:
                    s, g, a = next(shard_iter)
                    buf_s = np.concatenate([buf_s, s], axis=0)
                    buf_g = np.concatenate([buf_g, g], axis=0)
                    buf_a = np.concatenate([buf_a, a], axis=0)
                except StopIteration:
                    exhausted = True
                    break

            if buf_s.shape[0] == 0:
                return

            k = min(batch_size, buf_s.shape[0])
            out_s, out_g, out_a, buf_s, buf_g, buf_a = _take(buf_s, buf_g, buf_a, k)
            yield out_s, out_g, out_a
