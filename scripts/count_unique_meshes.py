"""Count unique objaverse meshes used across an extracted molmobot-data dir.

Walks every *.h5 under --data-dir, decodes obs_scene SavedEpisode pickles,
collects task_config.object_poses, and tallies unique 32-char objaverse UIDs.

Outputs:
  - total trajectories scanned
  - unique mesh UIDs (what the "mega-scene" slot count would need to be)
  - instances (sum across all trajs)
  - avg / max / p95 objects per trajectory
  - top-N most-used UIDs
  - histogram of how many trajs each UID appears in

Usage:
    uv run python scripts/count_unique_meshes.py \\
        --data-dir storage/molmobot_data_sample/extracted \\
        --top 20
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("GLI_PATH", "/tmp")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="storage/molmobot_data_sample/extracted")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    from latent_mj.envs.molmobot_manipulation.train.molmobot_data_loader import (
        decode_obs_scene,
    )

    data_dir = Path(args.data_dir)
    h5_files = sorted(data_dir.rglob("*.h5"))
    print(f"scanning {len(h5_files)} H5 files under {data_dir}")

    uid_re = re.compile(r"[0-9a-f]{32}")
    uid_instances = Counter()      # total instances (incl duplicates)
    uid_trajs = Counter()          # number of distinct trajs each UID appears in
    per_traj_counts: list[int] = []
    type_counter = Counter()
    n_trajs = 0
    skipped: list[str] = []

    for f_path in tqdm(h5_files, desc="files"):
        try:
            h5f = h5py.File(f_path, "r")
        except Exception as e:
            skipped.append(f"{f_path} (open failed: {e})")
            continue
        try:
            for key in sorted(k for k in h5f.keys() if k.startswith("traj_")):
                try:
                    _, saved_episode = decode_obs_scene(h5f[key]["obs_scene"][()])
                    poses = saved_episode.task_config.object_poses
                    traj_uids = set()
                    n_objs = 0
                    for name in poses:
                        if not name.startswith("obja"):
                            continue
                        m = uid_re.search(name)
                        if not m:
                            continue
                        uid = m.group()
                        uid_instances[uid] += 1
                        traj_uids.add(uid)
                        type_counter[name.split("_", 1)[0]] += 1
                        n_objs += 1
                    for uid in traj_uids:
                        uid_trajs[uid] += 1
                    per_traj_counts.append(n_objs)
                    n_trajs += 1
                except Exception as e:
                    skipped.append(f"{f_path}:{key} ({type(e).__name__}: {e})")
        finally:
            h5f.close()

    print()
    print(f"=== dataset object inventory ===")
    print(f"files scanned:                {len(h5_files)}")
    print(f"trajectories:                 {n_trajs}")
    print(f"unique objaverse UIDs:        {len(uid_instances)}")
    print(f"total obja instances:         {sum(uid_instances.values())}")
    if per_traj_counts:
        counts = np.asarray(per_traj_counts)
        print(f"objects / trajectory:         mean={counts.mean():.1f}  "
              f"median={int(np.median(counts))}  max={counts.max()}  "
              f"p95={int(np.percentile(counts, 95))}")
    print(f"unique obja-type prefixes:    {len(type_counter)}")
    if skipped:
        print(f"(skipped {len(skipped)} entries)")
        for s in skipped[:5]:
            print(f"  {s}")

    print(f"\n=== top-{args.top} UIDs by instance count ===")
    for uid, c in uid_instances.most_common(args.top):
        n_traj = uid_trajs[uid]
        print(f"  {uid}  instances={c:>6}  trajs={n_traj:>5}")

    print(f"\n=== top-{args.top} type prefixes ===")
    for t, c in type_counter.most_common(args.top):
        print(f"  {t}  instances={c}")

    # UID trajectory-coverage histogram
    if uid_trajs:
        bins = [1, 2, 5, 10, 25, 50, 100, 500, 1000, 10000]
        print(f"\n=== UID trajectory-coverage histogram (#UIDs appearing in ≥N trajs) ===")
        for b in bins:
            n = sum(1 for c in uid_trajs.values() if c >= b)
            print(f"  ≥{b:>5} trajs: {n} UIDs")


if __name__ == "__main__":
    main()
