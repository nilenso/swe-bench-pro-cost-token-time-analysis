#!/usr/bin/env python3
"""
Walk every trajectory under data/<model>/traj/, classify each step's
failure mode, and write aggregated stats to data/failure_modes.json.

Output schema:
    {
      "models": ["claude45", "gemini25pro", ...],
      "per_model": {
        "<model>": {
          "n_trajectories": int,
          "n_steps": int,
          "n_failures": int,
          "mode_counts": {mode: int, ...},
          "trajectories_with_mode": {mode: int, ...},
          "samples": {mode: [{"instance": "...", "action": "...",
                              "observation": "..."}, ...]}
        }, ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.failure_modes import (   # noqa: E402
    FAILURE_MODES, classify_failure, analyze_trajectory,
)
from analysis.models import MODELS     # noqa: E402

SAMPLES_PER_MODE = 4   # how many illustrative samples to collect per mode/model


def _process_one(args):
    model, path = args
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None
    traj = data.get("trajectory") or []
    if not traj:
        return None

    instance = Path(path).stem
    mode_counts: Counter = Counter()
    samples: dict[str, list] = defaultdict(list)
    n_failures = 0
    for step in traj:
        action = step.get("action") or ""
        obs = step.get("observation") or ""
        mode = classify_failure(action, obs)
        if mode:
            mode_counts[mode] += 1
            n_failures += 1
            if len(samples[mode]) < SAMPLES_PER_MODE:
                samples[mode].append({
                    "instance": instance,
                    "action": action[:600],
                    "observation": obs[:800],
                })

    return {
        "model": model,
        "instance": instance,
        "n_steps": len(traj),
        "n_failures": n_failures,
        "mode_counts": dict(mode_counts),
        "samples": {k: v for k, v in samples.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("-o", "--output", default="data/failure_modes.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    tasks: list[tuple[str, str]] = []
    for model in MODELS:
        base = data_root / model / "traj"
        if not base.exists():
            continue
        for p in sorted(base.glob("*/*.traj")):
            tasks.append((model, str(p)))

    print(f"Processing {len(tasks)} trajectories...")
    per_model: dict[str, dict] = defaultdict(lambda: {
        "n_trajectories": 0,
        "n_steps": 0,
        "n_failures": 0,
        "mode_counts": Counter(),
        "trajectories_with_mode": Counter(),
        "samples": defaultdict(list),
    })

    workers = min(8, os.cpu_count() or 1)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_process_one, tasks, chunksize=16):
            done += 1
            if r is None:
                continue
            m = r["model"]
            mm = per_model[m]
            mm["n_trajectories"] += 1
            mm["n_steps"] += r["n_steps"]
            mm["n_failures"] += r["n_failures"]
            for mode, cnt in r["mode_counts"].items():
                mm["mode_counts"][mode] += cnt
                mm["trajectories_with_mode"][mode] += 1
            # collect a few diverse samples per mode (one per instance)
            for mode, samps in r["samples"].items():
                bucket = mm["samples"][mode]
                if len(bucket) < SAMPLES_PER_MODE * 4:
                    bucket.extend(samps[:1])
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} processed")

    # Convert Counters / defaultdicts to plain dicts and trim sample lists
    out_per_model = {}
    for m, mm in per_model.items():
        # Dedup samples per mode by instance and trim
        samples_clean: dict[str, list] = {}
        for mode, samps in mm["samples"].items():
            seen = set()
            uniq = []
            for s in samps:
                if s["instance"] in seen:
                    continue
                seen.add(s["instance"])
                uniq.append(s)
                if len(uniq) >= SAMPLES_PER_MODE:
                    break
            samples_clean[mode] = uniq
        out_per_model[m] = {
            "n_trajectories": mm["n_trajectories"],
            "n_steps": mm["n_steps"],
            "n_failures": mm["n_failures"],
            "mode_counts": dict(mm["mode_counts"]),
            "trajectories_with_mode": dict(mm["trajectories_with_mode"]),
            "samples": samples_clean,
        }

    output = {
        "models": sorted(out_per_model.keys()),
        "modes": [{"key": k, "family": f, "label": lbl, "desc": d}
                  for k, f, lbl, d in FAILURE_MODES],
        "per_model": out_per_model,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Summary printout
    print(f"\nWrote {out_path}")
    for m in sorted(out_per_model.keys()):
        mm = out_per_model[m]
        rate = mm["n_failures"] / max(mm["n_steps"], 1) * 100
        print(f"  {m}: {mm['n_failures']}/{mm['n_steps']} "
              f"failure steps ({rate:.1f}%) across {mm['n_trajectories']} trajs")


if __name__ == "__main__":
    main()
