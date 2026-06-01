"""
Train and evaluate baseline scheduling heuristics.

Methods: Greedy SPT, Random, Round-Robin.

Usage:
    python train_baselines.py                      # test all datasets
    python train_baselines.py --data 10x10x3       # specific dataset
    python train_baselines.py --data 6x6x2 --runs 50
"""

import os
import sys
import argparse
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, EnvConfig
from utils import (
    load_csv_data, find_csv_files, get_data_path, set_seed,
)
from environment import JSPEnvironment, GreedyScheduler


def run_greedy(data: Dict, env_config: EnvConfig) -> Dict[str, float]:
    """Run Greedy SPT heuristic (deterministic)."""
    greedy = GreedyScheduler(data, env_config)
    ms, fat, schedule = greedy.solve()
    return {"makespan": ms, "fatigue": fat, "steps": len(schedule)}


def run_random(
    data: Dict, env_config: EnvConfig, num_runs: int = 30
) -> Dict[str, float]:
    """Run Random policy (averaged over multiple runs)."""
    makespans, fatigues = [], []
    env = JSPEnvironment(data, env_config)

    for _ in range(num_runs):
        env.reset()
        done = False
        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            _, _, done, _ = env.step(action)
        makespans.append(env.get_makespan())
        fatigues.append(env.get_avg_fatigue())

    return {
        "makespan_mean": np.mean(makespans),
        "makespan_std": np.std(makespans),
        "fatigue_mean": np.mean(fatigues),
        "fatigue_std": np.std(fatigues),
    }


def run_roundrobin(data: Dict, env_config: EnvConfig) -> Dict[str, float]:
    """Run Round-Robin worker assignment (deterministic given fixed seed)."""
    env = JSPEnvironment(data, env_config)
    env.reset()
    done = False
    rr_worker = 0
    steps = 0

    while not done:
        mask = env._get_action_mask()
        if not np.any(mask):
            break
        valid = np.where(mask)[0]
        # Prefer actions for the current round-robin worker
        rr_valid = [a for a in valid if a % env.num_workers == rr_worker]
        action = int(np.random.choice(rr_valid if rr_valid else valid))
        _, _, done, _ = env.step(action)
        rr_worker = (rr_worker + 1) % env.num_workers
        steps += 1

    return {
        "makespan": env.get_makespan(),
        "fatigue": env.get_avg_fatigue(),
        "steps": steps,
    }


def evaluate_dataset(data_path: str, config: Config, random_runs: int = 30):
    """Evaluate all baselines on a single dataset."""
    data = load_csv_data(data_path)
    N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
    name = f"{N}x{M}x{W}"

    print(f"\n{'='*55}")
    print(f"Dataset: {name}  ({N} jobs, {M} machines, {W} workers)")
    print(f"{'='*55}")

    t0 = time.time()

    # Greedy SPT
    greedy = run_greedy(data, config.env)
    print(f"Greedy SPT:      Makespan={greedy['makespan']:>10.1f}  "
          f"Fatigue={greedy['fatigue']:>8.3f}  Steps={greedy['steps']}")

    # Round-Robin
    rr = run_roundrobin(data, config.env)
    print(f"Round-Robin:     Makespan={rr['makespan']:>10.1f}  "
          f"Fatigue={rr['fatigue']:>8.3f}  Steps={rr['steps']}")

    # Random
    rand = run_random(data, config.env, num_runs=random_runs)
    print(f"Random ({random_runs}runs): Makespan={rand['makespan_mean']:>10.1f}±"
          f"{rand['makespan_std']:.0f}  Fatigue={rand['fatigue_mean']:>8.3f}±"
          f"{rand['fatigue_std']:.3f}")

    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    return {
        "dataset": name,
        "greedy": greedy,
        "roundrobin": rr,
        "random": rand,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSP baseline scheduling heuristics"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name (e.g. 6x6x2). Default: all CSV files."
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of runs for Random baseline (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    config = Config()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    # Determine datasets
    if args.data:
        data_files = [args.data]
    else:
        all_files = find_csv_files(data_dir)
        data_files = [os.path.basename(f).replace(".csv", "") for f in all_files]
        # Limit to representative sizes for display
        if len(data_files) > 6:
            data_files = data_files[:6]

    results = {}
    total_start = time.time()

    for data_file in data_files:
        data_path = get_data_path(data_dir, data_file)
        results[data_file] = evaluate_dataset(data_path, config, args.runs)

    # Summary
    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    print(f"{'Dataset':<12} {'Greedy MS':>10} {'RR MS':>10} {'Random MS':>12}  "
          f"{'Greedy F':>8} {'RR F':>8} {'Random F':>8}")
    print("-" * 72)
    for data_file, res in results.items():
        g, r, rr = res['greedy'], res['random'], res['roundrobin']
        print(
            f"{res['dataset']:<12} {g['makespan']:>10.1f} {rr['makespan']:>10.1f} "
            f"{r['makespan_mean']:>10.1f}±{r['makespan_std']:.0f}  "
            f"{g['fatigue']:>8.3f} {rr['fatigue']:>8.3f} {r['fatigue_mean']:>8.3f}"
        )

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
