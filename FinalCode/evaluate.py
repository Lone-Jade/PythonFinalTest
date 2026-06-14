import argparse
import json
from pathlib import Path

import numpy as np

from config import EnvConfig
from data_loader import list_instances, load_instances
from env import JobShopFatigueEnv
from heuristics import run_heuristic


def summarize_env(env):
    fatigue_after = [float(x.fatigue_after) for x in env.history]
    return {
        "makespan": env.time,
        "tasks": len(env.history),
        "force_rest_time": int(env.force_rest_time.sum()),
        "max_task_fatigue": max(fatigue_after) if fatigue_after else 0.0,
        "mean_task_fatigue": float(np.mean(fatigue_after)) if fatigue_after else 0.0,
        "invalid_actions": env.invalid_actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="basic_data.xlsx")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--instance", default="6x6_6x6x3")
    parser.add_argument("--policies", nargs="*", default=["spt", "rest_aware", "random"])
    parser.add_argument("--out", default="outputs/eval")
    args = parser.parse_args()

    if args.list:
        for item in list_instances(args.data):
            print(item)
        return

    instances = load_instances(args.data)
    inst = instances[args.instance]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in args.policies:
        env = JobShopFatigueEnv(inst, EnvConfig(), seed=42)
        result = run_heuristic(env, policy=policy)
        row = {"instance": inst.name, "policy": policy, **summarize_env(env), "reward": result["reward"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (out_dir / f"{inst.name}_heuristics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
