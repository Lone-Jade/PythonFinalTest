import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import EnvConfig, TrainConfig
from data_loader import load_instances
from env import JobShopFatigueEnv
from heuristics import select_action
from models import ActorCriticNetwork, PairScoringNetwork


def load_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Testing trained DQN/PPO models requires PyTorch.") from exc
    return torch


def env_metrics(env, total_reward, decisions, algorithm, model_name):
    fatigue_after = [float(task.fatigue_after) for task in env.history]
    return {
        "algorithm": algorithm,
        "model": model_name,
        "instance": env.instance.name,
        "makespan": int(env.time),
        "reward": float(total_reward),
        "decisions": int(decisions),
        "tasks": len(env.history),
        "force_rest_time": int(env.force_rest_time.sum()),
        "invalid_actions": int(env.invalid_actions),
        "mean_task_fatigue": float(np.mean(fatigue_after)) if fatigue_after else 0.0,
        "max_task_fatigue": float(np.max(fatigue_after)) if fatigue_after else 0.0,
    }


def choose_dqn_action(model, obs, device):
    torch = load_torch()
    if len(obs["features"]) == 0 or not obs["mask"].any():
        return 0
    with torch.no_grad():
        features = torch.tensor(obs["features"], dtype=torch.float32, device=device)
        q_values = model(features).detach().cpu().numpy()
    q_values[~obs["mask"]] = -1e9
    return int(np.argmax(q_values))


def choose_ppo_action(model, obs, device):
    torch = load_torch()
    if len(obs["features"]) == 0 or not obs["mask"].any():
        return 0
    with torch.no_grad():
        features = torch.tensor(obs["features"], dtype=torch.float32, device=device)
        mask = torch.tensor(obs["mask"], dtype=torch.bool, device=device)
        logits, _ = model(features, mask)
    return int(torch.argmax(logits).item())


def evaluate_policy(env, action_fn, max_decisions, fallback_fn=None, patience=20):
    obs = env.reset()
    total_reward = 0.0
    decisions = 0
    no_progress = 0
    last_tasks = 0
    while not obs["done"] and decisions < max_decisions:
        action = action_fn(obs)
        if fallback_fn is not None:
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action]:
                action = fallback_fn(obs)
            if no_progress >= patience:
                action = fallback_fn(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        decisions += 1
        if len(env.history) == last_tasks:
            no_progress += 1
        else:
            no_progress = 0
            last_tasks = len(env.history)
    return total_reward, decisions


def load_model(algorithm, model_path, feature_dim, hidden_dim):
    torch = load_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    if algorithm == "dqn":
        model = PairScoringNetwork(feature_dim, hidden_dim).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, device
    if algorithm == "ppo":
        model = ActorCriticNetwork(feature_dim, hidden_dim).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, device
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="basic_data.xlsx")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "heuristic"], default="heuristic")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--heuristic", default="rest_aware", choices=["spt", "rest_aware", "random", "fatigue"])
    parser.add_argument(
        "--test-instances",
        nargs="*",
        default=["10x5_10x5x3", "15x5_15x5x3", "10x10_10x10x3"],
    )
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--out", default="outputs/test_results")
    args = parser.parse_args()

    instances = load_instances(args.data)
    selected = []
    for name in args.test_instances:
        if name not in instances:
            raise KeyError(f"Unknown test instance {name!r}. Available examples: {list(instances)[:10]}")
        selected.append(instances[name])

    model = None
    device = None
    model_name = args.heuristic
    if args.algorithm in ("dqn", "ppo"):
        if not args.model_path:
            raise ValueError("--model-path is required when --algorithm is dqn or ppo")
        probe_env = JobShopFatigueEnv(selected[0], EnvConfig(), seed=0)
        model, device = load_model(args.algorithm, args.model_path, probe_env.feature_dim, args.hidden_dim)
        model_name = Path(args.model_path).name

    rows = []
    for idx, inst in enumerate(selected):
        env = JobShopFatigueEnv(inst, EnvConfig(), seed=1000 + idx)
        if args.algorithm == "heuristic":
            action_fn = lambda obs, policy=args.heuristic: select_action(obs, policy=policy)
        elif args.algorithm == "dqn":
            action_fn = lambda obs, m=model, d=device: choose_dqn_action(m, obs, d)
        else:
            action_fn = lambda obs, m=model, d=device: choose_ppo_action(m, obs, d)

        fallback_fn = None
        if args.algorithm in ("dqn", "ppo"):
            fallback_fn = lambda obs: select_action(obs, policy="rest_aware")
        total_reward, decisions = evaluate_policy(env, action_fn, args.max_decisions, fallback_fn=fallback_fn)
        row = env_metrics(env, total_reward, decisions, args.algorithm, model_name)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.algorithm}_{model_name}_results.json"
    csv_path = out_dir / f"{args.algorithm}_{model_name}_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
