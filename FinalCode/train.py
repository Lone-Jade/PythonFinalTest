import argparse
import json
import random
from pathlib import Path

import numpy as np

from agents import DQNAgent, PPOAgent, Transition
from config import EnvConfig, TrainConfig
from data_loader import load_instances
from env import JobShopFatigueEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_env(instance, seed):
    return JobShopFatigueEnv(instance, EnvConfig(), seed=seed)


def train_dqn(instances, cfg, out_dir):
    sample_env = make_env(instances[0], cfg.seed)
    agent = DQNAgent(sample_env.feature_dim, cfg)
    logs = []
    live_log = out_dir / "train_log_live.jsonl"
    live_log.write_text("", encoding="utf-8")

    for ep in range(1, cfg.episodes + 1):
        inst = instances[(ep - 1) % len(instances)]
        env = make_env(inst, cfg.seed + ep)
        obs = env.reset()
        total = 0.0
        losses = []
        decisions = 0
        while not obs["done"] and decisions < cfg.max_decisions:
            action = agent.act(obs, explore=True)
            features = obs["features"].copy()
            next_obs, reward, done, _ = env.step(action)
            agent.remember(
                Transition(
                    features=features,
                    action=action,
                    reward=reward,
                    next_features=next_obs["features"].copy(),
                    next_mask=next_obs["mask"].copy(),
                    done=done,
                )
            )
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total += reward
            decisions += 1

        row = {
            "episode": ep,
            "instance": inst.name,
            "reward": total,
            "makespan": env.time,
            "decisions": decisions,
            "epsilon": agent.epsilon,
            "loss": float(np.mean(losses)) if losses else None,
            "force_rest_time": int(env.force_rest_time.sum()),
            "invalid_actions": env.invalid_actions,
        }
        logs.append(row)
        with live_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if ep == 1 or ep % 10 == 0:
            print(json.dumps(row, ensure_ascii=False))

    agent.save(out_dir / "dqn_model.pt")
    return logs


def train_ppo(instances, cfg, out_dir):
    sample_env = make_env(instances[0], cfg.seed)
    agent = PPOAgent(sample_env.feature_dim, cfg)
    logs = []
    live_log = out_dir / "train_log_live.jsonl"
    live_log.write_text("", encoding="utf-8")

    for ep in range(1, cfg.episodes + 1):
        inst = instances[(ep - 1) % len(instances)]
        env = make_env(inst, cfg.seed + ep)
        obs = env.reset()
        rollout = []
        total = 0.0
        decisions = 0

        while not obs["done"] and decisions < cfg.max_decisions:
            action, logp, value = agent.act(obs, explore=True)
            item = {
                "features": obs["features"].copy(),
                "mask": obs["mask"].copy(),
                "action": action,
                "logp": logp,
                "value": value,
            }
            next_obs, reward, done, _ = env.step(action)
            item["reward"] = reward
            item["done"] = done
            rollout.append(item)
            obs = next_obs
            total += reward
            decisions += 1

            if len(rollout) >= cfg.rollout_steps:
                agent.update(rollout)
                rollout = []

        loss = agent.update(rollout) if rollout else None
        row = {
            "episode": ep,
            "instance": inst.name,
            "reward": total,
            "makespan": env.time,
            "decisions": decisions,
            "loss": loss,
            "force_rest_time": int(env.force_rest_time.sum()),
            "invalid_actions": env.invalid_actions,
        }
        logs.append(row)
        with live_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if ep == 1 or ep % 10 == 0:
            print(json.dumps(row, ensure_ascii=False))

    agent.save(out_dir / "ppo_model.pt")
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="basic_data.xlsx")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--instances", nargs="*", default=["6x6_6x6x3", "10x5_10x5x3"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    all_instances = load_instances(args.data)
    selected = []
    for name in args.instances:
        if name not in all_instances:
            raise KeyError(f"Unknown instance {name!r}. Available: {list(all_instances)[:10]} ...")
        selected.append(all_instances[name])

    out_dir = Path(args.out) / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed)

    if args.algorithm == "dqn":
        logs = train_dqn(selected, cfg, out_dir)
    else:
        logs = train_ppo(selected, cfg, out_dir)

    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
