import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from config import EnvConfig
from data_loader import load_instances
from env import JobShopFatigueEnv
from heuristics import run_heuristic, select_action
from test import choose_dqn_action, choose_ppo_action, load_model


def plot_gantt(history, n_machines, out_path):
    fig, ax = plt.subplots(figsize=(12, max(4, n_machines * 0.45)))
    colors = plt.cm.tab20.colors
    for task in history:
        y = task.machine
        ax.barh(
            y,
            task.finish - task.start,
            left=task.start,
            color=colors[task.worker % len(colors)],
            edgecolor="black",
            height=0.75,
        )
        ax.text(
            task.start + (task.finish - task.start) / 2,
            y,
            f"J{task.job + 1}-O{task.op + 1}/W{task.worker + 1}",
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{i + 1}" for i in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title("Machine Gantt Chart")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fatigue(history, n_workers, out_path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for worker in range(n_workers):
        xs = [0]
        ys = [0]
        for task in history:
            if task.worker != worker:
                continue
            xs.extend([task.start, task.finish])
            ys.extend([task.fatigue_before, task.fatigue_after])
        ax.step(xs, ys, where="post", label=f"W{worker + 1}")
    ax.axhline(0.8, linestyle="--", color="red", linewidth=1, label="F_force")
    ax.axhline(0.5, linestyle=":", color="green", linewidth=1, label="F_resume")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fatigue")
    ax.set_title("Worker Fatigue Curve")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="basic_data.xlsx")
    parser.add_argument("--instance", default="6x6_6x6x3")
    parser.add_argument("--algorithm", default="heuristic", choices=["heuristic", "dqn", "ppo"])
    parser.add_argument("--policy", default="rest_aware", choices=["spt", "rest_aware", "random", "fatigue"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-decisions", type=int, default=100000)
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()

    inst = load_instances(args.data)[args.instance]
    env = JobShopFatigueEnv(inst, EnvConfig(), seed=42)

    if args.algorithm == "heuristic":
        result = run_heuristic(env, policy=args.policy, max_decisions=args.max_decisions)
        label = args.policy
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --algorithm is dqn or ppo")
        model, device = load_model(args.algorithm, Path(args.model_path), env.feature_dim, args.hidden_dim)
        obs = env.reset()
        total_reward = 0.0
        decisions = 0
        no_progress = 0
        last_tasks = 0
        while not obs["done"] and decisions < args.max_decisions:
            if args.algorithm == "dqn":
                action = choose_dqn_action(model, obs, device)
            else:
                action = choose_ppo_action(model, obs, device)

            # Keep plotting robust even when an under-trained model repeatedly waits.
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or no_progress >= 20:
                action = select_action(obs, policy="rest_aware")

            obs, reward, _, _ = env.step(action)
            total_reward += reward
            decisions += 1
            if len(env.history) == last_tasks:
                no_progress += 1
            else:
                no_progress = 0
                last_tasks = len(env.history)

        result = {
            "policy": args.algorithm,
            "reward": total_reward,
            "makespan": env.time,
            "decisions": decisions,
            "force_rest_time": int(env.force_rest_time.sum()),
            "invalid_actions": env.invalid_actions,
        }
        label = f"{args.algorithm}_{Path(args.model_path).stem}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_gantt(env.history, inst.n_machines, out_dir / f"{inst.name}_{label}_gantt.png")
    plot_fatigue(env.history, inst.n_workers, out_dir / f"{inst.name}_{label}_fatigue.png")
    print(
        {
            "policy": result["policy"],
            "makespan": result["makespan"],
            "decisions": result["decisions"],
            "force_rest_time": result["force_rest_time"],
            "invalid_actions": result["invalid_actions"],
        }
    )


if __name__ == "__main__":
    main()
