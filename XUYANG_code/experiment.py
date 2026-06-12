"""One-click experiment runner for VSCode.

Edit the constants in the "Experiment settings" section, then run this file.
It will train the selected algorithms, evaluate on validation/test instances,
and save JSON/CSV result files under OUTPUT_DIR.

Supports resuming from checkpoints:
  Set RESUME_DQN / RESUME_PPO to a .pt checkpoint path, or leave as None
  for fresh training.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from agents import DQNAgent, PPOAgent, Transition
from config import EnvConfig, TrainConfig
from data_loader import load_instances
from env import JobShopFatigueEnv
from heuristics import select_action
from test import choose_dqn_action, choose_ppo_action, env_metrics, load_model


# =========================
# Experiment settings
# =========================

DATA_PATH = "basic_data.xlsx"
OUTPUT_DIR = Path("outputs_exp")

TRAIN_INSTANCES = [
    # Small
    "6x6_6x6x2", "6x6_6x6x3",
    # Medium-tall
    "10x5_10x5x2", "10x5_10x5x3",
    "15x5_15x5x2", "15x5_15x5x3",
    # Medium-square
    "10x10_10x10x2", "10x10_10x10x3", "10x10_10x10x4",
    # Medium-large
    "15x10_15x10x2", "15x10_15x10x3", "15x10_15x10x4",
    # Large-tall
    "20x5_20x5x2", "20x5_20x5x3",
]

VAL_INSTANCES = [
    "20x10_20x10x2", "20x10_20x10x3",
    "30x5_30x5x2", "30x5_30x5x3",
    "30x10_30x10x2", "30x10_30x10x3",
]

TEST_INSTANCES = [
    # Small (seen in train but different workers)
    "6x6_6x6x2",
    # Medium-small
    "10x5_10x5x3",
    # Medium-square (different workers from train)
    "10x10_10x10x2", "10x10_10x10x3", "10x10_10x10x5", "10x10_10x10x6",
    # Medium-large
    "15x5_15x5x3",
    "15x10_15x10x2", "15x10_15x10x3", "15x10_15x10x5", "15x10_15x10x6",
    # Large
    "20x5_20x5x3",
    "20x10_20x10x2", "20x10_20x10x3", "20x10_20x10x5", "20x10_20x10x6",
    # Very large
    "30x5_30x5x3",
    "30x10_30x10x2", "30x10_30x10x3", "30x10_30x10x5",
    # Massive
    "50x10_50x10x2", "50x10_50x10x3", "50x10_50x10x5",
    # Extreme
    "100x10_100x10x3",
]

TRAIN_DQN = False
TRAIN_PPO = False
RUN_VALIDATION = False
RUN_TEST = True
RUN_HEURISTIC_BASELINE = True

EPISODES = 50               # total episodes to train (including already-completed if resuming)
SEED = 42

# Set to a .pt checkpoint path to resume, or None for fresh training.
RESUME_DQN = None
RESUME_PPO = None


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


def write_rows(rows, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / f"{name}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


# ============================================================
#  DQN training  (with checkpoint resume support)
# ============================================================

CKPT_DQN = "dqn_checkpoint.pt"


def train_dqn(instances, cfg, out_dir, resume_from=None):
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    live_log = out_dir / "train_log_live.jsonl"

    agent = DQNAgent(make_env(instances[0], cfg.seed).feature_dim, cfg)
    logs = []
    start_ep = 0

    # --- Resume from checkpoint ---
    if resume_from and Path(resume_from).exists():
        print(f"[DQN] Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=agent.device)
        agent.q.load_state_dict(ckpt["q"])
        agent.target.load_state_dict(ckpt["target"])
        agent.opt.load_state_dict(ckpt["optimizer"])
        agent.epsilon = ckpt["epsilon"]
        agent.steps = ckpt["steps"]
        start_ep = ckpt["episode"]
        logs = ckpt.get("logs", [])
        print(f"[DQN] Resumed at episode {start_ep}, epsilon={agent.epsilon:.4f}")
    else:
        live_log.write_text("", encoding="utf-8")

    # --- Training loop ---
    for ep in range(start_ep + 1, cfg.episodes + 1):
        inst = instances[(ep - 1) % len(instances)]
        env = make_env(inst, cfg.seed + ep)
        obs = env.reset()
        total = 0.0
        losses = []
        decisions = 0
        ep_transitions = []  # collect raw transitions for n-step computation

        while not obs["done"] and decisions < cfg.max_decisions:
            action = agent.act(obs, explore=True)
            features = obs["features"].copy()
            mask = obs["mask"].copy()
            next_obs, reward, done, _ = env.step(action)
            ep_transitions.append({
                "features": features,
                "mask": mask,
                "action": action,
                "reward": reward,
                "next_features": next_obs["features"].copy(),
                "next_mask": next_obs["mask"].copy(),
                "done": done,
            })
            obs = next_obs
            total += reward
            decisions += 1

        # --- Compute n-step returns and push to replay buffer ---
        n = min(cfg.n_step, len(ep_transitions))
        for i in range(len(ep_transitions)):
            t = ep_transitions[i]
            n_step_return = 0.0
            terminal_in_n = False
            end = min(i + n, len(ep_transitions))
            for j in range(i, end):
                n_step_return += (cfg.gamma ** (j - i)) * ep_transitions[j]["reward"]
                if ep_transitions[j]["done"]:
                    terminal_in_n = True
                    break
            if not terminal_in_n and end < len(ep_transitions):
                # Bootstrap from Q_target at step i+n
                boot = ep_transitions[end]
                n_step_return += (cfg.gamma ** n) * agent.bootstrap_value(
                    boot["features"], boot["mask"]
                )

            agent.remember(
                Transition(
                    features=t["features"],
                    action=t["action"],
                    reward=n_step_return,
                    next_features=np.zeros((0,), dtype=np.float32),
                    next_mask=np.zeros((0,), dtype=bool),
                    done=terminal_in_n,
                    is_n_step=True,
                )
            )

        # --- Update from replay buffer (cap to avoid excessive updates) ---
        n_updates = min(decisions, 200)
        for _ in range(n_updates):
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

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
        agent.decay_epsilon()
        if cfg.lr_decay < 1.0:
            agent.decay_lr(cfg.lr_decay)
        logs.append(row)
        with live_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if ep == 1 or ep % 10 == 0:
            print("[DQN]", json.dumps(row, ensure_ascii=False))

    # --- Save model + checkpoint ---
    agent.save(out_dir / "dqn_model.pt")
    write_rows(logs, out_dir, "train_log")
    torch.save(
        {
            "q": agent.q.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "epsilon": agent.epsilon,
            "steps": agent.steps,
            "episode": cfg.episodes,
            "logs": logs,
        },
        out_dir / CKPT_DQN,
    )
    print(f"[DQN] Checkpoint saved (ep {cfg.episodes})")
    return out_dir / "dqn_model.pt"


# ============================================================
#  PPO training  (with checkpoint resume support)
# ============================================================

CKPT_PPO = "ppo_checkpoint.pt"


def train_ppo(instances, cfg, out_dir, resume_from=None):
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    live_log = out_dir / "train_log_live.jsonl"

    agent = PPOAgent(make_env(instances[0], cfg.seed).feature_dim, cfg)
    logs = []
    start_ep = 0

    # --- Resume from checkpoint ---
    if resume_from and Path(resume_from).exists():
        print(f"[PPO] Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=agent.device)
        agent.net.load_state_dict(ckpt["net"])
        agent.opt.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt["episode"]
        logs = ckpt.get("logs", [])
        print(f"[PPO] Resumed at episode {start_ep}")
    else:
        live_log.write_text("", encoding="utf-8")

    # --- Training loop ---
    for ep in range(start_ep + 1, cfg.episodes + 1):
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
        if cfg.lr_decay < 1.0:
            agent.decay_lr(cfg.lr_decay)
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
            print("[PPO]", json.dumps(row, ensure_ascii=False))

    # --- Save model + checkpoint ---
    agent.save(out_dir / "ppo_model.pt")
    write_rows(logs, out_dir, "train_log")
    torch.save(
        {
            "net": agent.net.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "episode": cfg.episodes,
            "logs": logs,
        },
        out_dir / CKPT_PPO,
    )
    print(f"[PPO] Checkpoint saved (ep {cfg.episodes})")
    return out_dir / "ppo_model.pt"


# ============================================================
#  Evaluation helpers
# ============================================================

def evaluate_heuristic(instances, split_name, out_dir):
    rows = []
    for inst in instances:
        env = make_env(inst, SEED)
        obs = env.reset()
        total = 0.0
        decisions = 0
        while not obs["done"] and decisions < TrainConfig.max_decisions:
            action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward
            decisions += 1
        rows.append(env_metrics(env, total, decisions, "heuristic", "rest_aware"))
        print("[BASELINE]", rows[-1])
    write_rows(rows, out_dir / split_name, "heuristic_rest_aware")


def evaluate_model(algorithm, model_path, instances, split_name, out_dir):
    probe_env = make_env(instances[0], SEED)
    model, device = load_model(algorithm, model_path, probe_env.feature_dim, TrainConfig.hidden_dim)
    rows = []
    for idx, inst in enumerate(instances):
        env = make_env(inst, SEED + idx)
        obs = env.reset()
        total = 0.0
        decisions = 0
        no_progress = 0
        last_tasks = 0
        while not obs["done"] and decisions < TrainConfig.max_decisions:
            if algorithm == "dqn":
                action = choose_dqn_action(model, obs, device)
            else:
                action = choose_ppo_action(model, obs, device)
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or no_progress >= 20:
                action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward
            decisions += 1
            if len(env.history) == last_tasks:
                no_progress += 1
            else:
                no_progress = 0
                last_tasks = len(env.history)
        rows.append(env_metrics(env, total, decisions, algorithm, model_path.name))
        print(f"[{algorithm.upper()} {split_name}]", rows[-1])
    write_rows(rows, out_dir / split_name, f"{algorithm}_{model_path.stem}")


# ============================================================
#  Visualization helpers
# ============================================================

def plot_gantt(history, n_machines, out_path):
    fig, ax = plt.subplots(figsize=(12, max(4, n_machines * 0.45)))
    colors = plt.cm.tab20.colors
    for task in history:
        y = task.machine
        ax.barh(
            y, task.finish - task.start, left=task.start,
            color=colors[task.worker % len(colors)], edgecolor="black", height=0.75,
        )
        ax.text(
            task.start + (task.finish - task.start) / 2, y,
            f"J{task.job + 1}-O{task.op + 1}/W{task.worker + 1}",
            ha="center", va="center", fontsize=7,
        )
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{i + 1}" for i in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title("Machine Gantt Chart")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Gantt chart saved to {out_path}")


def plot_fatigue(history, n_workers, out_path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for worker in range(n_workers):
        xs, ys = [0], [0]
        for task in history:
            if task.worker != worker:
                continue
            xs.extend([task.start, task.finish])
            ys.extend([task.fatigue_before, task.fatigue_after])
        ax.step(xs, ys, where="post", label=f"W{worker + 1}")
    ax.axhline(0.8, linestyle="--", color="red", linewidth=1, label="F_force")
    ax.axhline(0.5, linestyle=":", color="green", linewidth=1, label="F_resume")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time"); ax.set_ylabel("Fatigue")
    ax.set_title("Worker Fatigue Curve")
    ax.grid(alpha=0.25); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Fatigue curve saved to {out_path}")


def visualize_model(algorithm, model_path, instance, out_dir):
    from config import TrainConfig as TCfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(instance, SEED)
    model, device = load_model(algorithm, Path(model_path), env.feature_dim, TCfg.hidden_dim)
    obs = env.reset()
    total = 0.0; decisions = 0; no_progress = 0; last_tasks = 0
    while not obs["done"] and decisions < TCfg.max_decisions:
        if algorithm == "dqn":
            action = choose_dqn_action(model, obs, device)
        else:
            action = choose_ppo_action(model, obs, device)
        if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or no_progress >= 20:
            action = select_action(obs, policy="rest_aware")
        obs, reward, _, _ = env.step(action)
        total += reward; decisions += 1
        if len(env.history) == last_tasks:
            no_progress += 1
        else:
            no_progress = 0; last_tasks = len(env.history)
    label = f"{algorithm}_{Path(model_path).stem}"
    plot_gantt(env.history, instance.n_machines, out_dir / f"{instance.name}_{label}_gantt.png")
    plot_fatigue(env.history, instance.n_workers, out_dir / f"{instance.name}_{label}_fatigue.png")
    print({"algorithm": algorithm, "instance": instance.name, "makespan": env.time,
           "decisions": decisions, "force_rest_time": int(env.force_rest_time.sum())})


# ============================================================
#  Main
# ============================================================

def main():
    set_seed(SEED)
    all_instances = load_instances(DATA_PATH)
    train_instances = [all_instances[name] for name in TRAIN_INSTANCES]
    val_instances = [all_instances[name] for name in VAL_INSTANCES]
    test_instances = [all_instances[name] for name in TEST_INSTANCES]

    cfg = TrainConfig(episodes=EPISODES, seed=SEED)

    dqn_out = OUTPUT_DIR / "dqn"
    ppo_out = OUTPUT_DIR / "ppo"
    dqn_path = dqn_out / "dqn_model.pt"
    ppo_path = ppo_out / "ppo_model.pt"

    # --- Train ---
    if TRAIN_DQN:
        dqn_path = train_dqn(train_instances, cfg, dqn_out,
                             resume_from=RESUME_DQN or (dqn_out / CKPT_DQN))
    if TRAIN_PPO:
        ppo_path = train_ppo(train_instances, cfg, ppo_out,
                             resume_from=RESUME_PPO or (ppo_out / CKPT_PPO))

    # --- Evaluate ---
    if RUN_VALIDATION:
        if RUN_HEURISTIC_BASELINE:
            evaluate_heuristic(val_instances, "val_results", OUTPUT_DIR)
        if dqn_path.exists():
            evaluate_model("dqn", dqn_path, val_instances, "val_results", OUTPUT_DIR)
        if ppo_path.exists():
            evaluate_model("ppo", ppo_path, val_instances, "val_results", OUTPUT_DIR)

    if RUN_TEST:
        if RUN_HEURISTIC_BASELINE:
            evaluate_heuristic(test_instances, "test_results", OUTPUT_DIR)
        if dqn_path.exists():
            evaluate_model("dqn", dqn_path, test_instances, "test_results", OUTPUT_DIR)
        if ppo_path.exists():
            evaluate_model("ppo", ppo_path, test_instances, "test_results", OUTPUT_DIR)

    # --- Visualization ---
    figures_dir = OUTPUT_DIR / "figures"
    viz_instance = train_instances[0]
    if dqn_path.exists():
        print("\n=== Visualizing DQN ===")
        visualize_model("dqn", dqn_path, viz_instance, figures_dir)
    if ppo_path.exists():
        print("\n=== Visualizing PPO ===")
        visualize_model("ppo", ppo_path, viz_instance, figures_dir)

    print("\n=== All done! ===")
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print(f"Figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
