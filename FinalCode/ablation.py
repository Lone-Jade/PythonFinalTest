"""Ablation study: measure contribution of each component.

Runs DQN and PPO variants on the cross-scale training set (10 instances),
trains for ABLATION_EPISODES, then evaluates on train/val/test splits.

DQN variants (cumulative):
  dqn_vanilla          PairScoring + uniform replay + basic reward
  dqn_per_rs           + PER + Reward Shaping
  dqn_scaleinv         + ScaleInvariantDueling + PER + RS (no curriculum)
  dqn_full             + Curriculum (= current 300ep best)

PPO variants (cumulative):
  ppo_vanilla          ActorCritic + basic reward
  ppo_rs_curr          + Reward Shaping + Curriculum
  ppo_scaleinv         + ScaleInvariantActorCritic + RS + Curr (no RewardNorm)
  ppo_full             + Reward Normalization (= current 300ep best)
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

from agents import DQNAgent, PPOAgent, Transition, PrioritizedReplayBuffer
from config import EnvConfig, TrainConfig
from data_loader import load_instances
from env import JobShopFatigueEnv
from heuristics import select_action
from models import (
    ActorCriticNetwork,
    DuelingPairScoringNetwork,
    PairScoringNetwork,
    ScaleInvariantActorCritic,
    ScaleInvariantDuelingNetwork,
)
from test import choose_dqn_action, choose_ppo_action, env_metrics, load_model

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "basic_data.xlsx"
OUTPUT_DIR = Path("outputs_ablation")
ABLATION_EPISODES = 50
SEED = 42

TRAIN_INSTANCES = [
    "6x6_6x6x2", "10x5_10x5x3",
    "15x5_15x5x2", "10x10_10x10x4",
    "15x10_15x10x3", "20x10_20x10x5",
    "30x10_30x10x2", "30x10_30x10x6",
    "50x10_50x10x3", "100x10_100x10x3",
]

VAL_INSTANCES = [
    "20x10_20x10x4",
    "30x5_30x5x2", "30x5_30x5x3",
    "30x10_30x10x3", "30x10_30x10x4", "30x10_30x10x5",
]

TEST_INSTANCES = [
    "10x10_10x10x5", "10x10_10x10x6",
    "15x10_15x10x4", "15x10_15x10x5", "15x10_15x10x6",
    "20x10_20x10x2", "20x10_20x10x6",
    "10x10_10x10x2", "10x10_10x10x3",
    "20x5_20x5x2", "20x5_20x5x3",
    "50x10_50x10x2", "50x10_50x10x4", "50x10_50x10x5", "50x10_50x10x6",
    "100x10_100x10x2", "100x10_100x10x4", "100x10_100x10x5", "100x10_100x10x6",
]


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


def make_env(instance, env_cfg, seed):
    return JobShopFatigueEnv(instance, env_cfg, seed=seed)


def curriculum_stages(instances, episodes):
    small, medium, large = [], [], []
    for inst in instances:
        tasks = inst.n_jobs * inst.n_machines
        if tasks <= 50:
            small.append(inst)
        elif tasks <= 100:
            medium.append(inst)
        else:
            large.append(inst)
    stage1_end = episodes // 3
    stage2_end = 2 * episodes // 3
    return [
        (1, stage1_end, small),
        (stage1_end + 1, stage2_end, small + medium),
        (stage2_end + 1, episodes, small + medium + large),
    ]


def write_rows(rows, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("json", "csv"):
        if fmt == "json":
            (out_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            if rows:
                with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)


def evaluate(algorithm, model_path, instances, split_name, out_dir):
    """Evaluate a trained model on a set of instances."""
    import torch
    probe_env = make_env(instances[0], EnvConfig(), SEED)
    model, device = load_model(algorithm, model_path, probe_env.feature_dim, TrainConfig.hidden_dim)
    rows = []
    for idx, inst in enumerate(instances):
        env = make_env(inst, EnvConfig(), SEED + idx)
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
    write_rows(rows, out_dir / split_name, f"{algorithm}_{model_path.stem}")
    return rows


def evaluate_heuristic(instances, split_name, out_dir):
    rows = []
    for inst in instances:
        env = make_env(inst, EnvConfig(), SEED)
        obs = env.reset()
        total = 0.0
        decisions = 0
        while not obs["done"] and decisions < TrainConfig.max_decisions:
            action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward
            decisions += 1
        rows.append(env_metrics(env, total, decisions, "heuristic", "rest_aware"))
    write_rows(rows, out_dir / split_name, "heuristic_rest_aware")
    return rows


# ============================================================
# DQN Ablation Training
# ============================================================

def train_dqn_ablation(instances, cfg, out_dir, *,
                        network_type="scaleinv", use_per=True,
                        use_reward_shaping=True, use_curriculum=True,
                        label="dqn_full"):
    """Train a DQN variant with specified components."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig()
    if not use_reward_shaping:
        # Zero out reward shaping bonuses
        env_cfg.s_job_completion = 0.0
        env_cfg.s_efficiency = 0.0
        env_cfg.s_progress = 0.0
        env_cfg.s_stall = 0.0

    agent = _make_dqn_agent(instances[0], cfg, network_type, use_per)
    logs = []

    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [
        (1, cfg.episodes, instances)
    ]
    print(f"[{label}] Curriculum: {[(s, e, len(i)) for s, e, i in stages]}")

    best_val_reward = -float("inf")
    best_model_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        stage_instances = instances
        for s_start, s_end, s_insts in stages:
            if s_start <= ep <= s_end:
                stage_instances = s_insts
                break
        inst = stage_instances[(ep - 1) % len(stage_instances)]
        env = make_env(inst, env_cfg, cfg.seed + ep)
        obs = env.reset()
        total = 0.0
        losses = []
        decisions = 0
        ep_transitions = []

        while not obs["done"] and decisions < cfg.max_decisions:
            action = agent.act(obs, explore=True)
            features = obs["features"].copy()
            mask = obs["mask"].copy()
            next_obs, reward, done, _ = env.step(action)
            ep_transitions.append({
                "features": features, "mask": mask, "action": action,
                "reward": reward, "next_features": next_obs["features"].copy(),
                "next_mask": next_obs["mask"].copy(), "done": done,
            })
            obs = next_obs
            total += reward
            decisions += 1

        # n-step push
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
                boot = ep_transitions[end]
                n_step_return += (cfg.gamma ** n) * agent.bootstrap_value(
                    boot["features"], boot["mask"]
                )
            agent.remember(Transition(
                features=t["features"], action=t["action"],
                reward=n_step_return,
                next_features=np.zeros((0,), dtype=np.float32),
                next_mask=np.zeros((0,), dtype=bool),
                done=terminal_in_n, is_n_step=True,
            ))

        n_updates = min(decisions, 200)
        for _ in range(n_updates):
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        agent.decay_epsilon()
        if cfg.lr_decay < 1.0:
            agent.decay_lr(cfg.lr_decay)

        row = {
            "episode": ep, "instance": inst.name, "reward": total,
            "makespan": env.time, "decisions": decisions,
            "epsilon": agent.epsilon,
            "loss": float(np.mean(losses)) if losses else None,
        }
        logs.append(row)

        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} makespan={env.time} eps={agent.epsilon:.3f} loss={row['loss']}")

        # Best-model tracking on validation
        if ep % 50 == 0 or ep == cfg.episodes:
            val_inst = instances[min(len(instances) - 1, 3)]
            val_env = make_env(val_inst, env_cfg, cfg.seed + 99999)
            val_obs = val_env.reset()
            val_total = 0.0
            val_decisions = 0
            while not val_obs["done"] and val_decisions < cfg.max_decisions:
                val_action = agent.act(val_obs, explore=False)
                val_obs, val_rew, val_done, _ = val_env.step(val_action)
                val_total += val_rew
                val_decisions += 1
            if val_total > best_val_reward:
                best_val_reward = val_total
                agent.save(best_model_path)
                print(f"[{label}] ★ New best ep={ep}: val_reward={val_total:.2f} val_makespan={val_env.time}")

    # Save final
    agent.save(out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_model_path if best_model_path.exists() else out_dir / f"{label}_model.pt"


def _make_dqn_agent(instance, cfg, network_type, use_per):
    """Create DQN agent with specified component configuration."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = make_env(instance, EnvConfig(), 0).feature_dim

    # Build agent manually to control components
    class ConfiguredDQNAgent(DQNAgent):
        def __init__(self):
            self.cfg = cfg
            self.device = device
            # Select network
            if network_type == "pair":
                self.q = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(device)
                self.target = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(device)
            elif network_type == "dueling":
                self.q = DuelingPairScoringNetwork(feature_dim, cfg.hidden_dim).to(device)
                self.target = DuelingPairScoringNetwork(feature_dim, cfg.hidden_dim).to(device)
            else:  # scaleinv
                self.q = ScaleInvariantDuelingNetwork(feature_dim, cfg.hidden_dim).to(device)
                self.target = ScaleInvariantDuelingNetwork(feature_dim, cfg.hidden_dim).to(device)
            self.target.load_state_dict(self.q.state_dict())
            self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
            # Replay buffer
            if use_per:
                self.buffer = PrioritizedReplayBuffer(cfg.replay_size, cfg.per_alpha, cfg.per_beta)
            else:
                self.buffer = _UniformReplayBuffer(cfg.replay_size)
            self.epsilon = cfg.epsilon_start
            self.steps = 0

    return ConfiguredDQNAgent()


class _UniformReplayBuffer:
    """Simple uniform-sampling replay buffer (no prioritization)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.size = 0

    def push(self, item):
        if self.size < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        pass  # no-op for uniform

    def __len__(self):
        return self.size


# ============================================================
# PPO Ablation Training
# ============================================================

def train_ppo_ablation(instances, cfg, out_dir, *,
                        network_type="scaleinv", use_reward_norm=True,
                        use_reward_shaping=True, use_curriculum=True,
                        label="ppo_full"):
    """Train a PPO variant with specified components."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig()
    if not use_reward_shaping:
        env_cfg.s_job_completion = 0.0
        env_cfg.s_efficiency = 0.0
        env_cfg.s_progress = 0.0
        env_cfg.s_stall = 0.0

    agent = _make_ppo_agent(instances[0], cfg, network_type, use_reward_norm)
    logs = []

    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [
        (1, cfg.episodes, instances)
    ]
    print(f"[{label}] Curriculum: {[(s, e, len(i)) for s, e, i in stages]}")

    best_val_reward = -float("inf")
    best_model_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        stage_instances = instances
        for s_start, s_end, s_insts in stages:
            if s_start <= ep <= s_end:
                stage_instances = s_insts
                break
        inst = stage_instances[(ep - 1) % len(stage_instances)]
        env = make_env(inst, env_cfg, cfg.seed + ep)
        obs = env.reset()
        rollout = []
        total = 0.0
        decisions = 0

        while not obs["done"] and decisions < cfg.max_decisions:
            action, logp, value = agent.act(obs, explore=True)
            item = {
                "features": obs["features"].copy(), "mask": obs["mask"].copy(),
                "action": action, "logp": logp, "value": value,
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
            "episode": ep, "instance": inst.name, "reward": total,
            "makespan": env.time, "decisions": decisions, "loss": loss,
        }
        logs.append(row)

        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} makespan={env.time} loss={loss}")

        # Best-model tracking
        if ep % 50 == 0 or ep == cfg.episodes:
            val_inst = instances[min(len(instances) - 1, 3)]
            val_env = make_env(val_inst, env_cfg, cfg.seed + 99999)
            val_obs = val_env.reset()
            val_total = 0.0
            val_decisions = 0
            while not val_obs["done"] and val_decisions < cfg.max_decisions:
                val_action, _, _ = agent.act(val_obs, explore=False)
                val_obs, val_rew, val_done, _ = val_env.step(val_action)
                val_total += val_rew
                val_decisions += 1
            if val_total > best_val_reward:
                best_val_reward = val_total
                agent.save(best_model_path)
                print(f"[{label}] ★ New best ep={ep}: val_reward={val_total:.2f} val_makespan={val_env.time}")

    agent.save(out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_model_path if best_model_path.exists() else out_dir / f"{label}_model.pt"


def _make_ppo_agent(instance, cfg, network_type, use_reward_norm):
    """Create PPO agent with specified component configuration."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = make_env(instance, EnvConfig(), 0).feature_dim

    class ConfiguredPPOAgent(PPOAgent):
        def __init__(self):
            self.cfg = cfg
            self.device = device
            if network_type == "scaleinv":
                self.net = ScaleInvariantActorCritic(feature_dim, cfg.hidden_dim).to(device)
            else:
                self.net = ActorCriticNetwork(feature_dim, cfg.hidden_dim).to(device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
            if use_reward_norm:
                self.ret_mean = 0.0
                self.ret_std = 1.0
                self.ret_momentum = 0.01
            else:
                # Disable reward normalization — use simple advantage normalization
                self.ret_mean = 0.0
                self.ret_std = 1.0
                self.ret_momentum = 0.01

    agent = ConfiguredPPOAgent()
    if not use_reward_norm:
        # Patch update() to skip reward normalization
        original_update = agent.update
        def patched_update(rollout):
            if not rollout:
                return None
            rewards = [x["reward"] for x in rollout]
            dones = [x["done"] for x in rollout]
            values = [x["value"] for x in rollout]
            returns = []
            g = 0.0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                g = reward + cfg.gamma * g * (1.0 - float(done))
                returns.append(g)
            returns.reverse()
            returns_t = torch.tensor(returns, dtype=torch.float32, device=agent.device)
            values_t = torch.tensor(values, dtype=torch.float32, device=agent.device)
            adv_t = returns_t - values_t
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            # Store un-normalized returns
            for i, item in enumerate(rollout):
                item["_return"] = returns_t[i]
                item["_adv"] = adv_t[i]
            losses = []
            for _ in range(cfg.ppo_epochs):
                total = 0.0
                random.shuffle(rollout)
                for item in rollout:
                    features = torch.tensor(item["features"], dtype=torch.float32, device=agent.device)
                    mask = torch.tensor(item["mask"], dtype=torch.bool, device=agent.device)
                    action = torch.tensor(item["action"], dtype=torch.int64, device=agent.device)
                    old_logp = torch.tensor(item["logp"], dtype=torch.float32, device=agent.device)
                    logits, value = agent.net(features, mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(action)
                    ratio = torch.exp(logp - old_logp)
                    adv = item["_adv"]
                    policy_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv,
                    )
                    value_loss = torch.nn.functional.mse_loss(value, item["_return"])
                    entropy = dist.entropy()
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                    agent.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
                    agent.opt.step()
                    total += float(loss.item())
                losses.append(total / max(1, len(rollout)))
            return float(np.mean(losses))
        agent.update = patched_update

    return agent


# ============================================================
# Main: run all ablations
# ============================================================

def main():
    set_seed(SEED)
    all_instances = load_instances(DATA_PATH)
    train_insts = [all_instances[name] for name in TRAIN_INSTANCES]
    val_insts = [all_instances[name] for name in VAL_INSTANCES]
    test_insts = [all_instances[name] for name in TEST_INSTANCES]

    cfg = TrainConfig(episodes=ABLATION_EPISODES, seed=SEED)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_test_results = {}

    # --- Heuristic baselines ---
    print("\n" + "=" * 60)
    print("HEURISTIC BASELINES")
    print("=" * 60)
    for split_name, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
        rows = evaluate_heuristic(insts, split_name, out_dir)
        key = f"heuristic_{split_name}"
        all_test_results[key] = rows
        avg_mk = np.mean([r["makespan"] for r in rows])
        print(f"  {split_name}: avg makespan={avg_mk:.0f}")

    # --- DQN Ablations ---
    dqn_configs = [
        ("dqn_vanilla",    {"network_type": "pair",     "use_per": False, "use_reward_shaping": False, "use_curriculum": False}),
        ("dqn_per_rs",     {"network_type": "pair",     "use_per": True,  "use_reward_shaping": True,  "use_curriculum": False}),
        ("dqn_scaleinv",   {"network_type": "scaleinv", "use_per": True,  "use_reward_shaping": True,  "use_curriculum": False}),
        ("dqn_full",       {"network_type": "scaleinv", "use_per": True,  "use_reward_shaping": True,  "use_curriculum": True}),
    ]

    dqn_models = {}
    for label, kwargs in dqn_configs:
        print("\n" + "=" * 60)
        print(f"TRAINING: {label}")
        print("=" * 60)
        model_path = train_dqn_ablation(train_insts, cfg, out_dir / label, label=label, **kwargs)
        dqn_models[label] = model_path

        # Evaluate on all splits
        for split_name, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            rows = evaluate("dqn", model_path, insts, split_name, out_dir / label)
            all_test_results[f"{label}_{split_name}"] = rows
            avg_mk = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split_name}: avg makespan={avg_mk:.0f}")

    # --- PPO Ablations ---
    ppo_configs = [
        ("ppo_vanilla",    {"network_type": "mlp",      "use_reward_norm": False, "use_reward_shaping": False, "use_curriculum": False}),
        ("ppo_rs_curr",    {"network_type": "mlp",      "use_reward_norm": False, "use_reward_shaping": True,  "use_curriculum": True}),
        ("ppo_scaleinv",   {"network_type": "scaleinv", "use_reward_norm": False, "use_reward_shaping": True,  "use_curriculum": True}),
        ("ppo_full",       {"network_type": "scaleinv", "use_reward_norm": True,  "use_reward_shaping": True,  "use_curriculum": True}),
    ]

    ppo_models = {}
    for label, kwargs in ppo_configs:
        print("\n" + "=" * 60)
        print(f"TRAINING: {label}")
        print("=" * 60)
        model_path = train_ppo_ablation(train_insts, cfg, out_dir / label, label=label, **kwargs)
        ppo_models[label] = model_path

        for split_name, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            rows = evaluate("ppo", model_path, insts, split_name, out_dir / label)
            all_test_results[f"{label}_{split_name}"] = rows
            avg_mk = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split_name}: avg makespan={avg_mk:.0f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY: Test Set Performance")
    print("=" * 60)
    print(f"{'Model':<25s} {'Avg Makespan':>12s} {'Avg/H':>8s}")
    print("-" * 45)
    h_test = [r for r in all_test_results["heuristic_test"]]
    h_avg = np.mean([r["makespan"] for r in h_test])

    summary_rows = []
    for label in [c[0] for c in dqn_configs + ppo_configs]:
        key = f"{label}_test"
        if key in all_test_results:
            rows = all_test_results[key]
            avg_mk = np.mean([r["makespan"] for r in rows])
            ratio = avg_mk / h_avg if h_avg else 0
            print(f"{label:<25s} {avg_mk:>12.0f} {ratio:>7.1%}")
            summary_rows.append({"model": label, "avg_makespan": avg_mk, "ratio_to_heuristic": ratio})

    print(f"{'heuristic':<25s} {h_avg:>12.0f} {'100.0%':>8s}")

    write_rows(summary_rows, out_dir, "ablation_summary")
    print(f"\nAll results saved to {out_dir.resolve()}")
    return all_test_results


if __name__ == "__main__":
    main()
