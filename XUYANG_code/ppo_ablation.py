"""PPO ablation study — self-contained version."""
import json, csv, random, sys, numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from data_loader import load_instances
from config import EnvConfig, TrainConfig
from env import JobShopFatigueEnv
from heuristics import select_action
from models import ActorCriticNetwork, ScaleInvariantActorCritic
from agents import PPOAgent
from test import choose_ppo_action, env_metrics, load_model

# ============================================================
SEED = 42
EPISODES = 50   # reduced: no-curriculum variants hit large instances early → slow
OUT_DIR = Path("outputs_ablation")

TRAIN_INSTANCES = [
    "6x6_6x6x2", "10x5_10x5x3", "15x5_15x5x2", "10x10_10x10x4",
    "15x10_15x10x3", "20x10_20x10x5",
    "30x10_30x10x2", "30x10_30x10x6", "50x10_50x10x3", "100x10_100x10x3",
]
VAL_INSTANCES = [
    "20x10_20x10x4", "30x5_30x5x2", "30x5_30x5x3",
    "30x10_30x10x3", "30x10_30x10x4", "30x10_30x10x5",
]
TEST_INSTANCES = [
    "10x10_10x10x5", "10x10_10x10x6", "15x10_15x10x4", "15x10_15x10x5", "15x10_15x10x6",
    "20x10_20x10x2", "20x10_20x10x6", "10x10_10x10x2", "10x10_10x10x3",
    "20x5_20x5x2", "20x5_20x5x3",
    "50x10_50x10x2", "50x10_50x10x4", "50x10_50x10x5", "50x10_50x10x6",
    "100x10_100x10x2", "100x10_100x10x4", "100x10_100x10x5", "100x10_100x10x6",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env(instance, env_cfg, seed):
    return JobShopFatigueEnv(instance, env_cfg, seed=seed)

def curriculum_stages(instances, episodes):
    small, medium, large = [], [], []
    for inst in instances:
        tasks = inst.n_jobs * inst.n_machines
        if tasks <= 50: small.append(inst)
        elif tasks <= 100: medium.append(inst)
        else: large.append(inst)
    s1 = episodes // 3
    s2 = 2 * episodes // 3
    return [(1, s1, small), (s1+1, s2, small+medium), (s2+1, episodes, small+medium+large)]

def write_rows(rows, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

def evaluate_model(algorithm, model_path, instances, split_name, out_dir):
    probe_env = make_env(instances[0], EnvConfig(), SEED)
    model, device = load_model(algorithm, model_path, probe_env.feature_dim, TrainConfig.hidden_dim)
    rows = []
    for idx, inst in enumerate(instances):
        env = make_env(inst, EnvConfig(), SEED + idx)
        obs = env.reset(); total = 0.0; decisions = 0; no_progress = 0; last_tasks = 0
        while not obs["done"] and decisions < TrainConfig.max_decisions:
            if algorithm == "dqn":
                action = choose_dqn_action(model, obs, device)
            else:
                action = choose_ppo_action(model, obs, device)
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or no_progress >= 20:
                action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward; decisions += 1
            if len(env.history) == last_tasks: no_progress += 1
            else: no_progress = 0; last_tasks = len(env.history)
        rows.append(env_metrics(env, total, decisions, algorithm, model_path.name))
    write_rows(rows, out_dir / split_name, f"{algorithm}_{model_path.stem}")
    return rows


def train_ppo_variant(instances, cfg, out_dir, *,
                       network_type="scaleinv", use_reward_norm=True,
                       use_reward_shaping=True, use_curriculum=True,
                       label="ppo"):
    """Train a PPO variant with specified components."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = make_env(instances[0], EnvConfig(), 0).feature_dim

    # --- Build env config ---
    env_cfg = EnvConfig()
    if not use_reward_shaping:
        env_cfg.s_job_completion = 0.0
        env_cfg.s_efficiency = 0.0
        env_cfg.s_progress = 0.0
        env_cfg.s_stall = 0.0

    # --- Build agent ---
    if network_type == "scaleinv":
        net = ScaleInvariantActorCritic(feature_dim, cfg.hidden_dim).to(device)
    else:
        net = ActorCriticNetwork(feature_dim, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    ret_mean = 0.0
    ret_std = 1.0
    ret_momentum = 0.01

    logs = []
    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [(1, cfg.episodes, instances)]
    print(f"[{label}] Curriculum: {[(s,e,len(i)) for s,e,i in stages]}", flush=True)

    best_val_reward = -float("inf")
    best_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        # Pick instance
        stage_insts = instances
        for ss, se, si in stages:
            if ss <= ep <= se: stage_insts = si; break
        inst = stage_insts[(ep - 1) % len(stage_insts)]
        env = make_env(inst, env_cfg, cfg.seed + ep)
        obs = env.reset()
        rollout = []; total = 0.0; decisions = 0

        while not obs["done"] and decisions < cfg.max_decisions:
            feats = torch.tensor(obs["features"], dtype=torch.float32, device=device)
            mask = torch.tensor(obs["mask"], dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, value = net(feats, mask)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            item = {"features": obs["features"].copy(), "mask": obs["mask"].copy(),
                    "action": int(action.item()), "logp": float(logp.item()),
                    "value": float(value.item())}
            next_obs, reward, done, _ = env.step(int(action.item()))
            item["reward"] = reward; item["done"] = done
            rollout.append(item); obs = next_obs; total += reward; decisions += 1
            if len(rollout) >= cfg.rollout_steps:
                # Do PPO update
                _ppo_update(rollout, net, opt, cfg, device, use_reward_norm, ret_mean, ret_std, ret_momentum)
                if use_reward_norm:
                    ret_mean, ret_std = _update_running_stats(rollout, ret_mean, ret_std, ret_momentum, cfg, device)
                rollout = []

        if rollout:
            _ppo_update(rollout, net, opt, cfg, device, use_reward_norm, ret_mean, ret_std, ret_momentum)
            if use_reward_norm:
                ret_mean, ret_std = _update_running_stats(rollout, ret_mean, ret_std, ret_momentum, cfg, device)

        if cfg.lr_decay < 1.0:
            for pg in opt.param_groups: pg["lr"] *= cfg.lr_decay

        row = {"episode": ep, "instance": inst.name, "reward": total,
               "makespan": env.time, "decisions": decisions}
        logs.append(row)
        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} makespan={env.time}", flush=True)

        # Validation every 50 episodes
        if ep % 50 == 0 or ep == cfg.episodes:
            val_inst = instances[min(len(instances)-1, 3)]
            val_env = make_env(val_inst, env_cfg, cfg.seed + 99999)
            val_obs = val_env.reset()
            val_total = 0.0; val_dec = 0
            while not val_obs["done"] and val_dec < cfg.max_decisions:
                f2 = torch.tensor(val_obs["features"], dtype=torch.float32, device=device)
                m2 = torch.tensor(val_obs["mask"], dtype=torch.bool, device=device)
                with torch.no_grad():
                    l2, _ = net(f2, m2)
                    va = int(torch.argmax(l2).item())
                val_obs, vr, vd, _ = val_env.step(va)
                val_total += vr; val_dec += 1
            if val_total > best_val_reward:
                best_val_reward = val_total
                torch.save({"model": net.state_dict()}, best_path)
                print(f"[{label}] * New best ep={ep}: val_reward={val_total:.2f} val_makespan={val_env.time}", flush=True)

    torch.save({"model": net.state_dict()}, out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_path if best_path.exists() else out_dir / f"{label}_model.pt"


def _update_running_stats(rollout, ret_mean, ret_std, momentum, cfg, device):
    returns = []
    g = 0.0
    for x in reversed(rollout):
        g = x["reward"] + cfg.gamma * g * (1.0 - float(x["done"]))
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    bm = float(returns_t.mean().item()); bs = float(returns_t.std().item()) + 1e-8
    return (1-momentum)*ret_mean + momentum*bm, (1-momentum)*ret_std + momentum*bs


def _ppo_update(rollout, net, opt, cfg, device, use_reward_norm, ret_mean, ret_std, ret_momentum):
    rewards = [x["reward"] for x in rollout]
    dones = [x["done"] for x in rollout]
    values = [x["value"] for x in rollout]

    returns = []; g = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        g = reward + cfg.gamma * g * (1.0 - float(done))
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    if use_reward_norm:
        norm_returns_t = (returns_t - ret_mean) / ret_std
    else:
        norm_returns_t = returns_t

    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    adv_t = norm_returns_t - values_t
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # Store original indices for correct advantage/return lookup after shuffle
    for i, item in enumerate(rollout):
        item["_idx"] = i
        item["_ret"] = norm_returns_t[i]
        item["_adv"] = adv_t[i]

    for _ in range(cfg.ppo_epochs):
        random.shuffle(rollout)
        for item in rollout:
            features = torch.tensor(item["features"], dtype=torch.float32, device=device)
            mask = torch.tensor(item["mask"], dtype=torch.bool, device=device)
            action = torch.tensor(item["action"], dtype=torch.int64, device=device)
            old_logp = torch.tensor(item["logp"], dtype=torch.float32, device=device)

            logits, value = net(features, mask)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(action)
            ratio = torch.exp(logp - old_logp)
            policy_loss = -torch.min(ratio*item["_adv"], torch.clamp(ratio, 1-cfg.clip_ratio, 1+cfg.clip_ratio)*item["_adv"])
            value_loss = F.mse_loss(value, item["_ret"])
            entropy = dist.entropy()
            loss = policy_loss + cfg.value_coef*value_loss - cfg.entropy_coef*entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


# ============================================================
def main():
    print("Loading data...", flush=True)
    all_instances = load_instances("basic_data.xlsx")
    train_insts = [all_instances[n] for n in TRAIN_INSTANCES]
    val_insts = [all_instances[n] for n in VAL_INSTANCES]
    test_insts = [all_instances[n] for n in TEST_INSTANCES]

    cfg = TrainConfig(episodes=EPISODES, seed=SEED)

    ppo_configs = [
        ("ppo_vanilla",  {"network_type": "mlp",      "use_reward_norm": False, "use_reward_shaping": False, "use_curriculum": False}),
        ("ppo_rs_curr",  {"network_type": "mlp",      "use_reward_norm": False, "use_reward_shaping": True,  "use_curriculum": True}),
        ("ppo_scaleinv", {"network_type": "scaleinv", "use_reward_norm": False, "use_reward_shaping": True,  "use_curriculum": True}),
        ("ppo_full",     {"network_type": "scaleinv", "use_reward_norm": True,  "use_reward_shaping": True,  "use_curriculum": True}),
    ]

    for label, kwargs in ppo_configs:
        print(f"\n{'='*60}", flush=True)
        print(f"TRAINING: {label}", flush=True)
        print("="*60, flush=True)
        model_path = train_ppo_variant(train_insts, cfg, OUT_DIR / label, label=label, **kwargs)

        for split_name, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            print(f"  Evaluating {label} on {split_name}...", flush=True)
            rows = evaluate_model("ppo", model_path, insts, split_name, OUT_DIR / label)
            avg_mk = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split_name}: avg makespan={avg_mk:.0f}", flush=True)

    print("\nPPO ablations complete!", flush=True)


if __name__ == "__main__":
    main()
