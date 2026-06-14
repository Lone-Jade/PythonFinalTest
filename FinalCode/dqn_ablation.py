"""DQN ablation study — 50 episodes, self-contained."""
import json, csv, random, numpy as np
from pathlib import Path

import torch

from data_loader import load_instances
from config import EnvConfig, TrainConfig
from env import JobShopFatigueEnv
from heuristics import select_action
from models import PairScoringNetwork, DuelingPairScoringNetwork, ScaleInvariantDuelingNetwork
from agents import DQNAgent, Transition, PrioritizedReplayBuffer
from test import choose_dqn_action, env_metrics, load_model

SEED = 42
EPISODES = 50
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
            action = choose_dqn_action(model, obs, device)
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or no_progress >= 20:
                action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward; decisions += 1
            if len(env.history) == last_tasks: no_progress += 1
            else: no_progress = 0; last_tasks = len(env.history)
        rows.append(env_metrics(env, total, decisions, algorithm, model_path.name))
    write_rows(rows, out_dir / split_name, f"{algorithm}_{model_path.stem}")
    return rows


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity; self.buffer = []; self.pos = 0; self.size = 0
    def push(self, item):
        if self.size < self.capacity: self.buffer.append(item)
        else: self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        if self.size < batch_size: return None, None, None
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch, indices, np.ones(batch_size, dtype=np.float32)
    def update_priorities(self, indices, td_errors): pass
    def __len__(self): return self.size


def train_dqn_variant(instances, cfg, out_dir, *,
                       network_type="scaleinv", use_per=True,
                       use_reward_shaping=True, use_curriculum=True,
                       label="dqn"):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = make_env(instances[0], EnvConfig(), 0).feature_dim

    env_cfg = EnvConfig()
    if not use_reward_shaping:
        env_cfg.s_job_completion = 0.0; env_cfg.s_efficiency = 0.0
        env_cfg.s_progress = 0.0; env_cfg.s_stall = 0.0

    # Build networks
    net_cls = {"pair": PairScoringNetwork, "dueling": DuelingPairScoringNetwork,
               "scaleinv": ScaleInvariantDuelingNetwork}[network_type]
    q_net = net_cls(feature_dim, cfg.hidden_dim).to(device)
    target_net = net_cls(feature_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    buffer = PrioritizedReplayBuffer(cfg.replay_size, cfg.per_alpha, cfg.per_beta) if use_per else UniformReplayBuffer(cfg.replay_size)
    epsilon = cfg.epsilon_start
    steps = 0

    logs = []
    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [(1, cfg.episodes, instances)]
    print(f"[{label}] Curriculum: {[(s,e,len(i)) for s,e,i in stages]}", flush=True)

    best_val_reward = -float("inf")
    best_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        stage_insts = instances
        for ss, se, si in stages:
            if ss <= ep <= se: stage_insts = si; break
        inst = stage_insts[(ep - 1) % len(stage_insts)]
        env = make_env(inst, env_cfg, cfg.seed + ep)
        obs = env.reset(); total = 0.0; losses = []; decisions = 0; ep_transitions = []

        while not obs["done"] and decisions < cfg.max_decisions:
            # Epsilon-greedy
            mask = obs["mask"]; legal = np.flatnonzero(mask)
            if len(legal) == 0: action = 0
            elif random.random() < epsilon: action = int(random.choice(legal))
            else:
                with torch.no_grad():
                    feats = torch.tensor(obs["features"], dtype=torch.float32, device=device)
                    qv = q_net(feats).detach().cpu().numpy()
                qv[~mask] = -1e9; action = int(np.argmax(qv))

            features = obs["features"].copy(); mask_copy = obs["mask"].copy()
            next_obs, reward, done, _ = env.step(action)
            ep_transitions.append({
                "features": features, "mask": mask_copy, "action": action,
                "reward": reward, "next_features": next_obs["features"].copy(),
                "next_mask": next_obs["mask"].copy(), "done": done,
            })
            obs = next_obs; total += reward; decisions += 1

        # n-step push
        n = min(cfg.n_step, len(ep_transitions))
        for i in range(len(ep_transitions)):
            t = ep_transitions[i]
            n_step_return = 0.0; terminal_in_n = False
            end = min(i + n, len(ep_transitions))
            for j in range(i, end):
                n_step_return += (cfg.gamma ** (j - i)) * ep_transitions[j]["reward"]
                if ep_transitions[j]["done"]: terminal_in_n = True; break
            if not terminal_in_n and end < len(ep_transitions):
                boot = ep_transitions[end]
                with torch.no_grad():
                    nf = torch.tensor(boot["features"], dtype=torch.float32, device=device)
                    qn = target_net(nf)
                    m = torch.tensor(boot["mask"], dtype=torch.bool, device=device)
                    qn = qn.masked_fill(~m, -1e9)
                    n_step_return += (cfg.gamma ** n) * float(qn.max().item())
            buffer.push(Transition(
                features=t["features"], action=t["action"], reward=n_step_return,
                next_features=np.zeros((0,), dtype=np.float32),
                next_mask=np.zeros((0,), dtype=bool), done=terminal_in_n, is_n_step=True,
            ))

        n_updates = min(decisions, 200)
        for _ in range(n_updates):
            result = buffer.sample(cfg.batch_size)
            if result[0] is None: continue
            batch, indices, is_weights = result
            chosen = torch.tensor(np.stack([b.features[b.action] for b in batch]), dtype=torch.float32, device=device)
            weights_t = torch.tensor(is_weights, dtype=torch.float32, device=device)
            with torch.no_grad():
                targets = []
                for b in batch:
                    if b.is_n_step: targets.append(b.reward)
                    elif b.done or len(b.next_features) == 0 or not b.next_mask.any(): targets.append(b.reward)
                    else:
                        nf = torch.tensor(b.next_features, dtype=torch.float32, device=device)
                        qn = target_net(nf)
                        mk = torch.tensor(b.next_mask, dtype=torch.bool, device=device)
                        qn = qn.masked_fill(~mk, -1e9)
                        targets.append(b.reward + cfg.gamma * float(qn.max().item()))
                target = torch.tensor(targets, dtype=torch.float32, device=device)
            pred = q_net(chosen)
            td_errors = (pred - target).detach().cpu().numpy()
            loss = (weights_t * torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0); opt.step()
            buffer.update_priorities(indices, td_errors)
            steps += 1
            if steps % cfg.target_update == 0: target_net.load_state_dict(q_net.state_dict())
            losses.append(float(loss.item()))

        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
        if cfg.lr_decay < 1.0:
            for pg in opt.param_groups: pg["lr"] *= cfg.lr_decay

        row = {"episode": ep, "instance": inst.name, "reward": total,
               "makespan": env.time, "decisions": decisions, "epsilon": epsilon,
               "loss": float(np.mean(losses)) if losses else None}
        logs.append(row)
        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} makespan={env.time} eps={epsilon:.3f}", flush=True)

        if ep % 50 == 0 or ep == cfg.episodes:
            val_inst = instances[min(len(instances)-1, 3)]
            val_env = make_env(val_inst, env_cfg, cfg.seed + 99999)
            val_obs = val_env.reset(); val_total = 0.0; val_dec = 0
            while not val_obs["done"] and val_dec < cfg.max_decisions:
                mask = val_obs["mask"]; legal = np.flatnonzero(mask)
                if len(legal) == 0: va = 0
                else:
                    with torch.no_grad():
                        f2 = torch.tensor(val_obs["features"], dtype=torch.float32, device=device)
                        qv2 = q_net(f2).detach().cpu().numpy()
                    qv2[~mask] = -1e9; va = int(np.argmax(qv2))
                val_obs, vr, vd, _ = val_env.step(va); val_total += vr; val_dec += 1
            if val_total > best_val_reward:
                best_val_reward = val_total
                torch.save({"model": q_net.state_dict(), "epsilon": epsilon}, best_path)
                print(f"[{label}] * New best ep={ep}: val_reward={val_total:.2f} val_makespan={val_env.time}", flush=True)

    torch.save({"model": q_net.state_dict(), "epsilon": epsilon}, out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_path if best_path.exists() else out_dir / f"{label}_model.pt"


def main():
    print("Loading data...", flush=True)
    all_instances = load_instances("basic_data.xlsx")
    train_insts = [all_instances[n] for n in TRAIN_INSTANCES]
    val_insts = [all_instances[n] for n in VAL_INSTANCES]
    test_insts = [all_instances[n] for n in TEST_INSTANCES]

    cfg = TrainConfig(episodes=EPISODES, seed=SEED)

    dqn_configs = [
        ("dqn_vanilla",  {"network_type": "pair",     "use_per": False, "use_reward_shaping": False, "use_curriculum": False}),
        ("dqn_per_rs",   {"network_type": "pair",     "use_per": True,  "use_reward_shaping": True,  "use_curriculum": False}),
        ("dqn_scaleinv", {"network_type": "scaleinv", "use_per": True,  "use_reward_shaping": True,  "use_curriculum": False}),
        ("dqn_full",     {"network_type": "scaleinv", "use_per": True,  "use_reward_shaping": True,  "use_curriculum": True}),
    ]

    for label, kwargs in dqn_configs:
        print(f"\n{'='*60}", flush=True)
        print(f"TRAINING: {label}", flush=True)
        print("="*60, flush=True)
        model_path = train_dqn_variant(train_insts, cfg, OUT_DIR / label, label=label, **kwargs)

        for split_name, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            print(f"  Evaluating {label} on {split_name}...", flush=True)
            rows = evaluate_model("dqn", model_path, insts, split_name, OUT_DIR / label)
            avg_mk = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split_name}: avg makespan={avg_mk:.0f}", flush=True)

    print("\nDQN ablations complete!", flush=True)


if __name__ == "__main__":
    main()
