"""Extra ablation variants: 2 DQN (50ep) + 2 PPO (150ep)."""
import json, csv, os, random, sys, numpy as np
from pathlib import Path

# Ensure the script's directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from data_loader import load_instances
from config import EnvConfig, TrainConfig
from env import JobShopFatigueEnv
from heuristics import select_action
from models import (PairScoringNetwork, DuelingPairScoringNetwork,
                     ScaleInvariantDuelingNetwork, ActorCriticNetwork,
                     ScaleInvariantActorCritic)
from agents import DQNAgent, PPOAgent, Transition, PrioritizedReplayBuffer
from test import choose_dqn_action, choose_ppo_action, env_metrics, load_model

SEED = 42
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

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(inst, ecfg, s):
    return JobShopFatigueEnv(inst, ecfg, seed=s)

def curriculum_stages(instances, episodes):
    small, medium, large = [], [], []
    for inst in instances:
        t = inst.n_jobs * inst.n_machines
        if t <= 50: small.append(inst)
        elif t <= 100: medium.append(inst)
        else: large.append(inst)
    s1 = episodes // 3; s2 = 2 * episodes // 3
    return [(1, s1, small), (s1+1, s2, small+medium), (s2+1, episodes, small+medium+large)]

def write_rows(rows, d, name):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (d / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def evaluate(algo, mp, instances, split, out_dir):
    pe = make_env(instances[0], EnvConfig(), SEED)
    model, device = load_model(algo, mp, pe.feature_dim, TrainConfig.hidden_dim)
    rows = []
    for idx, inst in enumerate(instances):
        env = make_env(inst, EnvConfig(), SEED + idx)
        obs = env.reset(); total = 0.0; dec = 0; np_ = 0; lt = 0
        while not obs["done"] and dec < TrainConfig.max_decisions:
            if algo == "dqn": action = choose_dqn_action(model, obs, device)
            else: action = choose_ppo_action(model, obs, device)
            if action < 0 or action >= len(obs["mask"]) or not obs["mask"][action] or np_ >= 20:
                action = select_action(obs, policy="rest_aware")
            obs, reward, _, _ = env.step(action)
            total += reward; dec += 1
            if len(env.history) == lt: np_ += 1
            else: np_ = 0; lt = len(env.history)
        rows.append(env_metrics(env, total, dec, algo, mp.name))
    write_rows(rows, out_dir / split, f"{algo}_{mp.stem}")
    return rows


class UniformReplayBuffer:
    def __init__(self, cap):
        self.cap = cap; self.buf = []; self.pos = 0; self.size = 0
    def push(self, item):
        if self.size < self.cap: self.buf.append(item)
        else: self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap; self.size = min(self.size + 1, self.cap)
    def sample(self, bs):
        if self.size < bs: return None, None, None
        idx = np.random.choice(self.size, bs, replace=False)
        return [self.buf[i] for i in idx], idx, np.ones(bs, dtype=np.float32)
    def update_priorities(self, *a): pass
    def __len__(self): return self.size


# ==================== DQN ====================

def train_dqn(instances, cfg, out_dir, *, network_type, use_per,
              use_reward_shaping, use_curriculum, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd = make_env(instances[0], EnvConfig(), 0).feature_dim
    ecfg = EnvConfig()
    if not use_reward_shaping:
        ecfg.s_job_completion = 0.0; ecfg.s_efficiency = 0.0
        ecfg.s_progress = 0.0; ecfg.s_stall = 0.0

    nc = {"pair": PairScoringNetwork, "dueling": DuelingPairScoringNetwork,
          "scaleinv": ScaleInvariantDuelingNetwork}[network_type]
    qn = nc(fd, cfg.hidden_dim).to(device); tn = nc(fd, cfg.hidden_dim).to(device)
    tn.load_state_dict(qn.state_dict())
    opt = torch.optim.Adam(qn.parameters(), lr=cfg.lr)
    buf = PrioritizedReplayBuffer(cfg.replay_size, cfg.per_alpha, cfg.per_beta) if use_per else UniformReplayBuffer(cfg.replay_size)
    eps = cfg.epsilon_start; steps = 0; logs = []

    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [(1, cfg.episodes, instances)]
    print(f"[{label}] Curriculum: {[(s,e,len(i)) for s,e,i in stages]}", flush=True)

    best_vr = -float("inf"); best_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        si = instances
        for ss, se, sl in stages:
            if ss <= ep <= se: si = sl; break
        inst = si[(ep - 1) % len(si)]
        env = make_env(inst, ecfg, cfg.seed + ep)
        obs = env.reset(); total = 0.0; losses = []; decisions = 0; transitions = []

        while not obs["done"] and decisions < cfg.max_decisions:
            mask = obs["mask"]; legal = np.flatnonzero(mask)
            if len(legal) == 0: action = 0
            elif random.random() < eps: action = int(random.choice(legal))
            else:
                with torch.no_grad():
                    f = torch.tensor(obs["features"], dtype=torch.float32, device=device)
                    qv = qn(f).detach().cpu().numpy()
                qv[~mask] = -1e9; action = int(np.argmax(qv))
            fc = obs["features"].copy(); mc = obs["mask"].copy()
            no, rw, dn, _ = env.step(action)
            transitions.append({"features": fc, "mask": mc, "action": action,
                "reward": rw, "next_features": no["features"].copy(),
                "next_mask": no["mask"].copy(), "done": dn})
            obs = no; total += rw; decisions += 1

        n = min(cfg.n_step, len(transitions))
        for i in range(len(transitions)):
            t = transitions[i]; nsr = 0.0; tin = False
            end = min(i + n, len(transitions))
            for j in range(i, end):
                nsr += (cfg.gamma ** (j - i)) * transitions[j]["reward"]
                if transitions[j]["done"]: tin = True; break
            if not tin and end < len(transitions):
                b = transitions[end]
                with torch.no_grad():
                    nf = torch.tensor(b["features"], dtype=torch.float32, device=device)
                    qn2 = tn(nf)
                    mk = torch.tensor(b["mask"], dtype=torch.bool, device=device)
                    qn2 = qn2.masked_fill(~mk, -1e9)
                    nsr += (cfg.gamma ** n) * float(qn2.max().item())
            buf.push(Transition(features=t["features"], action=t["action"], reward=nsr,
                next_features=np.zeros((0,), dtype=np.float32),
                next_mask=np.zeros((0,), dtype=bool), done=tin, is_n_step=True))

        for _ in range(min(decisions, 200)):
            r = buf.sample(cfg.batch_size)
            if r[0] is None: continue
            batch, indices, iw = r
            ch = torch.tensor(np.stack([b.features[b.action] for b in batch]), dtype=torch.float32, device=device)
            wt = torch.tensor(iw, dtype=torch.float32, device=device)
            with torch.no_grad():
                tg = []
                for b in batch:
                    if b.is_n_step: tg.append(b.reward)
                    elif b.done or len(b.next_features) == 0 or not b.next_mask.any(): tg.append(b.reward)
                    else:
                        nf = torch.tensor(b.next_features, dtype=torch.float32, device=device)
                        qn3 = tn(nf); mk2 = torch.tensor(b.next_mask, dtype=torch.bool, device=device)
                        qn3 = qn3.masked_fill(~mk2, -1e9)
                        tg.append(b.reward + cfg.gamma * float(qn3.max().item()))
                target = torch.tensor(tg, dtype=torch.float32, device=device)
            pred = qn(ch); td = (pred - target).detach().cpu().numpy()
            loss = (wt * torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qn.parameters(), 1.0); opt.step()
            buf.update_priorities(indices, td); steps += 1
            if steps % cfg.target_update == 0: tn.load_state_dict(qn.state_dict())
            losses.append(float(loss.item()))

        eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)
        if cfg.lr_decay < 1.0:
            for pg in opt.param_groups: pg["lr"] *= cfg.lr_decay
        logs.append({"episode": ep, "instance": inst.name, "reward": total,
                     "makespan": env.time, "decisions": decisions, "epsilon": eps,
                     "loss": float(np.mean(losses)) if losses else None})
        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} mk={env.time} eps={eps:.3f}", flush=True)
        if ep % 50 == 0 or ep == cfg.episodes:
            vi = instances[min(len(instances)-1, 3)]
            ve = make_env(vi, ecfg, cfg.seed + 99999)
            vo = ve.reset(); vt = 0.0; vd = 0
            while not vo["done"] and vd < cfg.max_decisions:
                vm = vo["mask"]; vl = np.flatnonzero(vm)
                if len(vl) == 0: va = 0
                else:
                    with torch.no_grad():
                        vf = torch.tensor(vo["features"], dtype=torch.float32, device=device)
                        vq = qn(vf).detach().cpu().numpy()
                    vq[~vm] = -1e9; va = int(np.argmax(vq))
                vo, vr_, vd_, _ = ve.step(va); vt += vr_; vd += 1
            if vt > best_vr:
                best_vr = vt
                torch.save({"model": qn.state_dict(), "epsilon": eps}, best_path)
                print(f"[{label}] * Best ep={ep}: val_r={vt:.2f} val_mk={ve.time}", flush=True)

    torch.save({"model": qn.state_dict(), "epsilon": eps}, out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_path if best_path.exists() else out_dir / f"{label}_model.pt"


# ==================== PPO ====================

def train_ppo(instances, cfg, out_dir, *, network_type, use_reward_norm,
              use_reward_shaping, use_curriculum, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd = make_env(instances[0], EnvConfig(), 0).feature_dim
    ecfg = EnvConfig()
    if not use_reward_shaping:
        ecfg.s_job_completion = 0.0; ecfg.s_efficiency = 0.0
        ecfg.s_progress = 0.0; ecfg.s_stall = 0.0

    if network_type == "scaleinv":
        net = ScaleInvariantActorCritic(fd, cfg.hidden_dim).to(device)
    else:
        net = ActorCriticNetwork(fd, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    ret_mean = 0.0; ret_std = 1.0; ret_mm = 0.01 if use_reward_norm else 0.0
    logs = []
    stages = curriculum_stages(instances, cfg.episodes) if use_curriculum else [(1, cfg.episodes, instances)]
    print(f"[{label}] Curriculum: {[(s,e,len(i)) for s,e,i in stages]}", flush=True)

    best_vr = -float("inf"); best_path = out_dir / f"{label}_best.pt"

    for ep in range(1, cfg.episodes + 1):
        si = instances
        for ss, se, sl in stages:
            if ss <= ep <= se: si = sl; break
        inst = si[(ep - 1) % len(si)]
        env = make_env(inst, ecfg, cfg.seed + ep)
        obs = env.reset(); rollout = []; total = 0.0; decisions = 0

        while not obs["done"] and decisions < cfg.max_decisions:
            features = torch.tensor(obs["features"], dtype=torch.float32, device=device)
            mask = torch.tensor(obs["mask"], dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, value = net(features, mask)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            item = {"features": obs["features"].copy(), "mask": obs["mask"].copy(),
                    "action": int(action.item()), "logp": float(logp.item()),
                    "value": float(value.item())}
            next_obs, reward, done, _ = env.step(int(action.item()))
            item["reward"] = reward; item["done"] = done; rollout.append(item)
            obs = next_obs; total += reward; decisions += 1
            if len(rollout) >= cfg.rollout_steps:
                # PPO update
                ret_mean, ret_std = _ppo_update(rollout, net, opt, cfg, device, ret_mean, ret_std, ret_mm)
                rollout = []

        if rollout:
            ret_mean, ret_std = _ppo_update(rollout, net, opt, cfg, device, ret_mean, ret_std, ret_mm)
        if cfg.lr_decay < 1.0:
            for pg in opt.param_groups: pg["lr"] *= cfg.lr_decay
        logs.append({"episode": ep, "instance": inst.name, "reward": total,
                     "makespan": env.time, "decisions": decisions})
        if ep == 1 or ep % 50 == 0:
            print(f"[{label}] ep={ep} reward={total:.2f} mk={env.time}", flush=True)
        if ep % 50 == 0 or ep == cfg.episodes:
            vi = instances[min(len(instances)-1, 3)]
            ve = make_env(vi, ecfg, cfg.seed + 99999)
            vo = ve.reset(); vt = 0.0; vd2 = 0
            while not vo["done"] and vd2 < cfg.max_decisions:
                vf = torch.tensor(vo["features"], dtype=torch.float32, device=device)
                vm2 = torch.tensor(vo["mask"], dtype=torch.bool, device=device)
                with torch.no_grad():
                    vl2, _ = net(vf, vm2)
                    va = int(torch.argmax(vl2).item())
                vo, vr_, vd_, _ = ve.step(va); vt += vr_; vd2 += 1
            if vt > best_vr:
                best_vr = vt
                torch.save({"model": net.state_dict()}, best_path)
                print(f"[{label}] * Best ep={ep}: val_r={vt:.2f} val_mk={ve.time}", flush=True)

    torch.save({"model": net.state_dict()}, out_dir / f"{label}_model.pt")
    write_rows(logs, out_dir, f"{label}_train_log")
    return best_path if best_path.exists() else out_dir / f"{label}_model.pt"


def _ppo_update(rollout, net, opt, cfg, device, ret_mean, ret_std, ret_mm):
    rewards = [x["reward"] for x in rollout]
    dones = [x["done"] for x in rollout]
    values = [x["value"] for x in rollout]
    returns = []; g = 0.0
    for rw, dn in zip(reversed(rewards), reversed(dones)):
        g = rw + cfg.gamma * g * (1.0 - float(dn)); returns.append(g)
    returns.reverse()
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    val_t = torch.tensor(values, dtype=torch.float32, device=device)

    if ret_mm > 0:
        bm = float(ret_t.mean().item()); bs = float(ret_t.std().item()) + 1e-8
        ret_mean = (1 - ret_mm) * ret_mean + ret_mm * bm
        ret_std = (1 - ret_mm) * ret_std + ret_mm * bs
        ret_t = (ret_t - ret_mean) / ret_std

    adv_t = ret_t - val_t
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for i, item in enumerate(rollout):
        item["_return"] = ret_t[i]; item["_adv"] = adv_t[i]

    for _ in range(cfg.ppo_epochs):
        random.shuffle(rollout)
        for item in rollout:
            f = torch.tensor(item["features"], dtype=torch.float32, device=device)
            m = torch.tensor(item["mask"], dtype=torch.bool, device=device)
            a = torch.tensor(item["action"], dtype=torch.int64, device=device)
            ol = torch.tensor(item["logp"], dtype=torch.float32, device=device)
            lg, v = net(f, m)
            d = torch.distributions.Categorical(logits=lg)
            lp = d.log_prob(a); ratio = torch.exp(lp - ol)
            adv = item["_adv"]
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv)
            vl = torch.nn.functional.mse_loss(v, item["_return"])
            el = d.entropy()
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * el
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()

    return ret_mean, ret_std


# ==================== Main ====================

def main():
    print("Loading data...", flush=True)
    all_instances = load_instances("basic_data.xlsx")
    train_insts = [all_instances[n] for n in TRAIN_INSTANCES]
    val_insts = [all_instances[n] for n in VAL_INSTANCES]
    test_insts = [all_instances[n] for n in TEST_INSTANCES]

    # ── DQN extra variants (50ep) ──
    dqn_variants = [
        ("dqn_scaleinv_no_per", 50, {"network_type": "scaleinv", "use_per": False,
                                      "use_reward_shaping": True, "use_curriculum": False}),
        ("dqn_pair_curr",       50, {"network_type": "pair",     "use_per": True,
                                      "use_reward_shaping": True, "use_curriculum": True}),
    ]

    for label, episodes, kwargs in dqn_variants:
        print(f"\n{'='*60}\nTRAINING DQN: {label} ({episodes}ep)\n{'='*60}", flush=True)
        cfg = TrainConfig(episodes=episodes, seed=SEED)
        mp = train_dqn(train_insts, cfg, OUT_DIR / label, label=label, **kwargs)
        for split, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            rows = evaluate("dqn", mp, insts, split, OUT_DIR / label)
            avg = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split}: avg mk={avg:.0f}", flush=True)

    # ── PPO core variants (150ep) ──
    ppo_variants = [
        ("ppo_scaleinv_150", 150, {"network_type": "scaleinv", "use_reward_norm": False,
                                    "use_reward_shaping": False, "use_curriculum": False}),
        ("ppo_full_norn_150", 150, {"network_type": "scaleinv", "use_reward_norm": False,
                                     "use_reward_shaping": True, "use_curriculum": True}),
    ]

    for label, episodes, kwargs in ppo_variants:
        print(f"\n{'='*60}\nTRAINING PPO: {label} ({episodes}ep)\n{'='*60}", flush=True)
        cfg = TrainConfig(episodes=episodes, seed=SEED)
        mp = train_ppo(train_insts, cfg, OUT_DIR / label, label=label, **kwargs)
        for split, insts in [("train", train_insts), ("val", val_insts), ("test", test_insts)]:
            rows = evaluate("ppo", mp, insts, split, OUT_DIR / label)
            avg = np.mean([r["makespan"] for r in rows])
            print(f"  [{label}] {split}: avg mk={avg:.0f}", flush=True)

    print("\nAll extra ablations complete!", flush=True)


if __name__ == "__main__":
    main()
