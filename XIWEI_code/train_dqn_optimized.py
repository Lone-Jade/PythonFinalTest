"""
Optimized DQN Training — DQNConfigV2 with Noisy Nets, hard target updates,
wider network, N-step=7, cosine LR annealing, and higher gamma.

This script is a standalone counterpart to train_all.py, focusing on the
DQN architecture while keeping the same evaluation methodology for fair comparison.

Usage:
    python train_dqn_optimized.py --episodes 300     # 300-episode quick test
    python train_dqn_optimized.py --episodes 1000    # 1000-episode deep training
    python train_dqn_optimized.py --episodes 300 --datasets 10x5x3 10x10x6  # subset
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import torch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvConfig, DQNConfigV2
from utils import (
    load_csv_data, get_data_path, get_state_dim, get_action_dim,
    set_seed,
)
from environment import JSPEnvironment, GreedyScheduler
from agent import DQNAgent


# ─── All 10 datasets from Phase 3 + Phase 4 ─────────────────────────────────
DATASETS_PHASE3 = ["6x6x2", "10x10x3", "15x10x3", "20x10x3", "30x10x3"]
DATASETS_PHASE4 = ["10x5x3", "10x10x6", "15x10x2", "15x5x3", "20x5x3"]
ALL_DATASETS = DATASETS_PHASE3 + DATASETS_PHASE4


# ─── Training Episode (N-step returns) ─────────────────────────────────────
def train_episode(env: JSPEnvironment, agent: DQNAgent,
                  epsilon: float = None) -> Tuple[float, float, float, int]:
    """Run one training episode with N-step TD returns.

    Uses the agent's configured n_step for credit assignment. Returns are
    computed as discounted sums over n_step consecutive transitions, with
    bootstrapping from the target network when the episode doesn't end
    within the window.
    """
    n_step = agent.cfg.n_step
    gamma = agent.cfg.gamma
    state = env.reset()

    # Reset Noisy Net noise at episode start (if using Noisy Nets)
    agent.reset_noise()

    done = False
    total_reward = 0.0
    steps = 0

    # Episode trajectory buffer
    episode_buffer = []

    while not done:
        action_mask = env._get_action_mask()
        if not np.any(action_mask):
            break

        action = agent.select_action(state, action_mask, epsilon)
        next_state, reward, done, info = env.step(action)

        episode_buffer.append((state, action, reward, next_state, done))

        # N-step: when buffer has at least n_step entries, compute return
        if len(episode_buffer) >= n_step:
            s0, a0, _, _, _ = episode_buffer[0]

            G = 0.0
            episode_ended = False
            for i in range(n_step):
                _, _, ri, _, don_i = episode_buffer[i]
                G += (gamma ** i) * ri
                if don_i:
                    episode_ended = True
                    break

            if not episode_ended:
                sN = episode_buffer[n_step - 1][3]
                sN_tensor = torch.FloatTensor(sN).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_N = agent.q_network(sN_tensor).max(dim=1).values.item()
                G += (gamma ** n_step) * q_N

            agent.store(s0, a0, G, episode_buffer[n_step - 1][3],
                       any(episode_buffer[i][4] for i in range(n_step)))
            episode_buffer.pop(0)

        agent.update()
        total_reward += reward
        state = next_state
        steps += 1

    # Drain remaining buffer (terminal — no bootstrap)
    while len(episode_buffer) > 0:
        s0, a0, _, _, _ = episode_buffer[0]
        G = sum((gamma ** i) * episode_buffer[i][2]
                for i in range(len(episode_buffer)))
        agent.store(s0, a0, G, episode_buffer[-1][3], True)
        episode_buffer.pop(0)
        agent.update()

    return total_reward, env.get_makespan(), env.get_avg_fatigue(), steps


# ─── Evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(env: JSPEnvironment, agent: DQNAgent,
             num_episodes: int = 10) -> Dict[str, float]:
    """Greedy evaluation (epsilon=0, noise disabled)."""
    makespans, fatigues, rewards, step_counts = [], [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action_mask = env._get_action_mask()
            if not np.any(action_mask):
                break
            action = agent.select_action(state, action_mask, epsilon=0.0)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

        makespans.append(env.get_makespan())
        fatigues.append(env.get_avg_fatigue())
        rewards.append(ep_reward)
        step_counts.append(steps)

    return {
        "makespan_mean": float(np.mean(makespans)),
        "makespan_std": float(np.std(makespans)),
        "fatigue_mean": float(np.mean(fatigues)),
        "fatigue_std": float(np.std(fatigues)),
        "reward_mean": float(np.mean(rewards)),
        "avg_steps": float(np.mean(step_counts)),
    }


# ─── Baselines ──────────────────────────────────────────────────────────────
def run_baselines(data: Dict, env_config: EnvConfig) -> Dict:
    """Run baseline heuristics."""
    results = {}

    # Greedy SPT
    greedy = GreedyScheduler(data, env_config)
    ms, fat, _ = greedy.solve()
    results["greedy_makespan"] = ms
    results["greedy_fatigue"] = fat

    # Random (20 runs)
    env = JSPEnvironment(data, env_config)
    ms_list, fat_list = [], []
    for _ in range(20):
        env.reset()
        done = False
        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            valid = np.where(mask)[0]
            _, _, done, _ = env.step(int(np.random.choice(valid)))
        ms_list.append(env.get_makespan())
        fat_list.append(env.get_avg_fatigue())
    results["random_makespan"] = float(np.mean(ms_list))
    results["random_fatigue"] = float(np.mean(fat_list))

    # Round-Robin
    env.reset()
    done = False
    rr_worker = 0
    while not done:
        mask = env._get_action_mask()
        if not np.any(mask):
            break
        valid = np.where(mask)[0]
        rr_valid = [a for a in valid if a % env.num_workers == rr_worker]
        action = int(np.random.choice(rr_valid if rr_valid else valid))
        _, _, done, _ = env.step(action)
        rr_worker = (rr_worker + 1) % env.num_workers
    results["roundrobin_makespan"] = env.get_makespan()
    results["roundrobin_fatigue"] = env.get_avg_fatigue()

    return results


# ─── Load original DQN results (if available) ───────────────────────────────
def load_original_dqn_result(ds_name: str, episodes: int, log_dir: str) -> Dict:
    """Try to load original DQN evaluation from train_all.py logs."""
    log_path = os.path.join(log_dir, f"train_{ds_name}_dqn.json")
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        if log.get("episodes") and len(log["episodes"]) >= episodes:
            eval_data = log.get("final_eval", {})
            return {
                "makespan_mean": eval_data.get("makespan_mean",
                                                log.get("best_makespan", 0)),
                "makespan_std": eval_data.get("makespan_std", 0),
                "fatigue_mean": eval_data.get("fatigue_mean",
                                               log.get("best_fatigue", 0)),
                "fatigue_std": eval_data.get("fatigue_std", 0),
            }
    return None


# ─── JSON helper ────────────────────────────────────────────────────────────
def _to_native(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


# ─── Train DQN V2 on one dataset ───────────────────────────────────────────
def train_dqn_v2(data: Dict, env_config: EnvConfig, dqn_config: DQNConfigV2,
                 num_episodes: int, ds_name: str, result_dir: str,
                 verbose: bool = True) -> Tuple[DQNAgent, Dict, Dict]:
    """Train optimized DQN on a single dataset. Returns (agent, log, eval_result)."""
    N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
    env = JSPEnvironment(data, env_config)
    state_dim = get_state_dim(data)
    action_dim = get_action_dim(data)

    agent = DQNAgent(state_dim, action_dim, dqn_config)
    device = agent.device

    ep_rewards, ep_makespans, ep_fatigues = [], [], []
    best_ms = float('inf')
    best_fatigue = float('inf')
    no_improve = 0
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        total_r, ms, fat, steps = train_episode(env, agent)

        ep_rewards.append(float(total_r))
        ep_makespans.append(float(ms))
        ep_fatigues.append(float(fat))

        improved = False
        if ms < best_ms:
            best_ms = ms
            improved = True
        if fat < best_fatigue:
            best_fatigue = fat
            improved = True

        # Only count no-improvement during exploitation phase
        if agent._get_epsilon() < 0.3:
            no_improve = 0 if improved else no_improve + 1
        else:
            no_improve = 0

        agent.episodes_done += 1

        # Logging
        if verbose and (ep % 50 == 0 or ep == 1 or ep == num_episodes):
            elapsed = time.time() - start_time
            avg_r50 = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
            avg_ms50 = np.mean(ep_makespans[-50:]) if len(ep_makespans) >= 50 else np.mean(ep_makespans)
            eps_val = agent._get_epsilon()
            lr_now = agent.scheduler.get_last_lr()[0] if agent.scheduler else dqn_config.lr
            print(f"  Ep {ep:>5d}/{num_episodes} | eps={eps_val:.3f} "
                  f"lr={lr_now:.1e} | R50={avg_r50:>8.2f} | "
                  f"MS50={avg_ms50:>8.1f} | Best={best_ms:>8.1f} | {elapsed:.0f}s")

        # Early stopping
        if no_improve >= 500:
            if verbose:
                print(f"  Early stop at ep {ep} (no improvement for 500 eps)")
            break

    total_time = time.time() - start_time

    # Final evaluation
    ev = evaluate(env, agent, num_episodes=10)

    # Save checkpoint
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"dqn_v2_{ds_name}_ep{num_episodes}.pt"
    agent.save(os.path.join(ckpt_dir, ckpt_name), {
        'problem_size': {'N': N, 'M': M, 'W': W},
        'data_file': ds_name,
        'method': 'Dueling Double DQN V2 (NoisyNets + HardUpdate + N7 + CosLR)',
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hyperparams': {
            'lr': dqn_config.lr,
            'gamma': dqn_config.gamma,
            'n_step': dqn_config.n_step,
            'batch_size': dqn_config.batch_size,
            'hidden_dims': dqn_config.hidden_dims,
            'memory_capacity': dqn_config.memory_capacity,
            'use_noisy_nets': dqn_config.use_noisy_nets,
            'use_hard_update': dqn_config.use_hard_update,
            'hard_update_interval': dqn_config.hard_update_interval,
            'per_alpha': dqn_config.per_alpha,
            'epsilon_decay': dqn_config.epsilon_decay,
            'lr_decay_steps': dqn_config.lr_decay_steps,
        },
        'performance': {
            'best_makespan': best_ms,
            'best_fatigue': best_fatigue,
            'eval_makespan_mean': ev['makespan_mean'],
            'eval_makespan_std': ev['makespan_std'],
            'eval_fatigue_mean': ev['fatigue_mean'],
            'eval_fatigue_std': ev['fatigue_std'],
        },
        'training_time_s': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    # Convert all values to native Python types for JSON serialization
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return _to_native(obj.tolist())
        return obj

    training_log = {
        "problem": f"{N}x{M}x{W}",
        "method": "DQN_V2",
        "dataset": ds_name,
        "num_episodes": num_episodes,
        "config": {
            "lr": dqn_config.lr,
            "gamma": dqn_config.gamma,
            "n_step": dqn_config.n_step,
            "batch_size": dqn_config.batch_size,
            "hidden_dims": dqn_config.hidden_dims,
            "memory_capacity": dqn_config.memory_capacity,
            "use_noisy_nets": dqn_config.use_noisy_nets,
            "use_hard_update": dqn_config.use_hard_update,
            "hard_update_interval": dqn_config.hard_update_interval,
            "per_alpha": dqn_config.per_alpha,
            "epsilon_decay": dqn_config.epsilon_decay,
            "lr_decay_steps": dqn_config.lr_decay_steps,
        },
        "episodes": list(range(1, len(ep_rewards) + 1)),
        "rewards": [float(r) for r in ep_rewards],
        "makespans": [float(m) for m in ep_makespans],
        "fatigues": [float(f) for f in ep_fatigues],
        "best_makespan": float(best_ms),
        "best_fatigue": float(best_fatigue),
        "final_eval": _to_native(ev),
        "training_time_s": float(total_time),
    }

    return agent, training_log, ev


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train Optimized DQN (V2) on JSP scheduling"
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[300, 1000],
                        help="Training episodes (can specify multiple, e.g. 300 1000)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to use (default: all 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip final evaluation (faster)")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_DATASETS
    episode_list = args.episodes if isinstance(args.episodes, list) else [args.episodes]
    set_seed(args.seed)

    result_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(result_dir), "data")
    log_dir = os.path.join(result_dir, "logs")
    chart_dir = os.path.join(result_dir, "charts")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 72)
    print("Optimized DQN (V2) Multi-Scale Training")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Episode counts: {episode_list}")
    print(f"Device: {device}")
    print(f"Total runs: {len(datasets) * len(episode_list)}")
    print("=" * 72)
    print()
    print("V2 Improvements over baseline DQN:")
    print("  - Noisy Networks for state-dependent exploration")
    print("  - Hard target updates every 1000 steps")
    print("  - N-step=7 returns (faster credit propagation)")
    print("  - Gamma=0.99 (long-horizon value estimation)")
    print("  - 4-layer network [512,256,128,64] (more capacity)")
    print("  - Cosine LR annealing 1e-3 -> 1e-5")
    print("  - Larger batch (128) + replay buffer (100k)")
    print("  - Higher PER alpha (0.7)")
    print("=" * 72)

    env_config = EnvConfig()

    # ── Collect all results for summary ─────────────────────────────────────
    all_v2_results = {}  # ds_name -> {episodes: eval_dict}
    all_baselines = {}   # ds_name -> baseline_dict

    for ds_name in datasets:
        print(f"\n{'='*72}")
        print(f"[Dataset] {ds_name}")
        print(f"{'='*72}")

        data_path = get_data_path(data_dir, ds_name)
        data = load_csv_data(data_path)
        N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
        print(f"  Problem: {N} jobs x {M} machines x {W} workers")
        print(f"  State dim: {get_state_dim(data)}, Action dim: {get_action_dim(data)}")

        # Baselines (once per dataset)
        print(f"\n  Baselines:")
        baselines = run_baselines(data, env_config)
        all_baselines[ds_name] = baselines
        print(f"    Greedy SPT:     MS={baselines['greedy_makespan']:.1f}  "
              f"F={baselines['greedy_fatigue']:.3f}")
        print(f"    Random (20run): MS={baselines['random_makespan']:.1f}  "
              f"F={baselines['random_fatigue']:.3f}")
        print(f"    Round-Robin:    MS={baselines['roundrobin_makespan']:.1f}  "
              f"F={baselines['roundrobin_fatigue']:.3f}")

        ds_v2 = {}
        for num_ep in episode_list:
            print(f"\n  Training DQN V2 ({num_ep} episodes)")
            print(f"  {'-'*50}")
            t0 = time.time()

            dqn_config = DQNConfigV2()
            agent, log, ev = train_dqn_v2(
                data, env_config, dqn_config, num_ep, ds_name, result_dir
            )

            elapsed = time.time() - t0
            print(f"\n  DQN V2 ({num_ep}ep) Final: "
                  f"MS={ev['makespan_mean']:.1f}+-{ev['makespan_std']:.1f}  "
                  f"F={ev['fatigue_mean']:.3f}+-{ev['fatigue_std']:.3f}  "
                  f"Time={elapsed:.0f}s")

            # Load original DQN result for comparison
            orig = load_original_dqn_result(ds_name, num_ep, log_dir)
            if orig:
                ms_diff = (ev['makespan_mean'] - orig['makespan_mean']) / orig['makespan_mean'] * 100
                print(f"  vs Original DQN: MS={orig['makespan_mean']:.1f}  "
                      f"Delta={ms_diff:+.1f}%")
            else:
                ms_diff = None
                print(f"  vs Original DQN: (no previous result for {ds_name} at {num_ep}ep)")

            # Save training log
            log['baselines'] = {k: float(v) for k, v in baselines.items()}
            log['original_dqn'] = _to_native(orig)
            log['improvement_vs_original_dqn_pct'] = float(ms_diff) if ms_diff is not None else None

            log_path = os.path.join(log_dir, f"train_{ds_name}_dqn_v2_ep{num_ep}.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(_to_native(log), f, indent=2, ensure_ascii=False)
            print(f"  Log saved: logs/train_{ds_name}_dqn_v2_ep{num_ep}.json")

            ds_v2[num_ep] = {
                'eval': ev,
                'original': orig,
                'improvement_pct': ms_diff,
                'best_ms': log['best_makespan'],
            }

        all_v2_results[ds_name] = ds_v2

    # ── Comprehensive Summary ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Comprehensive Summary — DQN V2 vs Original DQN vs Baselines")
    print(f"{'='*72}")

    for num_ep in episode_list:
        print(f"\n{'─'*72}")
        print(f"  {num_ep}-Episode Results")
        print(f"{'─'*72}")
        print(f"  {'Dataset':<12} {'Greedy':>10} {'DQN_orig':>10} {'DQN_V2':>10} "
              f"{'V2vsOrig':>9} {'V2vsGreedy':>10}")
        print(f"  {'-'*62}")

        for ds_name in datasets:
            if ds_name not in all_v2_results:
                continue
            bl = all_baselines[ds_name]
            ds_data = all_v2_results[ds_name]
            if num_ep not in ds_data:
                continue

            v2_ms = ds_data[num_ep]['eval']['makespan_mean']
            orig = ds_data[num_ep]['original']
            greedy_ms = bl['greedy_makespan']

            dqn_orig_str = f"{orig['makespan_mean']:.1f}" if orig else "N/A"
            v2_vs_orig_str = f"{ds_data[num_ep]['improvement_pct']:+.1f}%" if ds_data[num_ep]['improvement_pct'] is not None else "N/A"
            v2_vs_greedy = (v2_ms - greedy_ms) / greedy_ms * 100

            print(f"  {ds_name:<12} {greedy_ms:>10.1f} {dqn_orig_str:>10} "
                  f"{v2_ms:>10.1f} {v2_vs_orig_str:>9} {v2_vs_greedy:>+9.1f}%")

    # ── Save combined V2 results ─────────────────────────────────────────────
    combined_path = os.path.join(log_dir, "dqn_v2_combined_results.json")
    combined_output = {}
    for ds_name, ep_dict in all_v2_results.items():
        combined_output[ds_name] = {}
        for ep, data in ep_dict.items():
            combined_output[ds_name][str(ep)] = {
                'makespan_mean': data['eval']['makespan_mean'],
                'makespan_std': data['eval']['makespan_std'],
                'fatigue_mean': data['eval']['fatigue_mean'],
                'fatigue_std': data['eval']['fatigue_std'],
                'best_makespan': data['best_ms'],
                'original_dqn_makespan': data['original']['makespan_mean'] if data['original'] else None,
                'improvement_vs_original_pct': data['improvement_pct'],
            }
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(_to_native(combined_output), f, indent=2, ensure_ascii=False)
    print(f"\nCombined V2 results saved: logs/dqn_v2_combined_results.json")

    print(f"\n[OK] DQN V2 training complete!")


if __name__ == "__main__":
    main()
