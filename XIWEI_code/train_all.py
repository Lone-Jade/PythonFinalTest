"""
Master training script: trains DQN + PPO on 5 datasets at different scales,
then runs comprehensive evaluation and generates comparison charts.

Usage:
    python train_all.py                        # train all 5 scales, 300 episodes
    python train_all.py --episodes 500         # longer training
    python train_all.py --skip_train           # only evaluate existing models
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List

import numpy as np
import torch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import (
    load_csv_data, get_data_path, get_state_dim, get_action_dim,
    set_seed, find_csv_files,
)
from environment import JSPEnvironment, GreedyScheduler
from agent import DQNAgent
from agent_ppo import PPOAgent
from train_baselines import run_greedy, run_random, run_roundrobin
from visualize import (
    set_style, plot_method_comparison, plot_dataset_comparison,
    plot_multi_run_comparison, plot_training_curves,
    load_training_log, save_training_log, find_logs,
)

# ─── Configuration ───────────────────────────────────────────────────────────
DATASETS = ["6x6x2", "10x10x3", "15x10x3", "20x10x3", "30x10x3"]
EPISODES = 300  # Quick training for comparison
EVAL_RUNS = 10  # Evaluation episodes per method


# ─── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_agent(env, agent, num_runs=EVAL_RUNS):
    """Evaluate any agent greedily. Returns metrics dict."""
    makespans, fatigues, rewards = [], [], []
    is_ppo = isinstance(agent, PPOAgent)

    for _ in range(num_runs):
        env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            if is_ppo:
                action, _ = agent.evaluate(env._get_state(), mask)
            else:
                action = agent.select_action(env._get_state(), mask, epsilon=0.0)
            _, reward, done, _ = env.step(action)
            ep_reward += reward
        makespans.append(env.get_makespan())
        fatigues.append(env.get_avg_fatigue())
        rewards.append(ep_reward)

    return {
        "makespan_mean": float(np.mean(makespans)),
        "makespan_std": float(np.std(makespans)),
        "fatigue_mean": float(np.mean(fatigues)),
        "fatigue_std": float(np.std(fatigues)),
        "reward_mean": float(np.mean(rewards)),
    }


# ─── Training ────────────────────────────────────────────────────────────────
def train_dqn_quick(data, env_config, dqn_config, num_episodes, verbose=True):
    """Quick-train a DQN agent and return with training log."""
    env = JSPEnvironment(data, env_config)
    state_dim = get_state_dim(data)
    action_dim = get_action_dim(data)
    agent = DQNAgent(state_dim, action_dim, dqn_config)
    device = agent.device

    ep_rewards, ep_makespans = [], []
    best_ms = float('inf')
    best_fatigue = float('inf')
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        # N-step training episode (simplified: standard TD(0) for speed)
        state = env.reset()
        done = False
        ep_r, steps = 0.0, 0

        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            action = agent.select_action(state, mask)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()
            ep_r += reward
            state = next_state
            steps += 1

        agent.episodes_done += 1
        ms = env.get_makespan()
        fat = env.get_avg_fatigue()

        ep_rewards.append(float(ep_r))
        ep_makespans.append(float(ms))
        if ms < best_ms:
            best_ms = ms
        if fat < best_fatigue:
            best_fatigue = fat

        if verbose and ep % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    Ep {ep:>4d}/{num_episodes}: eps={agent._get_epsilon():.3f}  "
                  f"ms={ms:.1f}  best={best_ms:.1f}  {elapsed:.0f}s")

    training_log = {
        "problem": f"{data['num_jobs']}x{data['num_machines']}x{data['num_workers']}",
        "method": "DQN",
        "config": {
            "n_step": dqn_config.n_step, "gamma": dqn_config.gamma,
            "lr": dqn_config.lr, "lambda_fatigue": env_config.lambda_fatigue,
            "use_per": dqn_config.use_per,
        },
        "episodes": list(range(1, num_episodes + 1)),
        "rewards": ep_rewards,
        "makespans": ep_makespans,
        "best_makespan": float(best_ms),
        "best_fatigue": float(best_fatigue),
    }
    return agent, training_log


def train_ppo_quick(data, env_config, ppo_config, num_episodes, verbose=True):
    """Quick-train a PPO agent and return with training log."""
    env = JSPEnvironment(data, env_config)
    state_dim = get_state_dim(data)
    action_dim = get_action_dim(data)
    agent = PPOAgent(state_dim, action_dim, ppo_config)

    ep_rewards, ep_makespans = [], []
    best_ms = float('inf')
    best_fatigue = float('inf')
    start_time = time.time()
    rollout_episodes = ppo_config.rollout_episodes

    ep = 0
    while ep < num_episodes:
        trajectories = []
        for _ in range(rollout_episodes):
            if ep >= num_episodes:
                break
            ep += 1
            agent.episodes_done += 1
            traj = agent.collect_episode(env)
            trajectories.append(traj)
            ep_rewards.append(float(traj['total_reward']))
            ep_makespans.append(float(traj['makespan']))
            if traj['makespan'] < best_ms:
                best_ms = traj['makespan']
            if traj['fatigue'] < best_fatigue:
                best_fatigue = traj['fatigue']

        if trajectories:
            agent.update(trajectories)

        if verbose and ep % 100 == 0:
            elapsed = time.time() - start_time
            avg_ms = np.mean(ep_makespans[-50:]) if ep_makespans else 0
            print(f"    Ep {ep:>4d}/{num_episodes}: "
                  f"ms50={avg_ms:.1f}  best={best_ms:.1f}  {elapsed:.0f}s")

    training_log = {
        "problem": f"{data['num_jobs']}x{data['num_machines']}x{data['num_workers']}",
        "method": "PPO",
        "config": {
            "lr": ppo_config.lr, "gamma": ppo_config.gamma,
            "gae_lambda": ppo_config.gae_lambda, "clip_epsilon": ppo_config.clip_epsilon,
            "entropy_coef": ppo_config.entropy_coef,
        },
        "episodes": list(range(1, num_episodes + 1)),
        "rewards": ep_rewards,
        "makespans": ep_makespans,
        "best_makespan": float(best_ms),
        "best_fatigue": float(best_fatigue),
    }
    return agent, training_log


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train all methods on 5 scales")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help=f"Training episodes per dataset (default: {EPISODES})")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--skip_dqn", action="store_true", help="Skip DQN training")
    parser.add_argument("--skip_ppo", action="store_true", help="Skip PPO training")
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS,
                        help="Datasets to use (default: 5 scales)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = args.datasets
    num_episodes = args.episodes
    set_seed(args.seed)
    config = Config()
    set_style()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    result_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    log_dir = os.path.join(result_dir, "logs")
    chart_dir = os.path.join(result_dir, "charts")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 72)
    print("Multi-Scale Multi-Method JSP Scheduling Training")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Episodes per run: {num_episodes}")
    print(f"Device: {device}")
    print(f"Total training runs: {len(datasets) * 2} (DQN + PPO each)")
    print("=" * 72)

    # ── Storage for all results ──────────────────────────────────────────────
    all_results = {}  # dataset -> {method: metrics}
    all_training_logs = {}  # dataset -> {method: log_dict}

    for ds_name in datasets:
        print(f"\n{'='*72}")
        print(f"[Dataset] {ds_name}")
        print(f"{'='*72}")

        data_path = get_data_path(data_dir, ds_name)
        data = load_csv_data(data_path)
        N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
        print(f"  Problem: {N} jobs × {M} machines × {W} workers")
        print(f"  State dim: {get_state_dim(data)}, Action dim: {get_action_dim(data)}")

        t_start = time.time()
        ds_results = {}

        # ── Baselines ──────────────────────────────────────────────────────────
        print(f"\n  [1/4] Baselines")
        t0 = time.time()
        greedy = run_greedy(data, config.env)
        rr = run_roundrobin(data, config.env)
        rand = run_random(data, config.env, num_runs=30)
        print(f"    Greedy SPT:     MS={greedy['makespan']:.1f}  F={greedy['fatigue']:.3f}")
        print(f"    Round-Robin:    MS={rr['makespan']:.1f}  F={rr['fatigue']:.3f}")
        print(f"    Random (30run): MS={rand['makespan_mean']:.1f}±{rand['makespan_std']:.0f}  "
              f"F={rand['fatigue_mean']:.3f}")
        print(f"    Time: {time.time()-t0:.1f}s")

        ds_results['Greedy'] = {'makespan': greedy['makespan'], 'fatigue': greedy['fatigue']}
        ds_results['RoundRobin'] = {'makespan': rr['makespan'], 'fatigue': rr['fatigue']}
        ds_results['Random'] = {'makespan': rand['makespan_mean'],
                                'makespan_std': rand['makespan_std'],
                                'fatigue': rand['fatigue_mean'],
                                'fatigue_std': rand['fatigue_std']}

        # ── DQN Training ───────────────────────────────────────────────────────
        if not args.skip_train and not args.skip_dqn:
            print(f"\n  [2/4] Training DQN ({num_episodes} episodes)")
            t0 = time.time()
            env = JSPEnvironment(data, config.env)
            dqn_agent, dqn_log = train_dqn_quick(data, config.env, config.dqn, num_episodes)
            dqn_eval = evaluate_agent(env, dqn_agent)
            print(f"    DQN Eval: MS={dqn_eval['makespan_mean']:.1f}±{dqn_eval['makespan_std']:.1f}  "
                  f"F={dqn_eval['fatigue_mean']:.3f}  Time: {time.time()-t0:.1f}s")

            # Save checkpoint
            dqn_ckpt_path = os.path.join(ckpt_dir, f"dqn_{ds_name}_ep{num_episodes}.pt")
            dqn_agent.save(dqn_ckpt_path, {
                'problem_size': {'N': N, 'M': M, 'W': W},
                'method': 'Dueling Double DQN + PER',
                'performance': {
                    'best_makespan': dqn_log['best_makespan'],
                    'eval_makespan_mean': dqn_eval['makespan_mean'],
                    'eval_fatigue_mean': dqn_eval['fatigue_mean'],
                },
                'hyperparams': {'lr': config.dqn.lr, 'gamma': config.dqn.gamma,
                                'n_step': config.dqn.n_step},
            })

            # Save log
            dqn_log['baselines'] = {
                'greedy_makespan': greedy['makespan'],
                'greedy_fatigue': greedy['fatigue'],
                'random_makespan': rand['makespan_mean'],
                'random_fatigue': rand['fatigue_mean'],
                'roundrobin_makespan': rr['makespan'],
                'roundrobin_fatigue': rr['fatigue'],
            }
            dqn_log['final_eval'] = dqn_eval
            log_path = os.path.join(log_dir, f"train_{ds_name}_dqn.json")
            save_training_log(dqn_log, log_path)

            ds_results['DQN'] = {
                'makespan': dqn_eval['makespan_mean'],
                'makespan_std': dqn_eval['makespan_std'],
                'fatigue': dqn_eval['fatigue_mean'],
                'fatigue_std': dqn_eval['fatigue_std'],
                'best_makespan': dqn_log['best_makespan'],
            }
            all_training_logs.setdefault(ds_name, {})['DQN'] = dqn_log
        else:
            print(f"\n  [2/4] DQN: SKIPPED")

        # ── PPO Training ───────────────────────────────────────────────────────
        if not args.skip_train and not args.skip_ppo:
            print(f"\n  [3/4] Training PPO ({num_episodes} episodes)")
            t0 = time.time()
            env = JSPEnvironment(data, config.env)
            ppo_agent, ppo_log = train_ppo_quick(data, config.env, config.ppo, num_episodes)
            ppo_eval = evaluate_agent(env, ppo_agent)
            print(f"    PPO Eval: MS={ppo_eval['makespan_mean']:.1f}±{ppo_eval['makespan_std']:.1f}  "
                  f"F={ppo_eval['fatigue_mean']:.3f}  Time: {time.time()-t0:.1f}s")

            # Save checkpoint
            ppo_ckpt_path = os.path.join(ckpt_dir, f"ppo_{ds_name}_ep{num_episodes}.pt")
            ppo_agent.save(ppo_ckpt_path, {
                'problem_size': {'N': N, 'M': M, 'W': W},
                'method': 'PPO (Actor-Critic + GAE)',
                'performance': {
                    'best_makespan': ppo_log['best_makespan'],
                    'eval_makespan_mean': ppo_eval['makespan_mean'],
                    'eval_fatigue_mean': ppo_eval['fatigue_mean'],
                },
                'hyperparams': {'lr': config.ppo.lr, 'gamma': config.ppo.gamma,
                                'gae_lambda': config.ppo.gae_lambda},
            })

            # Save log
            ppo_log['baselines'] = {
                'greedy_makespan': greedy['makespan'],
                'greedy_fatigue': greedy['fatigue'],
                'random_makespan': rand['makespan_mean'],
                'random_fatigue': rand['fatigue_mean'],
                'roundrobin_makespan': rr['makespan'],
                'roundrobin_fatigue': rr['fatigue'],
            }
            ppo_log['final_eval'] = ppo_eval
            log_path = os.path.join(log_dir, f"train_{ds_name}_ppo.json")
            save_training_log(ppo_log, log_path)

            ds_results['PPO'] = {
                'makespan': ppo_eval['makespan_mean'],
                'makespan_std': ppo_eval['makespan_std'],
                'fatigue': ppo_eval['fatigue_mean'],
                'fatigue_std': ppo_eval['fatigue_std'],
                'best_makespan': ppo_log['best_makespan'],
            }
            all_training_logs.setdefault(ds_name, {})['PPO'] = ppo_log
        else:
            print(f"\n  [3/4] PPO: SKIPPED")

        all_results[ds_name] = ds_results
        print(f"\n  [4/4] Dataset {ds_name} total time: {time.time()-t_start:.1f}s")

    # ── Generate Charts ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Generating Comparison Charts")
    print(f"{'='*72}")

    # Method comparison per dataset
    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        results = all_results[ds_name]
        plot_method_comparison(
            results,
            title=f"Method Comparison — {ds_name} ({num_episodes} episodes)",
            save_path=os.path.join(chart_dir, f"method_comparison_{ds_name}.png"),
        )

    # Cross-dataset comparison (Makespan)
    plot_dataset_comparison(
        all_results, metric='makespan',
        save_path=os.path.join(chart_dir, "dataset_comparison_makespan.png"),
    )

    # Cross-dataset comparison (Fatigue)
    plot_dataset_comparison(
        all_results, metric='fatigue',
        save_path=os.path.join(chart_dir, "dataset_comparison_fatigue.png"),
    )

    # ── Print Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Comprehensive Comparison Summary")
    print(f"{'='*72}")

    # Makespan table
    print(f"\n{'Dataset':<12}", end="")
    methods_list = ['Greedy', 'RoundRobin', 'Random', 'DQN', 'PPO']
    for m in methods_list:
        print(f" {m:>12}", end="")
    print(f"\n{'-'*(12 + 13*len(methods_list))}")

    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        print(f"{ds_name:<12}", end="")
        res = all_results[ds_name]
        for m in methods_list:
            if m in res:
                val = res[m].get('makespan', res[m].get('makespan_mean', 0))
                print(f" {val:>12.1f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # Fatigue table
    print(f"\n{'Dataset':<12}", end="")
    for m in methods_list:
        print(f" {m:>12}", end="")
    print(f"\n{'-'*(12 + 13*len(methods_list))}")

    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        print(f"{ds_name:<12}", end="")
        res = all_results[ds_name]
        for m in methods_list:
            if m in res:
                val = res[m].get('fatigue', res[m].get('fatigue_mean', 0))
                print(f" {val:>12.3f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # ── Improvement Analysis ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Improvement vs Greedy SPT Baseline")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'Method':<12} {'MS vs Greedy':>14} {'F vs Greedy':>14}")
    print(f"{'-'*54}")

    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        greedy_ms = all_results[ds_name]['Greedy']['makespan']
        greedy_f = all_results[ds_name]['Greedy']['fatigue']
        for m in ['RoundRobin', 'DQN', 'PPO']:
            if m in all_results[ds_name]:
                ms = all_results[ds_name][m].get('makespan', all_results[ds_name][m].get('makespan_mean', 0))
                f_val = all_results[ds_name][m].get('fatigue', all_results[ds_name][m].get('fatigue_mean', 0))
                ms_diff = (ms - greedy_ms) / greedy_ms * 100
                f_diff = (f_val - greedy_f) / greedy_f * 100
                print(f"{ds_name:<12} {m:<12} {ms_diff:>+13.1f}% {f_diff:>+13.1f}%")

    # ── Save Combined Results JSON ─────────────────────────────────────────────
    combined_path = os.path.join(log_dir, "combined_results.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results saved: logs/combined_results.json")
    print(f"Charts saved: charts/")
    print(f"\n[OK] Training complete!")


if __name__ == "__main__":
    main()
