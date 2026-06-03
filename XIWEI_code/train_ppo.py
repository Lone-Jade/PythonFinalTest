"""
Train PPO (Proximal Policy Optimization) for JSP scheduling.

PPO directly optimizes the policy using GAE advantage estimates,
avoiding the credit assignment problems that plague DQN on this task.

Usage:
    python train_ppo.py                                    # default: 10x10x3, 2000 ep
    python train_ppo.py --data 6x6x2 --episodes 500        # quick test
    python train_ppo.py --data 20x10x4 --episodes 5000     # larger problem
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import torch

# Suppress OpenMP duplicate lib warning on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, EnvConfig, PPOConfig, TrainConfig
from utils import load_csv_data, get_data_path, get_state_dim, get_action_dim, set_seed
from environment import JSPEnvironment, GreedyScheduler
from agent_ppo import PPOAgent


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    env: JSPEnvironment, agent: PPOAgent, num_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate agent greedily. Returns dict of metrics."""
    makespans, fatigues, rewards = [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            action, _ = agent.evaluate(state, mask)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

        makespans.append(env.get_makespan())
        fatigues.append(env.get_avg_fatigue())
        rewards.append(ep_reward)

    return {
        "avg_makespan": np.mean(makespans),
        "std_makespan": np.std(makespans),
        "avg_fatigue": np.mean(fatigues),
        "std_fatigue": np.std(fatigues),
        "avg_reward": np.mean(rewards),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════

def run_baselines(data: Dict, env_config: EnvConfig) -> Dict:
    """Run baseline heuristics for comparison."""
    results = {}

    # Greedy SPT
    greedy = GreedyScheduler(data, env_config)
    ms, fat, _ = greedy.solve()
    results["greedy_makespan"] = ms
    results["greedy_fatigue"] = fat

    # Random (averaged)
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
    results["random_makespan"] = np.mean(ms_list)
    results["random_fatigue"] = np.mean(fat_list)

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


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(config: Config, data_file: str, data_dir: str = None):
    """Main PPO training loop."""
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )

    set_seed(config.train.seed)

    # Load data
    data_path = get_data_path(data_dir, data_file)
    print(f"Loading data: {data_path}")
    data = load_csv_data(data_path)
    N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
    print(f"Problem: {N} jobs x {M} machines x {W} workers")

    state_dim = get_state_dim(data)
    action_dim = get_action_dim(data)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")
    else:
        print("GPU: N/A (using CPU)")

    # Create environment and agent
    env = JSPEnvironment(data, config.env)
    agent = PPOAgent(state_dim, action_dim, config.ppo)
    print(f"Agent: PPO (Actor-Critic + GAE)")
    print(f"  Rollout: {config.ppo.rollout_episodes} episodes, "
          f"PPO epochs: {config.ppo.ppo_epochs}, "
          f"γ={config.ppo.gamma}, λ={config.ppo.gae_lambda}")

    # Baselines
    print(f"\n{'='*55}")
    print("Baselines")
    print(f"{'='*55}")
    baselines = run_baselines(data, config.env)
    print(f"  Greedy SPT:     Makespan={baselines['greedy_makespan']:>10.1f}  "
          f"Fatigue={baselines['greedy_fatigue']:.3f}")
    print(f"  Random (20run): Makespan={baselines['random_makespan']:>10.1f}  "
          f"Fatigue={baselines['random_fatigue']:.3f}")
    print(f"  Round-Robin:    Makespan={baselines['roundrobin_makespan']:>10.1f}  "
          f"Fatigue={baselines['roundrobin_fatigue']:.3f}")

    # Training
    episodes = config.train.num_episodes
    rollout_size = config.ppo.rollout_episodes
    print(f"\n{'='*55}")
    print(f"Training PPO ({episodes} episodes, update every {rollout_size})")
    print(f"{'='*55}")

    best_makespan = float("inf")
    best_fatigue = float("inf")
    all_makespans = []
    all_rewards = []
    no_improve = 0
    start_time = time.time()

    ep = 0
    while ep < episodes:
        # --- Collect rollout ---
        trajectories = []
        rollout_makespans = []
        rollout_rewards = []

        for _ in range(rollout_size):
            if ep >= episodes:
                break
            ep += 1
            agent.episodes_done += 1

            traj = agent.collect_episode(env)
            trajectories.append(traj)
            rollout_makespans.append(traj['makespan'])
            rollout_rewards.append(traj['total_reward'])

            all_makespans.append(traj['makespan'])
            all_rewards.append(traj['total_reward'])

            # Track best
            improved = False
            if traj['makespan'] < best_makespan:
                best_makespan = traj['makespan']
                improved = True
            if traj['fatigue'] < best_fatigue:
                best_fatigue = traj['fatigue']
                improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        # --- PPO Update ---
        metrics = agent.update(trajectories)

        # --- Logging ---
        if ep % config.train.log_interval == 0 or ep <= rollout_size:
            elapsed = time.time() - start_time
            avg_ms = np.mean(all_makespans[-100:]) if all_makespans else 0
            avg_r = np.mean(all_rewards[-100:]) if all_rewards else 0
            print(
                f"Ep {ep:>5d}/{episodes} | "
                f"MS100={avg_ms:>8.1f} | "
                f"R100={avg_r:>8.2f} | "
                f"Best={best_makespan:>8.1f} | "
                f"P.Loss={metrics['policy_loss']:.4f} | "
                f"{elapsed:.0f}s"
            )

        # --- Evaluation ---
        if ep % config.train.eval_interval == 0:
            ev = evaluate(env, agent, num_episodes=5)
            print(
                f"  Eval: MS={ev['avg_makespan']:.1f}±{ev['std_makespan']:.1f}  "
                f"Fatigue={ev['avg_fatigue']:.3f}±{ev['std_fatigue']:.3f}  "
                f"R={ev['avg_reward']:.2f}"
            )

        # --- Checkpoint ---
        if ep % config.train.save_interval == 0:
            ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = (f"ppo_{N}x{M}x{W}_lr{config.ppo.lr:.0e}"
                         f"_g{config.ppo.gamma}_ep{ep}.pt")
            metadata = _build_metadata(config, data_file, N, M, W,
                                       best_makespan, best_fatigue,
                                       evaluate(env, agent, num_episodes=5),
                                       baselines, agent)
            agent.save(os.path.join(ckpt_dir, ckpt_name), metadata)
            print(f"  Saved checkpoint at episode {ep}")

        # --- Early stopping ---
        if no_improve >= config.train.early_stop_patience:
            print(f"\nEarly stopping at episode {ep} "
                  f"(no improvement for {no_improve} episodes)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete ({total_time:.1f}s)")
    print(f"  Best Makespan: {best_makespan:.1f}")
    print(f"  Best Fatigue:  {best_fatigue:.3f}")

    # Final evaluation
    print(f"\n{'='*55}")
    print("Final Evaluation (10 episodes, greedy)")
    print(f"{'='*55}")
    ev = evaluate(env, agent, num_episodes=10)
    print(f"  PPO Agent:  Makespan={ev['avg_makespan']:.1f}±{ev['std_makespan']:.1f}  "
          f"Fatigue={ev['avg_fatigue']:.3f}±{ev['std_fatigue']:.3f}")

    # Save final model
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_name = (f"ppo_{N}x{M}x{W}_lr{config.ppo.lr:.0e}"
                  f"_g{config.ppo.gamma}_final.pt")
    final_metadata = _build_metadata(config, data_file, N, M, W,
                                     best_makespan, best_fatigue, ev, baselines, agent)
    agent.save(os.path.join(ckpt_dir, final_name), final_metadata)
    print(f"  Final model saved: {final_name}")

    # Save training log
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"train_ppo_{N}x{M}x{W}.json"
    training_log = {
        'problem': f"{N}x{M}x{W}",
        'method': 'PPO',
        'config': {
            'lr': config.ppo.lr,
            'gamma': config.ppo.gamma,
            'gae_lambda': config.ppo.gae_lambda,
            'clip_epsilon': config.ppo.clip_epsilon,
            'entropy_coef': config.ppo.entropy_coef,
            'rollout_episodes': config.ppo.rollout_episodes,
            'ppo_epochs': config.ppo.ppo_epochs,
            'seed': config.train.seed,
        },
        'episodes': [int(e) for e in range(1, len(all_makespans) + 1)],
        'rewards': [float(r) for r in all_rewards],
        'makespans': [float(m) for m in all_makespans],
        'best_makespan': float(best_makespan),
        'best_fatigue': float(best_fatigue),
        'baselines': {k: float(v) for k, v in baselines.items()},
        'final_eval': {k: float(v) for k, v in ev.items()},
    }
    with open(os.path.join(log_dir, log_name), 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved: logs/{log_name}")

    # Comparison
    print(f"\n{'='*55}")
    print("Comparison")
    print(f"{'='*55}")
    print(f"  {'Method':<18} {'Makespan':>10} {'Fatigue':>10}")
    print(f"  {'-'*40}")
    print(f"  {'PPO Agent':<18} {ev['avg_makespan']:>10.1f} {ev['avg_fatigue']:>10.3f}")
    print(f"  {'Greedy SPT':<18} {baselines['greedy_makespan']:>10.1f} "
          f"{baselines['greedy_fatigue']:>10.3f}")
    print(f"  {'Random':<18} {baselines['random_makespan']:>10.1f} "
          f"{baselines['random_fatigue']:>10.3f}")
    print(f"  {'Round-Robin':<18} {baselines['roundrobin_makespan']:>10.1f} "
          f"{baselines['roundrobin_fatigue']:>10.3f}")

    return agent, ev, baselines


# ═══════════════════════════════════════════════════════════════════════════
# Metadata Helper
# ═══════════════════════════════════════════════════════════════════════════

def _build_metadata(config: Config, data_file: str, N: int, M: int, W: int,
                    best_makespan: float, best_fatigue: float,
                    eval_result: Dict, baselines: Dict, agent: PPOAgent) -> Dict:
    """Build metadata dict for model checkpoint."""
    perf = {
        'best_makespan': best_makespan,
        'best_fatigue': best_fatigue,
        'num_episodes_trained': agent.episodes_done,
    }
    if eval_result is not None:
        perf.update({
            'eval_makespan_mean': eval_result['avg_makespan'],
            'eval_makespan_std': eval_result['std_makespan'],
            'eval_fatigue_mean': eval_result['avg_fatigue'],
        })

    return {
        'problem_size': {'N': N, 'M': M, 'W': W},
        'data_file': data_file,
        'method': 'PPO (Actor-Critic + GAE)',
        'state_dim': 3 * M + 3 * W + 3 * N + 1,
        'action_dim': N * W,
        'hyperparams': {
            'lr': config.ppo.lr,
            'gamma': config.ppo.gamma,
            'gae_lambda': config.ppo.gae_lambda,
            'clip_epsilon': config.ppo.clip_epsilon,
            'entropy_coef': config.ppo.entropy_coef,
            'value_coef': config.ppo.value_coef,
            'rollout_episodes': config.ppo.rollout_episodes,
            'ppo_epochs': config.ppo.ppo_epochs,
            'hidden_dims': config.ppo.hidden_dims,
        },
        'env_params': {
            'alpha': config.env.alpha,
            'beta': config.env.beta,
            'gamma_fatigue': config.env.gamma,
            'lambda_fatigue': config.env.lambda_fatigue,
            'use_terminal_ms_reward': config.env.use_terminal_ms_reward,
        },
        'performance': perf,
        'baselines': {
            'greedy_makespan': float(baselines['greedy_makespan']),
            'greedy_fatigue': float(baselines['greedy_fatigue']),
            'random_makespan': float(baselines['random_makespan']),
            'random_fatigue': float(baselines['random_fatigue']),
            'roundrobin_makespan': float(baselines['roundrobin_makespan']),
            'roundrobin_fatigue': float(baselines['roundrobin_fatigue']),
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': config.train.seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO for JSP scheduling with worker fatigue"
    )
    parser.add_argument("--data", type=str, default="10x10x3",
                        help="Dataset name (default: 10x10x3)")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for GAE")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO clipping range")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--rollout", type=int, default=10,
                        help="Episodes per PPO update")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--early_stop", type=int, default=200,
                        help="Early stopping patience")
    args = parser.parse_args()

    config = Config()
    config.train.num_episodes = args.episodes
    config.train.seed = args.seed
    config.train.early_stop_patience = args.early_stop
    config.ppo.lr = args.lr
    config.ppo.gamma = args.gamma
    config.ppo.gae_lambda = args.gae_lambda
    config.ppo.clip_epsilon = args.clip_epsilon
    config.ppo.entropy_coef = args.entropy_coef
    config.ppo.rollout_episodes = args.rollout

    train(config, args.data)


if __name__ == "__main__":
    main()
