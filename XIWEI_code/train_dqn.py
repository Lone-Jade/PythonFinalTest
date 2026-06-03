"""
Train Dueling Double DQN with Prioritized Experience Replay on JSP scheduling.

GPU-accelerated training with action masking and fatigue-aware reward.

Usage:
    python train_dqn.py                                    # default: 10x10x3, 2000 ep
    python train_dqn.py --data 6x6x2 --episodes 500        # quick test
    python train_dqn.py --data 20x10x4 --episodes 5000     # larger problem
    python train_dqn.py --data 10x10x3 --no_per --lr 1e-4  # custom params
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

from config import Config, EnvConfig, DQNConfig, TrainConfig
from utils import load_csv_data, get_data_path, get_state_dim, get_action_dim, set_seed
from environment import JSPEnvironment, GreedyScheduler
from agent import DQNAgent


def train_episode(
    env: JSPEnvironment, agent: DQNAgent, epsilon: float = None
) -> Tuple[float, float, float, int]:
    """Run one training episode with N-step returns.

    Episode transitions are buffered and N-step discounted returns are
    computed before pushing to the replay buffer. This accelerates credit
    propagation by a factor of N compared to standard TD(0).

    Returns (total_reward, makespan, avg_fatigue, steps).
    """
    n_step = agent.cfg.n_step
    gamma = agent.cfg.gamma
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    # Episode trajectory buffer: list of (state, action, reward, next_state, done)
    episode_buffer = []

    while not done:
        action_mask = env._get_action_mask()
        if not np.any(action_mask):
            break

        action = agent.select_action(state, action_mask, epsilon)
        next_state, reward, done, info = env.step(action)

        episode_buffer.append((state, action, reward, next_state, done))

        # When we have N transitions buffered, compute N-step return for the oldest
        if len(episode_buffer) >= n_step:
            s0, a0, _, _, _ = episode_buffer[0]

            # Accumulate discounted rewards for steps 0..n_step-1
            G = 0.0
            episode_ended_in_window = False
            for i in range(n_step):
                _, _, ri, _, don_i = episode_buffer[i]
                G += (gamma ** i) * ri
                if don_i:
                    episode_ended_in_window = True
                    break

            # Bootstrap from step n_step if episode didn't end within window
            if not episode_ended_in_window:
                sN = episode_buffer[n_step - 1][3]  # next_state after (n_step-1)th transition
                sN_tensor = torch.FloatTensor(sN).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_N = agent.q_network(sN_tensor).max(dim=1).values.item()
                G += (gamma ** n_step) * q_N

            # Store N-step transition
            agent.store(s0, a0, G, episode_buffer[n_step - 1][3],
                       any(episode_buffer[i][4] for i in range(n_step)))
            episode_buffer.pop(0)

        agent.update()

        total_reward += reward
        state = next_state
        steps += 1

    # Drain remaining buffer at episode end
    while len(episode_buffer) > 0:
        s0, a0, _, _, _ = episode_buffer[0]

        # Sum all remaining discounted rewards (no bootstrap — episode ended)
        G = sum((gamma ** i) * episode_buffer[i][2] for i in range(len(episode_buffer)))
        # No bootstrap needed: terminal state has Q=0

        agent.store(s0, a0, G, episode_buffer[-1][3], True)
        episode_buffer.pop(0)
        agent.update()

    return total_reward, env.get_makespan(), env.get_avg_fatigue(), steps


@torch.no_grad()
def evaluate(
    env: JSPEnvironment, agent: DQNAgent, num_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate agent greedily (epsilon=0). Returns dict of metrics."""
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
        "avg_makespan": np.mean(makespans),
        "std_makespan": np.std(makespans),
        "avg_fatigue": np.mean(fatigues),
        "std_fatigue": np.std(fatigues),
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(step_counts),
    }


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


def train(config: Config, data_file: str, data_dir: str = None):
    """Main DQN training loop."""
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
    agent = DQNAgent(state_dim, action_dim, config.dqn)
    print(f"Agent: Dueling Double DQN" +
          (" + PER" if config.dqn.use_per else " (uniform replay)"))

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
    print(f"\n{'='*55}")
    print(f"Training DQN ({episodes} episodes)")
    print(f"{'='*55}")

    best_makespan = float("inf")
    best_fatigue = float("inf")
    ep_rewards = []
    ep_makespans = []
    no_improve = 0
    start_time = time.time()

    for ep in range(1, episodes + 1):
        total_r, ms, fat, steps = train_episode(env, agent)

        ep_rewards.append(total_r)
        ep_makespans.append(ms)

        improved = False
        if ms < best_makespan:
            best_makespan = ms
            improved = True
        if fat < best_fatigue:
            best_fatigue = fat
            improved = True

        # Only count no-improvement during exploitation phase (epsilon < 0.3)
        # During exploration, improvements come from random search and are rare.
        if agent._get_epsilon() < 0.3:
            no_improve = 0 if improved else no_improve + 1
        else:
            no_improve = 0  # reset counter during exploration

        # Logging
        if ep % config.train.log_interval == 0 or ep == 1:
            elapsed = time.time() - start_time
            avg_r100 = np.mean(ep_rewards[-100:])
            avg_ms100 = np.mean(ep_makespans[-100:])
            eps = agent._get_epsilon()
            print(
                f"Ep {ep:>5d}/{episodes} | "
                f"eps={eps:.3f} | "
                f"R100={avg_r100:>8.2f} | "
                f"MS100={avg_ms100:>8.1f} | "
                f"Best={best_makespan:>8.1f} | "
                f"{elapsed:.0f}s"
            )

        # Evaluation
        if ep % config.train.eval_interval == 0:
            ev = evaluate(env, agent, num_episodes=5)
            print(
                f"  Eval: MS={ev['avg_makespan']:.1f}±{ev['std_makespan']:.1f}  "
                f"Fatigue={ev['avg_fatigue']:.3f}±{ev['std_fatigue']:.3f}  "
                f"R={ev['avg_reward']:.2f}"
            )

        # Checkpoint
        if ep % config.train.save_interval == 0:
            ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = (f"dqn_{N}x{M}x{W}_nstep{config.dqn.n_step}"
                         f"_lr{config.dqn.lr:.0e}_g{config.dqn.gamma}_ep{ep}.pt")
            metadata = _build_metadata(config, data_file, N, M, W,
                                       best_makespan, best_fatigue,
                                       ev if ep % config.train.eval_interval == 0 else None,
                                       baselines, agent)
            agent.save(os.path.join(ckpt_dir, ckpt_name), metadata)
            print(f"  Saved checkpoint at episode {ep}")

        # Early stopping
        if no_improve >= config.train.early_stop_patience:
            print(f"\nEarly stopping at episode {ep} "
                  f"(no improvement for {no_improve} episodes)")
            break

        agent.episodes_done += 1

    total_time = time.time() - start_time
    print(f"\nTraining complete ({total_time:.1f}s)")
    print(f"  Best Makespan: {best_makespan:.1f}")
    print(f"  Best Fatigue:  {best_fatigue:.3f}")

    # Final evaluation
    print(f"\n{'='*55}")
    print("Final Evaluation (10 episodes, epsilon=0)")
    print(f"{'='*55}")
    ev = evaluate(env, agent, num_episodes=10)
    print(f"  DQN Agent:  Makespan={ev['avg_makespan']:.1f}±{ev['std_makespan']:.1f}  "
          f"Fatigue={ev['avg_fatigue']:.3f}±{ev['std_fatigue']:.3f}")

    # Save final model
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_name = (f"dqn_{N}x{M}x{W}_nstep{config.dqn.n_step}"
                       f"_lr{config.dqn.lr:.0e}_g{config.dqn.gamma}_final.pt")
    final_metadata = _build_metadata(config, data_file, N, M, W,
                                     best_makespan, best_fatigue, ev, baselines, agent)
    agent.save(os.path.join(ckpt_dir, final_ckpt_name), final_metadata)
    print(f"  Final model saved: {final_ckpt_name}")

    # Save training log JSON for visualization
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"train_{N}x{M}x{W}_nstep{config.dqn.n_step}.json"
    training_log = {
        'problem': f"{N}x{M}x{W}",
        'config': {
            'n_step': config.dqn.n_step,
            'gamma': config.dqn.gamma,
            'lr': config.dqn.lr,
            'lambda_fatigue': config.env.lambda_fatigue,
            'use_per': config.dqn.use_per,
            'epsilon_decay': config.dqn.epsilon_decay,
            'seed': config.train.seed,
        },
        'episodes': [int(e) for e in range(1, len(ep_rewards) + 1)],
        'rewards': [float(r) for r in ep_rewards],
        'makespans': [float(m) for m in ep_makespans],
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
    print(f"  {'DQN Agent':<18} {ev['avg_makespan']:>10.1f} {ev['avg_fatigue']:>10.3f}")
    print(f"  {'Greedy SPT':<18} {baselines['greedy_makespan']:>10.1f} "
          f"{baselines['greedy_fatigue']:>10.3f}")
    print(f"  {'Random':<18} {baselines['random_makespan']:>10.1f} "
          f"{baselines['random_fatigue']:>10.3f}")
    print(f"  {'Round-Robin':<18} {baselines['roundrobin_makespan']:>10.1f} "
          f"{baselines['roundrobin_fatigue']:>10.3f}")

    return agent, ev, baselines


def _build_metadata(config: Config, data_file: str, N: int, M: int, W: int,
                    best_makespan: float, best_fatigue: float,
                    eval_result: Dict, baselines: Dict, agent: DQNAgent) -> Dict:
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
            'eval_fatigue_std': eval_result['std_fatigue'],
        })

    return {
        'problem_size': {'N': N, 'M': M, 'W': W},
        'data_file': data_file,
        'method': f'Dueling Double DQN + PER + {config.dqn.n_step}-step',
        'state_dim': 3 * M + 3 * W + 3 * N + 1,
        'action_dim': N * W,
        'hyperparams': {
            'lr': config.dqn.lr,
            'gamma': config.dqn.gamma,
            'n_step': config.dqn.n_step,
            'batch_size': config.dqn.batch_size,
            'hidden_dims': config.dqn.hidden_dims,
            'memory_capacity': config.dqn.memory_capacity,
            'target_update': config.dqn.target_update,
            'use_per': config.dqn.use_per,
            'per_alpha': config.dqn.per_alpha,
            'epsilon_decay': config.dqn.epsilon_decay,
        },
        'env_params': {
            'alpha': config.env.alpha,
            'beta': config.env.beta,
            'gamma_fatigue': config.env.gamma,
            'F_threshold': config.env.F_threshold,
            'F_max': config.env.F_max,
            'lambda_fatigue': config.env.lambda_fatigue,
            'use_terminal_ms_reward': config.env.use_terminal_ms_reward,
        },
        'performance': perf,
        'baselines': {
            'greedy_makespan': baselines['greedy_makespan'],
            'greedy_fatigue': baselines['greedy_fatigue'],
            'random_makespan': baselines['random_makespan'],
            'random_fatigue': baselines['random_fatigue'],
            'roundrobin_makespan': baselines['roundrobin_makespan'],
            'roundrobin_fatigue': baselines['roundrobin_fatigue'],
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': config.train.seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Dueling Double DQN for JSP scheduling with worker fatigue"
    )
    parser.add_argument("--data", type=str, default="10x10x3",
                        help="Dataset name without .csv (default: 10x10x3)")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--lambda_fatigue", type=float, default=2.0,
                        help="Fatigue penalty weight")
    parser.add_argument("--no_per", action="store_true",
                        help="Disable Prioritized Experience Replay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval (episodes)")
    parser.add_argument("--early_stop", type=int, default=200,
                        help="Early stopping patience")
    args = parser.parse_args()

    config = Config()
    config.train.num_episodes = args.episodes
    config.train.data_file = args.data
    config.train.seed = args.seed
    config.train.eval_interval = args.eval_interval
    config.train.early_stop_patience = args.early_stop
    config.dqn.lr = args.lr
    config.env.lambda_fatigue = args.lambda_fatigue
    config.dqn.use_per = not args.no_per

    train(config, args.data)


if __name__ == "__main__":
    main()
