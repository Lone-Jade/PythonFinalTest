"""
Comprehensive evaluation and comparison script for JSP scheduling.

Tests all methods (Greedy, Random, Round-Robin, DQN) on multiple problem
instances and compares performance.

Usage:
    python test.py                                    # test all available datasets
    python test.py --data 10x10x3                     # specific dataset
    python test.py --data 6x6x2 --episodes 500        # quick test
    python test.py --compare_weights                  # compare fatigue weights
    python test.py --model checkpoints/dqn_10x10x3_final.pt --data 10x10x3
"""

import os
import sys
import time
import argparse
from typing import Dict, List

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, EnvConfig, DQNConfig
from utils import (
    load_csv_data,
    find_csv_files,
    get_data_path,
    get_state_dim,
    get_action_dim,
    set_seed,
)
from environment import JSPEnvironment, GreedyScheduler
from agent import DQNAgent
from agent_ppo import PPOAgent
from train_baselines import run_greedy, run_random, run_roundrobin


def _detect_model_type(checkpoint_path: str) -> str:
    """Peek at checkpoint to determine if it's DQN or PPO."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ac_network' in ckpt:
        return 'ppo'
    elif 'q_network' in ckpt:
        return 'dqn'
    else:
        raise KeyError(f"Unknown checkpoint format: {list(ckpt.keys())}")

# ═══════════════════════════════════════════════════════════════════════
# DQN Evaluation
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_agent(
    env: JSPEnvironment, agent, num_runs: int = 10
) -> Dict[str, float]:
    """Evaluate a trained agent greedily. Compatible with DQN and PPO."""
    makespans, fatigues, rewards, steps_list = [], [], [], []
    is_ppo = isinstance(agent, PPOAgent)

    for _ in range(num_runs):
        env.reset()
        done = False
        ep_reward, steps = 0.0, 0

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
            steps += 1

        makespans.append(env.get_makespan())
        fatigues.append(env.get_avg_fatigue())
        rewards.append(ep_reward)
        steps_list.append(steps)

    return {
        "makespan_mean": np.mean(makespans),
        "makespan_std": np.std(makespans),
        "fatigue_mean": np.mean(fatigues),
        "fatigue_std": np.std(fatigues),
        "reward_mean": np.mean(rewards),
        "steps_mean": np.mean(steps_list),
    }


def train_dqn_quick(
    data: Dict,
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    num_episodes: int = 500,
) -> DQNAgent:
    """Quick-train a DQN agent for testing."""
    env = JSPEnvironment(data, env_config)
    state_dim = get_state_dim(data)
    action_dim = get_action_dim(data)
    agent = DQNAgent(state_dim, action_dim, dqn_config)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break
            action = agent.select_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        agent.episodes_done += 1

        if ep % 200 == 0:
            print(
                f"    Ep {ep}: eps={agent._get_epsilon():.3f}  "
                f"ms={env.get_makespan():.1f}"
            )

    return agent


# ═══════════════════════════════════════════════════════════════════════
# Fatigue Weight Comparison
# ═══════════════════════════════════════════════════════════════════════


def compare_fatigue_weights(
    data: Dict,
    base_env_config: EnvConfig,
    dqn_config: DQNConfig,
    fatigue_weights: List[float],
    num_episodes: int = 500,
) -> Dict[float, Dict]:
    """Compare DQN performance under different fatigue penalty weights."""
    results = {}

    for lam in fatigue_weights:
        print(f"\n  lambda_fatigue = {lam:.1f} ...")
        cfg = EnvConfig(
            alpha=base_env_config.alpha,
            beta=base_env_config.beta,
            gamma=base_env_config.gamma,
            F_threshold=base_env_config.F_threshold,
            lambda_fatigue=lam,
            eta=base_env_config.eta,
        )

        env = JSPEnvironment(data, cfg)
        state_dim = get_state_dim(data)
        action_dim = get_action_dim(data)
        agent = DQNAgent(state_dim, action_dim, dqn_config)

        for ep in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            while not done:
                mask = env._get_action_mask()
                if not np.any(mask):
                    break
                action = agent.select_action(state, mask)
                next_state, reward, done, _ = env.step(action)
                agent.store(state, action, reward, next_state, done)
                agent.update()
                state = next_state
            agent.episodes_done += 1

        results[lam] = evaluate_agent(env, agent, num_runs=10)
        print(
            f"    MS={results[lam]['makespan_mean']:.1f}±"
            f"{results[lam]['makespan_std']:.1f}  "
            f"F={results[lam]['fatigue_mean']:.3f}±"
            f"{results[lam]['fatigue_std']:.3f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="JSP Scheduling — Comprehensive Evaluation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Specific dataset (default: test representative set)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to pre-trained model checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Training episodes for DQN (default: 500)",
    )
    parser.add_argument(
        "--compare_weights",
        action="store_true",
        help="Compare different fatigue penalty weights",
    )
    parser.add_argument(
        "--no_per", action="store_true", help="Disable Prioritized Experience Replay"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = Config()
    config.dqn.use_per = not args.no_per

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    # Determine datasets to test
    if args.data:
        data_files = [args.data]
    else:
        all_files = find_csv_files(data_dir)
        data_files = [os.path.basename(f).replace(".csv", "") for f in all_files]
        # Limit to reasonable set for comprehensive test
        if len(data_files) > 5:
            data_files = data_files[:5]

    print("=" * 72)
    print("JSP Scheduling — Comprehensive Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: N/A (CPU)")
    print(f"Datasets: {', '.join(data_files)}")
    print("=" * 72)

    all_results = {}

    for data_file in data_files:
        print(f"\n{'='*72}")
        print(f"Testing: {data_file}")
        print(f"{'='*72}")

        data_path = get_data_path(data_dir, data_file)
        data = load_csv_data(data_path)
        _ = data["num_jobs"], data["num_machines"], data["num_workers"]

        t0 = time.time()

        # --- Baselines (from train_baselines module) ---
        print("\n[1] Baselines")
        greedy_res = run_greedy(data, config.env)
        print(
            f"  Greedy SPT:    MS={greedy_res['makespan']:.1f}  "
            f"Fatigue={greedy_res['fatigue']:.3f}"
        )

        env = JSPEnvironment(data, config.env)
        random_res = run_random(data, config.env, num_runs=30)
        print(
            f"  Random (30):   MS={random_res['makespan_mean']:.1f}±"
            f"{random_res['makespan_std']:.0f}  "
            f"Fatigue={random_res['fatigue_mean']:.3f}"
        )

        rr_res = run_roundrobin(data, config.env)
        print(
            f"  Round-Robin:   MS={rr_res['makespan']:.1f}  "
            f"Fatigue={rr_res['fatigue']:.3f}"
        )

        # --- Model Loading / Training ---
        model_type = 'dqn'  # default
        if args.model:
            model_type = _detect_model_type(args.model)
            print(f"\n[2] Loading pre-trained {model_type.upper()} model: {args.model}")
            env = JSPEnvironment(data, config.env)
            state_dim = get_state_dim(data)
            action_dim = get_action_dim(data)

            if model_type == 'ppo':
                agent = PPOAgent(state_dim, action_dim, config.ppo)
            else:
                agent = DQNAgent(state_dim, action_dim, config.dqn)

            metadata = agent.load(args.model)
            if metadata:
                print(f"  Problem: {metadata.get('problem_size', '?')}")
                print(f"  Method:  {metadata.get('method', '?')}")
                perf = metadata.get('performance', {})
                print(f"  Best MS: {perf.get('best_makespan', '?')}, "
                      f"Best Fatigue: {perf.get('best_fatigue', '?')}")
                hp = metadata.get('hyperparams', {})
                if model_type == 'ppo':
                    print(f"  HP: lr={hp.get('lr','?')}, γ={hp.get('gamma','?')}, "
                          f"λ={hp.get('gae_lambda','?')}, ent={hp.get('entropy_coef','?')}")
                else:
                    print(f"  HP: N-step={hp.get('n_step','?')}, γ={hp.get('gamma','?')}, "
                          f"lr={hp.get('lr','?')}")
        else:
            print(f"\n[2] Training DQN ({args.episodes} episodes)")
            env = JSPEnvironment(data, config.env)
            agent = train_dqn_quick(data, config.env, config.dqn, args.episodes)

        model_res = evaluate_agent(env, agent, num_runs=10)
        model_label = f"{model_type.upper()}+Fatigue" if model_type == 'dqn' else f"PPO Agent"
        print(
            f"  {model_label:<14}:  MS={model_res['makespan_mean']:.1f}±"
            f"{model_res['makespan_std']:.1f}  "
            f"Fatigue={model_res['fatigue_mean']:.3f}±"
            f"{model_res['fatigue_std']:.3f}"
        )

        # --- DQN without fatigue (skip when loading pre-trained model) ---
        if not args.model:
            print(f"\n[3] Training DQN (no fatigue, {args.episodes} episodes)")
            nofat_cfg = EnvConfig(
                alpha=config.env.alpha,
                beta=config.env.beta,
                gamma=config.env.gamma,
                F_threshold=config.env.F_threshold,
                lambda_fatigue=0.0,
                eta=config.env.eta,
            )
            nofat_agent = train_dqn_quick(data, nofat_cfg, config.dqn, args.episodes)
            nofat_env = JSPEnvironment(data, nofat_cfg)
            nofat_res = evaluate_agent(nofat_env, nofat_agent, num_runs=10)
            print(
                f"  DQN(no fat):   MS={nofat_res['makespan_mean']:.1f}±"
                f"{nofat_res['makespan_std']:.1f}  "
                f"Fatigue={nofat_res['fatigue_mean']:.3f}±"
                f"{nofat_res['fatigue_std']:.3f}"
            )
        else:
            nofat_res = None

        # --- Fatigue weight comparison ---
        weight_results = None
        if args.compare_weights:
            print(f"\n[4] Comparing fatigue weights")
            weights = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
            weight_results = compare_fatigue_weights(
                data,
                config.env,
                config.dqn,
                weights,
                num_episodes=max(args.episodes // 2, 300),
            )
            print(f"\n  {'lambda':>8} {'Makespan':>10} {'Fatigue':>10}")
            print(f"  {'-'*30}")
            for lam in weights:
                r = weight_results[lam]
                print(
                    f"  {lam:>8.1f} {r['makespan_mean']:>10.1f} "
                    f"{r['fatigue_mean']:>10.3f}"
                )

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed:.1f}s")

        all_results[data_file] = {
            "greedy": greedy_res,
            "random": random_res,
            "roundrobin": rr_res,
            "model": model_res,
            "weights": weight_results,
        }
        # Only include DQN no-fatigue if we trained DQN ourselves
        if not args.model:
            all_results[data_file]["dqn_nofatigue"] = nofat_res

    # ═══════════════════════════════════════════════════════════════
    # Overall Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("Overall Summary")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'Method':<16} {'Makespan':>10} {'Fatigue':>10}")
    print("-" * 50)
    for data_file, res in all_results.items():
        name = data_file[:12]
        entries = [
            ("Greedy", res["greedy"]["makespan"], res["greedy"]["fatigue"]),
            (
                "Round-Robin",
                res["roundrobin"]["makespan"],
                res["roundrobin"]["fatigue"],
            ),
            ("Random", res["random"]["makespan_mean"], res["random"]["fatigue_mean"]),
        ]
        # Model result (DQN or PPO)
        if "model" in res:
            entries.append(
                ("Model (loaded)", res["model"]["makespan_mean"],
                 res["model"]["fatigue_mean"]),
            )
        if "dqn_nofatigue" in res:
            entries.append(
                ("DQN(no fat)", res["dqn_nofatigue"]["makespan_mean"],
                 res["dqn_nofatigue"]["fatigue_mean"]),
            )
        for method, ms, fat in entries:
            print(f"{name:<12} {method:<16} {ms:>10.1f} {fat:>10.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
