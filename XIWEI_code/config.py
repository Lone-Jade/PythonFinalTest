"""
Configuration for the RL-based JSP scheduling system.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    """Environment configuration (fatigue model parameters)."""
    alpha: float = 0.02            # fatigue accumulation rate per time unit
    beta: float = 0.025            # fatigue recovery rate per time unit (> alpha for stability)
    gamma: float = 0.05            # fatigue impact coefficient on processing time
    F_threshold: float = 0.5       # fatigue threshold below which no penalty
    F_max: float = 10.0            # fatigue cap to prevent runaway feedback loop
    lambda_fatigue: float = 2.0    # terminal fatigue penalty weight
    eta: float = 0.0               # step reward fatigue coefficient (0 = disabled)
    use_terminal_ms_reward: bool = True  # add terminal -makespan/scale reward


@dataclass
class DQNConfig:
    """DQN agent configuration."""
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Training
    lr: float = 3e-4
    gamma: float = 0.95            # lower discount for finite-horizon JSP (was 0.99)
    n_step: int = 3                # N-step TD returns (1 = standard TD(0))
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: int = 100000    # linear decay steps (keeps exploration high longer)
    batch_size: int = 64
    memory_capacity: int = 50000   # larger replay buffer for diverse experiences
    target_update: float = 0.005   # soft update rate (τ for polyak averaging)
    use_per: bool = True           # prioritized experience replay
    per_alpha: float = 0.6         # PER alpha (priority exponent)
    per_beta_start: float = 0.4    # PER beta initial (importance sampling)
    per_beta_frames: int = 100000  # PER beta annealing frames


@dataclass
class PPOConfig:
    """PPO (Proximal Policy Optimization) agent configuration."""
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # Training
    lr: float = 3e-4                 # learning rate (actor + critic share optimizer)
    gamma: float = 0.99              # discount factor (used in GAE + returns)
    gae_lambda: float = 0.95         # GAE lambda (1.0 = Monte Carlo, 0.0 = TD(0))
    clip_epsilon: float = 0.2        # PPO clipping range
    value_coef: float = 0.5          # value loss coefficient
    entropy_coef: float = 0.01       # entropy bonus coefficient
    max_grad_norm: float = 0.5       # gradient clipping

    # Data collection
    rollout_episodes: int = 10       # episodes to collect before each PPO update
    ppo_epochs: int = 8              # number of epochs per PPO update
    mini_batch_size: int = 128       # mini-batch size for PPO updates


@dataclass
class TrainConfig:
    """Training configuration."""
    num_episodes: int = 2000
    log_interval: int = 50
    eval_interval: int = 20       # less frequent eval (speed)
    save_interval: int = 500
    early_stop_patience: int = 500  # longer patience with slow exploration
    seed: int = 42
    data_file: str = "10x10x3"


@dataclass
class Config:
    """Master configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
