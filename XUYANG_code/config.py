from dataclasses import dataclass


@dataclass
class EnvConfig:
    # Fatigue dynamics.
    alpha: float = 0.020
    gamma_rest: float = 0.045
    f_force: float = 0.80
    f_resume: float = 0.50
    active_rest_duration: int = 20

    # Processing-time multiplier: 1 + beta * sigmoid(k * (F - theta)).
    beta: float = 0.60
    sigmoid_k: float = 12.0
    theta: float = 0.60

    # Nonlinear reward scales.
    s_time: float = 1.0
    s_avg_fatigue: float = 0.45
    s_max_fatigue: float = 0.80
    s_machine_idle: float = 0.30
    s_worker_idle: float = 0.10
    s_force_rest: float = 1.20
    s_invalid: float = 0.50
    s_makespan: float = 2.0

    k_avg_fatigue: float = 3.0
    k_max_fatigue: float = 4.0
    k_machine_idle: float = 3.0
    k_worker_idle: float = 4.0
    k_makespan: float = 2.0

    # If no heuristic reference is provided, this keeps rewards finite.
    min_t_ref: float = 1.0


@dataclass
class TrainConfig:
    episodes: int = 200
    gamma: float = 0.99
    lr: float = 3e-4
    lr_decay: float = 0.995     # multiplicative per-episode LR decay (0.995^100 ≈ 0.61)
    hidden_dim: int = 128
    seed: int = 42
    max_decisions: int = 100000

    # DQN.
    batch_size: int = 64
    replay_size: int = 100000    # larger buffer for diverse experience
    target_update: int = 500     # less frequent target updates for stability
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.96  # per-episode decay (0.96^100 ≈ 0.017)

    # PPO.
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
