"""
Dueling Double DQN Agent with Prioritized Experience Replay
for the JSP scheduling problem.
"""

import math
import random
from typing import Tuple, List
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DQNConfig

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class SumTree:
    """
    SumTree data structure for Prioritized Experience Replay.
    Stores priorities and provides efficient weighted sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        right = left + 1
        # Avoid going into zero-valued subtrees when possible
        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx  # leaf zone: return current (should not happen normally)
        if s <= self.tree[left] or self.tree[right] == 0:
            return self._retrieve(left, min(s, self.tree[left]))
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        # Clamp s to valid range to avoid floating-point drift
        total = self.tree[0]
        if total <= 0:
            s = 0.0
        else:
            s = max(0.0, min(s, total * 0.999999))
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        # Safety: if data is None, fall back to last valid entry
        if self.data[data_idx] is None and self.n_entries > 0:
            data_idx = (self.write_idx - 1) % self.capacity
            idx = data_idx + self.capacity - 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6  # small constant for non-zero priority
        self.capacity = capacity

    def push(self, transition: Transition, error: float = None):
        """Add transition with priority. If no error given, use max priority."""
        if error is None:
            max_priority = max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
            priority = max_priority
        else:
            priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample a batch of transitions with importance-sampling weights."""
        batch = []
        indices = []
        priorities = []

        total = max(self.tree.total(), 1e-8)
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b_val = segment * (i + 1)
            s = random.uniform(a, min(b_val, total))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(max(priority, 1e-8))

        # Importance sampling weights
        sampling_prob = np.array(priorities) / total
        n = self.tree.n_entries
        # Clamp sampling_prob away from 0 to avoid division by zero
        sampling_prob = np.maximum(sampling_prob, 1e-8)
        weights = (n * sampling_prob) ** (-beta)
        weights /= max(weights.max(), 1e-8)  # normalize, avoid div by zero

        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor(np.array([t.action for t in batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(np.array([t.done for t in batch]))
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], errors: np.ndarray):
        """Update priorities of sampled transitions."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class ReplayBuffer:
    """Standard uniform replay buffer (without prioritization)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor(np.array([t.action for t in batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(np.array([t.done for t in batch]))
        return states, actions, rewards, next_states, dones, None, None

    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    """Noisy Linear layer with factorized Gaussian noise for exploration.

    Replaces epsilon-greedy with state-dependent exploration — the network
    learns when and how much to explore. Key DQN improvement from Rainbow.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialization from factorized noisy nets paper
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Resample noise for all parameters."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture for DQN.
    Separates state value V(s) and action advantage A(s,a).

    V(s) + A(s,a) - mean(A(s,a))

    Supports Noisy Networks for exploration (Rainbow DQN component).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list,
                 use_noisy: bool = False, noisy_std_init: float = 0.5):
        super().__init__()
        self.use_noisy = use_noisy

        LinearLayer = (lambda i, o: NoisyLinear(i, o, noisy_std_init)) if use_noisy else nn.Linear

        # Shared feature layers
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims[:-1]:
            layers.extend([LinearLayer(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.feature = nn.Sequential(*layers)

        # Value stream: estimates V(s)
        self.value_hidden = LinearLayer(prev_dim, hidden_dims[-1])
        self.value_out = LinearLayer(hidden_dims[-1], 1)

        # Advantage stream: estimates A(s,a) for each action
        self.adv_hidden = LinearLayer(prev_dim, hidden_dims[-1])
        self.adv_out = LinearLayer(hidden_dims[-1], action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value)
        advantage = F.relu(self.adv_hidden(features))
        advantage = self.adv_out(advantage)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Resample noise for all NoisyLinear layers (call once per episode)."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class DQNAgent:
    """
    Dueling Double DQN Agent with optional Prioritized Experience Replay.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = None):
        self.cfg = config or DQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # TF32 on Ampere+ GPUs for faster matmul (RTX 4060 supports this)
            torch.set_float32_matmul_precision('high')

        # Networks
        noisy = self.cfg.use_noisy_nets
        self.q_network = DuelingDQN(state_dim, action_dim, self.cfg.hidden_dims,
                                    use_noisy=noisy, noisy_std_init=self.cfg.noisy_std_init).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, self.cfg.hidden_dims,
                                         use_noisy=noisy, noisy_std_init=self.cfg.noisy_std_init).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.cfg.lr)

        # LR scheduler (cosine annealing, if configured)
        self.scheduler = None
        if self.cfg.lr_decay_steps > 0:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.lr_decay_steps, eta_min=self.cfg.lr * 0.01
            )

        # Replay buffer
        if self.cfg.use_per:
            self.memory = PrioritizedReplayBuffer(self.cfg.memory_capacity, self.cfg.per_alpha)
        else:
            self.memory = ReplayBuffer(self.cfg.memory_capacity)

        # Training state
        self.steps_done = 0
        self.episodes_done = 0

    def reset_noise(self):
        """Resample noise for all NoisyLinear layers (call once per episode)."""
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def select_action(self, state: np.ndarray, action_mask: np.ndarray,
                      epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy with action masking.
        Masked (invalid) actions are excluded from both exploration and exploitation.

        With Noisy Nets, epsilon is scaled down (the noise handles exploration).

        Args:
            state: current state vector
            action_mask: boolean array, True = valid
            epsilon: exploration rate (if None, use schedule)

        Returns:
            action index
        """
        if epsilon is None:
            epsilon = self._get_epsilon()

        # With Noisy Nets, scale down epsilon — noise provides exploration
        if self.cfg.use_noisy_nets:
            epsilon = epsilon * 0.3

        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # should not happen if environment is correctly implemented

        if random.random() < epsilon:
            # Random exploration among valid actions
            return int(np.random.choice(valid_actions))

        # Greedy exploitation among valid actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

        # Mask invalid actions with very negative Q-values
        masked_q = np.where(action_mask, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def _get_epsilon(self) -> float:
        """Linear epsilon decay schedule (better exploration coverage)."""
        progress = min(self.steps_done / self.cfg.epsilon_decay, 1.0)
        return self.cfg.epsilon_start + \
               (self.cfg.epsilon_end - self.cfg.epsilon_start) * progress

    def _get_per_beta(self) -> float:
        """Linearly anneal PER beta from start to 1.0."""
        progress = min(self.steps_done / self.cfg.per_beta_frames, 1.0)
        return self.cfg.per_beta_start + progress * (1.0 - self.cfg.per_beta_start)

    def store(self, state, action, reward, next_state, done, td_error=None):
        """Store transition in replay buffer."""
        self.memory.push(
            Transition(state, action, reward, next_state, done),
            error=td_error
        )

    def update(self) -> float:
        """Perform one training step. Returns the loss value."""
        if len(self.memory) < self.cfg.batch_size:
            return 0.0

        # Sample batch
        if self.cfg.use_per:
            beta = self._get_per_beta()
            states, actions, rewards, next_states, dones, weights, indices = \
                self.memory.sample(self.cfg.batch_size, beta)
        else:
            states, actions, rewards, next_states, dones, weights, indices = \
                self.memory.sample(self.cfg.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions).squeeze(1)

        # Double DQN: select action using online network, evaluate with target
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.cfg.gamma * next_q

        # Loss
        td_errors = target_q - current_q

        if self.cfg.use_per and weights is not None:
            weights = weights.to(self.device)
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.grad_clip_value)
        self.optimizer.step()

        # Step LR scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # Target network update
        if self.cfg.use_hard_update:
            # Periodic hard copy: cleaner learning signal
            if self.steps_done % self.cfg.hard_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft (Polyak) update
            for target_param, online_param in zip(
                    self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    self.cfg.target_update * online_param.data +
                    (1 - self.cfg.target_update) * target_param.data
                )

        # Update PER priorities
        if self.cfg.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.steps_done += 1
        return loss.item()

    def save(self, path: str, metadata: dict = None):
        """Save model checkpoint with optional metadata.

        Args:
            path: File path for the checkpoint.
            metadata: Optional dict with problem size, hyperparams, performance, etc.
        """
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        if metadata is not None:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint. Returns metadata dict if present, else None."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)

        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        metadata = checkpoint.get('metadata', None)
        if metadata is not None:
            print(f"  Loaded metadata: {metadata.get('problem_size', '?')} "
                  f"method={metadata.get('method', '?')} "
                  f"best_ms={metadata.get('performance', {}).get('best_makespan', '?')}")
        return metadata
