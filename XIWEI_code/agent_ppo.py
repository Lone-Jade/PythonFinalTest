"""
PPO (Proximal Policy Optimization) Agent with Action Masking for JSP scheduling.

Uses Actor-Critic architecture with GAE (Generalized Advantage Estimation).
Action masking ensures only valid (job, worker) pairs are selected.

Key advantages over DQN for this problem:
- Direct policy optimization (no Q-function bootstrapping error)
- Full episode returns eliminate credit propagation delay
- GAE provides low-variance advantage estimates
- Naturally handles the discrete combinatorial action space
"""

from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import PPOConfig


# ═══════════════════════════════════════════════════════════════════════════
# Actor-Critic Network
# ═══════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic network with action masking.

    Actor outputs: logits for each (job × worker) action.
    Critic outputs: scalar state value V(s).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.feature = nn.Sequential(*layers)

        # Actor head: action logits
        self.actor = nn.Linear(prev_dim, action_dim)

        # Critic head: state value
        self.critic = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small weights for actor output to start near uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (logits, value)."""
        features = self.feature(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Sample action with action masking.

        Args:
            state: (1, state_dim) tensor.
            action_mask: (action_dim,) boolean tensor, True = valid.
            deterministic: If True, return argmax (for evaluation).

        Returns:
            (action_idx, log_prob, value) — all Python scalars.
        """
        logits, value = self.forward(state)  # logits: (1, action_dim), value: (1, 1)

        # Action masking: set invalid action logits to -inf
        # action_mask: (action_dim,) → expand to (1, action_dim)
        masked_logits = logits.clone()
        mask_expanded = action_mask.unsqueeze(0)  # (1, action_dim)
        masked_logits[~mask_expanded] = -1e10

        # Compute probabilities
        probs = F.softmax(masked_logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action] + 1e-10).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=logits.device)).item()

        return action, log_prob, value.item()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of (state, action, mask) tuples.

        Args:
            states: (batch, state_dim)
            actions: (batch,)
            action_masks: (batch, action_dim)

        Returns:
            (log_probs, values, entropies) — each (batch,)
        """
        logits, values = self.forward(states)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~action_masks] = -1e10

        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, values.squeeze(-1), entropies


# ═══════════════════════════════════════════════════════════════════════════
# PPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """PPO Agent with GAE and action masking."""

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig = None):
        self.cfg = config or PPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        # Actor-Critic network
        self.ac = ActorCritic(state_dim, action_dim, self.cfg.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.cfg.lr)

        # Training state
        self.episodes_done = 0
        self.total_steps = 0

    def select_action(
        self, state: np.ndarray, action_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select an action given state and mask.

        Args:
            state: (state_dim,) numpy array.
            action_mask: (action_dim,) boolean array.
            deterministic: greedy if True (for evaluation).

        Returns:
            (action_idx, log_prob, state_value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(action_mask).to(self.device)
        return self.ac.get_action(state_t, mask_t, deterministic)

    @torch.no_grad()
    def evaluate(
        self, state: np.ndarray, action_mask: np.ndarray
    ) -> Tuple[int, float]:
        """Greedy action selection (no log_prob/value needed)."""
        action, _, _ = self.select_action(state, action_mask, deterministic=True)
        return action, _

    def collect_episode(self, env) -> Dict[str, list]:
        """Run one episode and collect trajectory data.

        Returns dict with keys: states, actions, rewards, values,
                                 action_masks, log_probs, makespan, fatigue.
        """
        trajectory = {
            'states': [], 'actions': [], 'rewards': [],
            'values': [], 'action_masks': [], 'log_probs': [],
        }

        state = env.reset()
        done = False

        while not done:
            mask = env._get_action_mask()
            if not np.any(mask):
                break

            action, log_prob, value = self.select_action(state, mask)

            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['values'].append(value)
            trajectory['action_masks'].append(mask)
            trajectory['log_probs'].append(log_prob)

            next_state, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)

            state = next_state
            self.total_steps += 1

        trajectory['makespan'] = env.get_makespan()
        trajectory['fatigue'] = env.get_avg_fatigue()
        trajectory['total_reward'] = sum(trajectory['rewards'])

        return trajectory

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        last_value: float = 0.0,
        last_done: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Args:
            rewards: (T,) array of step rewards.
            values: (T,) array of V(s_t) estimates.
            last_value: V(s_{T+1}), 0 if terminal.
            last_done: True if episode ended normally.

        Returns:
            (returns, advantages) — both (T,)
        """
        gamma = self.cfg.gamma
        gae_lambda = self.cfg.gae_lambda
        T = len(rewards)

        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            if t == T - 1 and last_done:
                delta = rewards[t] - values[t]  # terminal: V(s_{T+1}) = 0
            else:
                delta = rewards[t] + gamma * next_value - values[t]

            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return returns, advantages

    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """PPO update from collected trajectories.

        Args:
            trajectories: List of trajectory dicts from collect_episode().

        Returns:
            Dict of loss metrics.
        """
        # Concatenate all trajectories and compute GAE for each
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        all_masks = []

        for traj in trajectories:
            rewards = np.array(traj['rewards'], dtype=np.float32)
            values = np.array(traj['values'], dtype=np.float32)

            # Compute GAE for this trajectory
            ret, adv = self.compute_gae(rewards, values)

            all_states.append(np.array(traj['states'], dtype=np.float32))
            all_actions.append(np.array(traj['actions'], dtype=np.int64))
            all_log_probs.append(np.array(traj['log_probs'], dtype=np.float32))
            all_returns.append(ret)
            all_advantages.append(adv)
            all_masks.append(np.array(traj['action_masks'], dtype=bool))

        # Concatenate
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        old_log_probs = np.concatenate(all_log_probs, axis=0)
        returns = np.concatenate(all_returns, axis=0)
        advantages = np.concatenate(all_advantages, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_samples = len(states)
        indices = np.arange(total_samples)
        batch_size = self.cfg.mini_batch_size

        metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'n_updates': 0}

        for epoch in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                s_batch = torch.FloatTensor(states[batch_idx]).to(self.device)
                a_batch = torch.LongTensor(actions[batch_idx]).to(self.device)
                old_lp_batch = torch.FloatTensor(old_log_probs[batch_idx]).to(self.device)
                ret_batch = torch.FloatTensor(returns[batch_idx]).to(self.device)
                adv_batch = torch.FloatTensor(advantages[batch_idx]).to(self.device)
                mask_batch = torch.BoolTensor(masks[batch_idx]).to(self.device)

                # Evaluate current policy on batch
                new_log_probs, values, entropies = self.ac.evaluate_actions(
                    s_batch, a_batch, mask_batch
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon,
                                    1.0 + self.cfg.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, ret_batch)

                # Entropy bonus (encourage exploration)
                entropy = entropies.mean()

                # Total loss
                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['n_updates'] += 1

        # Average
        n = max(metrics['n_updates'], 1)
        metrics['policy_loss'] /= n
        metrics['value_loss'] /= n
        metrics['entropy'] /= n

        return metrics

    def save(self, path: str, metadata: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'ac_network': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes_done': self.episodes_done,
            'total_steps': self.total_steps,
        }
        if metadata is not None:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint. Returns metadata if present."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episodes_done = checkpoint.get('episodes_done', 0)
        self.total_steps = checkpoint.get('total_steps', 0)

        metadata = checkpoint.get('metadata', None)
        if metadata is not None:
            print(f"  Loaded PPO: {metadata.get('problem_size', '?')} "
                  f"best_ms={metadata.get('performance', {}).get('best_makespan', '?')}")
        return metadata
