import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from config import TrainConfig
from models import ActorCriticNetwork, PairScoringNetwork


def require_torch():
    if torch is None:
        raise RuntimeError(f"PyTorch is required for RL training: {TORCH_IMPORT_ERROR}")


@dataclass
class Transition:
    features: np.ndarray
    action: int
    reward: float
    next_features: np.ndarray
    next_mask: np.ndarray
    done: bool
    # When n_step_return is pre-computed, set is_n_step=True and store it in reward.
    # The update() will use reward directly as target (no TD bootstrap).
    is_n_step: bool = False


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay with importance-sampling weights."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.frame = 0  # counts number of times sample() is called, for beta annealing

    def push(self, item):
        """Store transition with max priority (ensures new experiences get sampled)."""
        max_prio = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch with proportional prioritization; returns (batch, indices, IS_weights)."""
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / 50000)

        if self.size < batch_size:
            return None, None, None

        prios = self.priorities[: self.size]
        probs = np.power(prios, self.alpha)
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Importance-sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        weights = np.power(self.size * probs[indices], -beta)
        weights /= weights.max()  # normalize for stability

        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update priorities after a learning step."""
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td_err) + 1e-6)

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, feature_dim, cfg: TrainConfig, device=None):
        require_torch()
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(self.device)
        self.target = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplayBuffer(cfg.replay_size, cfg.per_alpha, cfg.per_beta)
        self.epsilon = cfg.epsilon_start
        self.steps = 0

    def act(self, obs, explore=True):
        mask = obs["mask"]
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0
        if explore and random.random() < self.epsilon:
            return int(random.choice(legal))
        with torch.no_grad():
            feats = torch.tensor(obs["features"], dtype=torch.float32, device=self.device)
            q_values = self.q(feats).detach().cpu().numpy()
        q_values[~mask] = -1e9
        return int(np.argmax(q_values))

    def remember(self, transition):
        self.buffer.push(transition)

    def bootstrap_value(self, features, mask):
        """Return max Q-value for n-step bootstrapping."""
        with torch.no_grad():
            nf = torch.tensor(features, dtype=torch.float32, device=self.device)
            qn = self.target(nf)
            m = torch.tensor(mask, dtype=torch.bool, device=self.device)
            qn = qn.masked_fill(~m, -1e9)
            return float(qn.max().item())

    def update(self):
        result = self.buffer.sample(self.cfg.batch_size)
        if result[0] is None:
            return None
        batch, indices, is_weights = result

        chosen = torch.tensor(
            np.stack([b.features[b.action] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            targets = []
            for b in batch:
                if b.is_n_step:
                    # n-step return already pre-computed
                    targets.append(b.reward)
                elif b.done or len(b.next_features) == 0 or not b.next_mask.any():
                    targets.append(b.reward)
                else:
                    nf = torch.tensor(b.next_features, dtype=torch.float32, device=self.device)
                    qn = self.target(nf)
                    mask = torch.tensor(b.next_mask, dtype=torch.bool, device=self.device)
                    qn = qn.masked_fill(~mask, -1e9)
                    targets.append(b.reward + self.cfg.gamma * float(qn.max().item()))
            target = torch.tensor(targets, dtype=torch.float32, device=self.device)

        pred = self.q(chosen)
        td_errors = (pred - target).detach().cpu().numpy()
        element_loss = F.smooth_l1_loss(pred, target, reduction="none")
        loss = (weights_t * element_loss).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        # Update priorities based on TD-error
        self.buffer.update_priorities(indices, td_errors)

        self.steps += 1
        if self.steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        """Decay epsilon once per episode (called from training loop)."""
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def decay_lr(self, factor):
        """Decay learning rate by factor."""
        for param_group in self.opt.param_groups:
            param_group["lr"] *= factor

    def save(self, path):
        torch.save({"model": self.q.state_dict(), "epsilon": self.epsilon}, path)


class PPOAgent:
    def __init__(self, feature_dim, cfg: TrainConfig, device=None):
        require_torch()
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCriticNetwork(feature_dim, cfg.hidden_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def act(self, obs, explore=True):
        features = torch.tensor(obs["features"], dtype=torch.float32, device=self.device)
        mask = torch.tensor(obs["mask"], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, value = self.net(features, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if explore else torch.argmax(logits)
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def update(self, rollout):
        if not rollout:
            return None
        rewards = [x["reward"] for x in rollout]
        dones = [x["done"] for x in rollout]
        values = [x["value"] for x in rollout]

        returns = []
        g = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            g = reward + self.cfg.gamma * g * (1.0 - float(done))
            returns.append(g)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        adv_t = returns_t - values_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Store per-item return and advantage BEFORE shuffling (fixes shuffle bug)
        for i, item in enumerate(rollout):
            item["_return"] = returns_t[i]
            item["_adv"] = adv_t[i]

        losses = []
        for _ in range(self.cfg.ppo_epochs):
            total = 0.0
            random.shuffle(rollout)
            for item in rollout:
                features = torch.tensor(item["features"], dtype=torch.float32, device=self.device)
                mask = torch.tensor(item["mask"], dtype=torch.bool, device=self.device)
                action = torch.tensor(item["action"], dtype=torch.int64, device=self.device)
                old_logp = torch.tensor(item["logp"], dtype=torch.float32, device=self.device)

                logits, value = self.net(features, mask)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(action)
                ratio = torch.exp(logp - old_logp)
                adv = item["_adv"]
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv,
                )
                value_loss = F.mse_loss(value, item["_return"])
                entropy = dist.entropy()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                total += float(loss.item())
            losses.append(total / max(1, len(rollout)))
        return float(np.mean(losses))

    def decay_lr(self, factor):
        """Decay learning rate by factor."""
        for param_group in self.opt.param_groups:
            param_group["lr"] *= factor

    def save(self, path):
        torch.save({"model": self.net.state_dict()}, path)
