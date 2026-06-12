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


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def push(self, item):
        self.data.append(item)

    def sample(self, batch_size):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)


class DQNAgent:
    def __init__(self, feature_dim, cfg: TrainConfig, device=None):
        require_torch()
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(self.device)
        self.target = PairScoringNetwork(feature_dim, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.replay_size)
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

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        chosen = torch.tensor(
            np.stack([b.features[b.action] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_max = []
            for b in batch:
                if b.done or len(b.next_features) == 0 or not b.next_mask.any():
                    next_max.append(0.0)
                else:
                    nf = torch.tensor(b.next_features, dtype=torch.float32, device=self.device)
                    qn = self.target(nf)
                    mask = torch.tensor(b.next_mask, dtype=torch.bool, device=self.device)
                    qn = qn.masked_fill(~mask, -1e9)
                    next_max.append(float(qn.max().item()))
            next_max = torch.tensor(next_max, dtype=torch.float32, device=self.device)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_max

        pred = self.q(chosen)
        loss = F.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

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
