try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - lets non-torch utilities import.
    torch = None
    nn = None


if nn is not None:

    class PairScoringNetwork(nn.Module):
        """MLP that scores each action's feature vector independently.

        Returns a scalar Q-value per action.
        Input: [..., feature_dim]  ->  Output: [...] (scalar per action)
        """

        def __init__(self, feature_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features):
            return self.net(features).squeeze(-1)


    class ActorCriticNetwork(nn.Module):
        """Actor-Critic with independent actor and value networks.

        Actor and value have separate feature extractors (no shared layers),
        matching the original architecture that showed strong PPO results.
        """

        def __init__(self, feature_dim, hidden_dim=128):
            super().__init__()
            # Independent actor network
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Independent value network
            self.value_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features, mask):
            # Actor: score each action independently
            logits = self.actor(features).squeeze(-1)
            logits = logits.masked_fill(~mask, -1e9)

            # Value: mean-pool over valid actions only
            valid_mask = mask.float().unsqueeze(-1)
            pooled = (features * valid_mask).sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1e-6)
            value = self.value_net(pooled).squeeze(-1)
            return logits, value

else:

    class PairScoringNetwork:  # pragma: no cover
        pass


    class ActorCriticNetwork:  # pragma: no cover
        pass
