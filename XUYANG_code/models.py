try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - lets non-torch utilities import.
    torch = None
    nn = None


if nn is not None:

    class PairScoringNetwork(nn.Module):
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
            # features: [n_actions, feature_dim] or [batch, feature_dim]
            return self.net(features).squeeze(-1)


    class ActorCriticNetwork(nn.Module):
        def __init__(self, feature_dim, hidden_dim=128):
            super().__init__()
            self.actor = PairScoringNetwork(feature_dim, hidden_dim)
            self.value = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features, mask):
            logits = self.actor(features)
            logits = logits.masked_fill(~mask, -1e9)
            # Pool only valid (masked) action features for value estimation
            valid_mask = mask.float().unsqueeze(-1)
            pooled = (features * valid_mask).sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1e-6)
            value = self.value(pooled).squeeze(-1)
            return logits, value

else:

    class PairScoringNetwork:  # pragma: no cover
        pass


    class ActorCriticNetwork:  # pragma: no cover
        pass
