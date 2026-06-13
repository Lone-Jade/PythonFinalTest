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


    class DuelingPairScoringNetwork(nn.Module):
        """Dueling DQN: separates state-value V(s) and advantage A(s,a).

        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

        The feature vector layout is [global(6), worker(4), action(9)] = 19 dims.
        V(s) uses only state features (first 10 dims); A(s,a) uses full features.
        This separation helps when many actions have similar Q-values (common in
        large-scale scheduling where many job assignments are roughly equivalent).
        """

        def __init__(self, feature_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = 10  # global(6) + worker(4)

            # Value stream: V(s) from state features only
            self.value_net = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

            # Advantage stream: A(s,a) from full features
            self.advantage_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features):
            # features: (n_actions, feature_dim)
            # State value V(s) — use first action's state features (all share same state)
            state_feats = features[:, : self.state_dim].mean(dim=0, keepdim=True)
            v = self.value_net(state_feats)  # (1, 1)

            # Advantages A(s,a) — per-action
            a = self.advantage_net(features).squeeze(-1)  # (n_actions,)

            # Q(s,a) = V(s) + A(s,a) - mean(A)
            q = v.squeeze(-1) + (a - a.mean())
            return q


    class ScaleInvariantDuelingNetwork(nn.Module):
        """Dueling DQN with scale-invariant feature representations.

        Key improvements over DuelingPairScoringNetwork:
        1. Input LayerNorm — normalizes features to consistent distribution across instance sizes
        2. Separate state/action encoders — state (global+worker) and action features
           processed independently before fusion, preventing scale leakage
        3. LayerNorm in hidden layers (Pre-LN style) — stabilizes training across scales
        4. State-action fusion with residual — combines state and action representations

        Feature layout: [global(6), worker(4), action(9)] = 19 dims
        State encoder: 10 dims → emb
        Action encoder: 9 dims → emb  (with state context concatenated)
        Q = V(state_emb) + [A(fusion(state_emb, action_emb)) - mean(A)]
        """

        def __init__(self, feature_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = 10  # global(6) + worker(4)
            self.action_dim = feature_dim - self.state_dim  # 9

            # ── Input normalization ──
            self.input_norm = nn.LayerNorm(feature_dim)

            # ── State encoder (shared across all actions) ──
            self.state_encoder = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            # ── Action encoder (per-job features, with state context) ──
            self.action_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),  # full features for action context
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            # ── Value head: V(s) from state only ──
            self.value_head = nn.Linear(hidden_dim, 1)

            # ── Advantage head: A(s,a) from fused state+action ──
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features):
            # features: (n_actions, feature_dim)
            normed = self.input_norm(features)

            # ── State embedding (shared, from first 10 dims) ──
            state_feats = normed[:, : self.state_dim].mean(dim=0, keepdim=True)
            state_emb = self.state_encoder(state_feats)  # (1, hidden_dim)

            # ── Action embeddings (per-action, from full features) ──
            action_emb = self.action_encoder(normed)  # (n_actions, hidden_dim)

            # ── Value V(s) ──
            v = self.value_head(state_emb)  # (1, 1)

            # ── Advantage A(s,a) = head([state_emb | action_emb]) ──
            state_expanded = state_emb.expand(action_emb.shape[0], -1)  # (n_actions, hidden_dim)
            fused = torch.cat([state_expanded, action_emb], dim=-1)  # (n_actions, 2*hidden_dim)
            a = self.advantage_head(fused).squeeze(-1)  # (n_actions,)

            # ── Q(s,a) = V(s) + A(s,a) - mean(A) ──
            q = v.squeeze(-1) + (a - a.mean())
            return q


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
