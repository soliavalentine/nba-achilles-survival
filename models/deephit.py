"""
DeepHit competing-risks survival model for Achilles tendon rupture.

Architecture:
  - Shared sub-network: MLP over the covariate vector x
  - Cause-specific output heads: one per event type (rupture vs. other retirement)
  - Output: joint density h(t, k | x)  for time t and cause k

Loss:
  L = alpha * L_log_likelihood + (1 - alpha) * L_ranking

  L_log_likelihood:  negative log-likelihood of the observed event time/type
  L_ranking:         pairwise concordance ranking loss (pairs with known ordering)

Reference: Lee et al. (2018) "DeepHit: A Deep Learning Approach to Survival
           Analysis with Competing Risks"

Training entry point: models/train.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared sub-network
# ---------------------------------------------------------------------------

class SharedNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_features = prev

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Cause-specific output head
# ---------------------------------------------------------------------------

class CauseSpecificHead(nn.Module):
    """
    Maps shared representation to a probability mass function over T time bins
    for a single cause.
    """

    def __init__(self, in_features: int, n_time_bins: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_time_bins),
        )

    def forward(self, z: Tensor) -> Tensor:
        # Raw logits over time bins
        return self.net(z)


# ---------------------------------------------------------------------------
# Full DeepHit model
# ---------------------------------------------------------------------------

class DeepHit(nn.Module):
    """
    DeepHit competing-risks model.

    Args:
        in_features:   Number of input covariates.
        n_time_bins:   Number of discrete time intervals.
        n_causes:      Number of competing event types (default=2: rupture + other).
        shared_layers: Hidden layer sizes for the shared sub-network.
        cs_hidden:     Hidden layer size for each cause-specific head.
        dropout:       Dropout rate in shared network.
    """

    def __init__(
        self,
        in_features: int,
        n_time_bins: int,
        n_causes: int = 2,
        shared_layers: list[int] | None = None,
        cs_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        if shared_layers is None:
            shared_layers = [256, 128]

        self.shared = SharedNet(in_features, shared_layers, dropout=dropout)
        self.heads = nn.ModuleList(
            [
                CauseSpecificHead(self.shared.out_features, n_time_bins, cs_hidden)
                for _ in range(n_causes)
            ]
        )
        self.n_time_bins = n_time_bins
        self.n_causes = n_causes

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns:
            h: Tensor of shape (batch, n_causes, n_time_bins)
               Joint sub-distribution PMF over (cause, time).
               Rows across causes AND times sum to ≤ 1 (residual = survival).
        """
        z = self.shared(x)
        logits = torch.stack([head(z) for head in self.heads], dim=1)
        # Softmax jointly over (causes × time) so probabilities sum to 1
        batch = logits.shape[0]
        flat = logits.view(batch, -1)
        h = F.softmax(flat, dim=-1).view(batch, self.n_causes, self.n_time_bins)
        return h

    def survival_function(self, x: Tensor) -> Tensor:
        """
        S(t | x) = P(T > t, no event by t)
                 = 1 - CIF_1(t) - CIF_2(t) - ...
        Returns: (batch, n_time_bins+1)  — S(0) = 1 prepended.
        """
        h = self.forward(x)
        # CIF_k(t) = sum_{s<=t} h_k(s)
        cif_all = h.sum(dim=1).cumsum(dim=-1)  # (batch, n_time_bins)
        S = 1.0 - cif_all
        ones = torch.ones(S.shape[0], 1, device=S.device)
        return torch.cat([ones, S], dim=-1)

    def cif(self, x: Tensor, cause: int) -> Tensor:
        """
        CIF_k(t | x) = P(T ≤ t, K = k | x)
        Returns: (batch, n_time_bins+1) with CIF(0) = 0 prepended.
        """
        h = self.forward(x)
        cif_k = h[:, cause, :].cumsum(dim=-1)
        zeros = torch.zeros(cif_k.shape[0], 1, device=cif_k.device)
        return torch.cat([zeros, cif_k], dim=-1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def log_likelihood_loss(
    h: Tensor,
    t: Tensor,
    e: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """
    Negative log-likelihood for DeepHit.

    Args:
        h:   (batch, n_causes, n_time_bins)  — joint PMF from forward().
        t:   (batch,) int64 — observed time bin index (0-indexed).
        e:   (batch,) int64 — event indicator (0 = censored, 1..n_causes = event cause).

    Returns:
        Scalar mean NLL.
    """
    batch = h.shape[0]
    device = h.device

    # Probability of observed event
    # For cause k at time t:  h[i, k-1, t[i]]
    nll = torch.zeros(batch, device=device)
    event_mask = e > 0  # uncensored

    if event_mask.any():
        cause_idx = (e[event_mask] - 1).long()
        time_idx = t[event_mask].long()
        h_obs = h[event_mask][torch.arange(event_mask.sum()), cause_idx, time_idx]
        nll[event_mask] = -torch.log(h_obs + eps)

    return nll.mean()


def ranking_loss(
    h: Tensor,
    t: Tensor,
    e: Tensor,
    sigma: float = 0.1,
) -> Tensor:
    """
    Pairwise ranking loss encouraging correct ordering of predicted risk.

    For every pair (i, j) where subject i had an event before j was censored,
    penalise if the model assigns lower risk to i than j.
    """
    batch = h.shape[0]
    # CIF at observed time for each subject summed over causes
    cif_at_t = torch.stack(
        [h[:, k, :].cumsum(dim=-1)[torch.arange(batch), t.long()] for k in range(h.shape[1])],
        dim=1,
    ).sum(dim=1)  # (batch,)

    loss = torch.tensor(0.0, device=h.device)
    n_pairs = 0

    for i in range(batch):
        for j in range(batch):
            # Pair is valid when i had an observed event before j's observed time
            if e[i] > 0 and t[i] < t[j]:
                loss += torch.exp(-(cif_at_t[i] - cif_at_t[j]) / sigma)
                n_pairs += 1

    return loss / max(n_pairs, 1)


def deephit_loss(
    h: Tensor,
    t: Tensor,
    e: Tensor,
    alpha: float = 0.5,
    sigma: float = 0.1,
) -> Tensor:
    """Combined DeepHit loss: alpha * NLL + (1-alpha) * ranking."""
    nll = log_likelihood_loss(h, t, e)
    rank = ranking_loss(h, t, e, sigma=sigma)
    return alpha * nll + (1.0 - alpha) * rank
