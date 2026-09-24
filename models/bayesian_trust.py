"""
bayesian_trust.py — Bayesian Beta-Binomial Trust Posterior for FL Clients

Each client maintains a Beta(α, β) posterior over its trustworthiness.
Per-round anomaly scores are converted into soft Bernoulli observations
and used to update the posterior. Outputs:
  - τ_hat : posterior mean trust   (replaces deterministic τ = 1 - a)
  - σ      : posterior std          (uncertainty signal for KALT v2)
  - ci_low : 5th percentile of Beta posterior
  - ci_high: 95th percentile of Beta posterior

Theory:
  Prior:   Beta(α_0, β_0)  →  prior mean = α_0/(α_0+β_0)
  Update:  soft observation strength k ∈ [0, 1]:
             α_new = α_old + k × s_pos
             β_new = β_old + k × s_neg
  where s_pos/s_neg are soft evidence weights derived from anomaly score.

  Posterior mean:  τ_hat = α / (α + β)
  Posterior var:   V = αβ / ((α+β)² (α+β+1))
  Posterior std:   σ = sqrt(V)
"""

import numpy as np


class BayesianTrust:
    """
    Beta-Binomial trust tracker for a single FL client.

    Parameters
    ----------
    alpha0 : float
        Prior α (pseudo-observations of trust). Default 2.0 → mildly optimistic.
    beta0  : float
        Prior β (pseudo-observations of distrust). Default 1.0.
    decay  : float
        Forgetting factor ∈ (0,1]. At each round, posteriors are pulled
        toward the prior by (1-decay). decay=1.0 = no forgetting.
    """

    def __init__(self, alpha0: float = 2.0, beta0: float = 1.0, decay: float = 0.95):
        self.alpha0 = float(alpha0)
        self.beta0  = float(beta0)
        self.decay  = float(decay)
        self.alpha  = float(alpha0)
        self.beta   = float(beta0)

    # ------------------------------------------------------------------
    def update(self, anomaly_score: float, strength: float = 1.0) -> dict:
        """
        Update posterior with a new anomaly observation.

        anomaly_score ∈ [0, 1]:
          0   → perfect trust, full positive evidence
          1   → certain attacker, full negative evidence
        strength ∈ (0, 1]:
          Scales the effective number of pseudo-observations per update.

        Returns dict with τ_hat, σ, ci_low, ci_high, alpha, beta.
        """
        a = float(np.clip(anomaly_score, 0.0, 1.0))

        # Soft positive/negative evidence weights
        # High anomaly → more β evidence; low anomaly → more α evidence
        # Use a smooth mapping: w_pos = (1-a)^1.5, w_neg = a^1.5
        w_pos = (1.0 - a) ** 1.5
        w_neg = a ** 1.5

        # Apply forgetting: shrink toward prior
        self.alpha = self.alpha0 + self.decay * (self.alpha - self.alpha0)
        self.beta  = self.beta0  + self.decay * (self.beta  - self.beta0)

        # Accumulate evidence
        self.alpha += strength * w_pos
        self.beta  += strength * w_neg

        return self.stats()

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return current posterior statistics."""
        a, b = self.alpha, self.beta
        s    = a + b

        mean  = a / s
        var   = (a * b) / (s * s * (s + 1.0))
        std   = float(np.sqrt(var))

        # Beta CDF percentiles via Normal approximation (accurate for large s)
        z_5  = mean - 1.645 * std
        z_95 = mean + 1.645 * std

        return {
            "tau_hat":  float(np.clip(mean, 0.0, 1.0)),
            "sigma":    float(std),
            "ci_low":   float(np.clip(z_5,  0.0, 1.0)),
            "ci_high":  float(np.clip(z_95, 0.0, 1.0)),
            "alpha":    float(a),
            "beta":     float(b),
        }

    def reset(self):
        """Reset to prior (e.g., client rejoins federation)."""
        self.alpha = self.alpha0
        self.beta  = self.beta0


class BayesianTrustPool:
    """
    Manages per-client BayesianTrust objects.

    Usage
    -----
        pool = BayesianTrustPool(n_clients=10)
        stats = pool.update(client_id=3, anomaly_score=0.72)
        tau_hat = stats["tau_hat"]
        sigma   = stats["sigma"]
    """

    def __init__(self, n_clients: int, alpha0: float = 2.0, beta0: float = 1.0,
                 decay: float = 0.95):
        self.trackers = {
            i: BayesianTrust(alpha0=alpha0, beta0=beta0, decay=decay)
            for i in range(n_clients)
        }

    def update(self, client_id: int, anomaly_score: float,
               strength: float = 1.0) -> dict:
        return self.trackers[client_id].update(anomaly_score, strength)

    def stats(self, client_id: int) -> dict:
        return self.trackers[client_id].stats()

    def all_stats(self) -> dict:
        return {cid: t.stats() for cid, t in self.trackers.items()}

    def tau_hat(self, client_id: int) -> float:
        return self.trackers[client_id].stats()["tau_hat"]

    def sigma(self, client_id: int) -> float:
        return self.trackers[client_id].stats()["sigma"]
