"""
ka_trainer.py — KALT v2: Knowledge-Aware Adaptive Local Trainer

Upgrades from v1:
  - Uses Bayesian posterior (τ_hat, σ) instead of deterministic τ = 1−a
  - Adds temporal EMA trend signal to all four adaptation equations
  - New Eq 5: Uncertainty-gated gradient clipping (σ tightens clip independently)
  - New Eq 6: Trend-aware epoch adaptation (escalating trend → fewer epochs)

KALT v2 equations
-----------------
  Eq 1 (LR):     η* = clip(η_base × τ_hat × exp(−σ),          η_min, η_max)
  Eq 2 (L2):     λ* = clip(λ_base × (1 + 10·a) × (1 + trend⁺), λ_base, λ_max)
  Eq 3 (Clip):   γ* = clip(ĝ_c × (2·τ_hat + 0.1),             γ_min, γ_max)
  Eq 5 (Unc):    γ* = γ* × max(0.3, 1 − 2σ)                   [uncertainty gate]
  Eq 4/6 (Ep):   E* = E_max   if τ_hat ≥ τ_high AND trend ≤ 0
                      E_min   if a ≥ a_high OR trend > 0.10
                      interpolated otherwise

where trend⁺ = max(0, trend)  (only escalation drives stronger L2)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class KALTConfig:
    # Base hyperparameters
    base_lr:        float = 0.01
    base_l2:        float = 1e-4
    base_grad_clip: float = 5.0
    base_epochs:    int   = 3
    batch_size:     int   = 64

    # LR bounds
    lr_min:  float = 0.001
    lr_max:  float = 0.05

    # L2 bounds
    l2_max:  float = 0.05

    # Clip bounds
    clip_min: float = 1.0
    clip_max: float = 10.0

    # Epoch bounds
    epoch_min: int = 2
    epoch_max: int = 7

    # Decision thresholds
    high_trust_th: float = 0.75   # τ_hat threshold for max epochs
    high_risk_th:  float = 0.50   # anomaly threshold for min epochs
    high_trend_th: float = 0.10   # trend threshold for epoch reduction


class KALTAdapter:
    """
    KALT v2 adapter.

    Parameters
    ----------
    cfg : KALTConfig

    adapt(tau_hat, sigma, anomaly_score, trend, kb_grad_norm, round_idx)
        → dict of adapted hyperparameters
    """

    def __init__(self, cfg: KALTConfig = None):
        self.cfg = cfg or KALTConfig()

    def adapt(self,
              tau_hat:       float = 0.667,
              sigma:         float = 0.20,
              anomaly_score: float = 0.0,
              trend:         float = 0.0,
              kb_grad_norm:  float = None,
              round_idx:     int   = 0) -> dict:
        """
        Compute adapted hyperparameters for one client this round.

        Parameters
        ----------
        tau_hat       : Bayesian posterior mean trust score ∈ [0,1]
        sigma         : Bayesian posterior std (uncertainty) ∈ [0, ~0.25]
        anomaly_score : KB multi-factor anomaly score ∈ [0,1]
        trend         : Temporal EMA trend (slope of last-K anomalies)
        kb_grad_norm  : KB-logged gradient norm from previous round (for clip)
        round_idx     : Current FL round (round 0 uses base params)
        """
        cfg = self.cfg

        if round_idx == 0:
            return self._base_params()

        tau   = float(np.clip(tau_hat, 0.0, 1.0))
        sig   = float(np.clip(sigma,   0.0, 0.5))
        a     = float(np.clip(anomaly_score, 0.0, 1.0))
        tr    = float(trend)
        tr_p  = max(0.0, tr)   # positive trend only (escalation)

        # ── Eq 1: Adaptive Learning Rate ─────────────────────────────
        # Uncertainty penalty: exp(−σ) shrinks LR as uncertainty grows
        lr = cfg.base_lr * tau * np.exp(-sig)
        lr = float(np.clip(lr, cfg.lr_min, cfg.lr_max))

        # ── Eq 2: Adaptive L2 Regularization ─────────────────────────
        # Escalation trend multiplies L2 further: (1 + trend⁺) ≥ 1
        l2 = cfg.base_l2 * (1.0 + 10.0 * a) * (1.0 + tr_p)
        l2 = float(np.clip(l2, cfg.base_l2, cfg.l2_max))

        # ── Eq 3: Base gradient clip (from KB-logged norm) ───────────
        g_ref = float(kb_grad_norm) if kb_grad_norm and kb_grad_norm > 0 else cfg.base_grad_clip
        clip  = g_ref * (2.0 * tau + 0.1)
        clip  = float(np.clip(clip, cfg.clip_min, cfg.clip_max))

        # ── Eq 5: Uncertainty gate on clip ───────────────────────────
        # High uncertainty → tighter clip regardless of trust
        unc_gate = max(0.30, 1.0 - 2.0 * sig)
        clip     = float(np.clip(clip * unc_gate, cfg.clip_min, cfg.clip_max))

        # ── Eq 4/6: Adaptive epochs with trend adjustment ─────────────
        if tau >= cfg.high_trust_th and tr <= 0.0:
            epochs = cfg.epoch_max
        elif a >= cfg.high_risk_th or tr > cfg.high_trend_th:
            epochs = cfg.epoch_min
        else:
            # Interpolate: τ_hat drives most of it, negative trend helps
            frac   = tau * max(0.5, 1.0 - 5.0 * tr_p)
            epochs = int(round(cfg.epoch_min + frac * (cfg.epoch_max - cfg.epoch_min)))
            epochs = int(np.clip(epochs, cfg.epoch_min, cfg.epoch_max))

        return {
            "lr":            lr,
            "l2_lambda":     l2,
            "grad_clip":     clip,
            "epochs":        epochs,
            "batch_size":    cfg.batch_size,
            # audit fields
            "_tau_hat":      tau,
            "_sigma":        sig,
            "_anomaly":      a,
            "_trend":        tr,
            "_unc_gate":     unc_gate,
            "_g_ref":        g_ref,
        }

    def _base_params(self) -> dict:
        cfg = self.cfg
        return {
            "lr":         cfg.base_lr,
            "l2_lambda":  cfg.base_l2,
            "grad_clip":  cfg.base_grad_clip,
            "epochs":     cfg.base_epochs,
            "batch_size": cfg.batch_size,
            "_tau_hat":   0.667, "_sigma": 0.20,
            "_anomaly":   0.0,   "_trend": 0.0,
            "_unc_gate":  1.0,   "_g_ref": cfg.base_grad_clip,
        }

    def adapt_log(self, client_id: int, round_idx: int,
                  tau_hat: float, sigma: float,
                  anomaly_score: float, trend: float,
                  adapted: dict) -> str:
        return (
            f"[KALT-v2 R{round_idx:02d} C{client_id}] "
            f"τ_hat={tau_hat:.3f} σ={sigma:.3f} a={anomaly_score:.3f} trend={trend:+.3f} | "
            f"lr={adapted['lr']:.5f} l2={adapted['l2_lambda']:.5f} "
            f"clip={adapted['grad_clip']:.2f} ep={adapted['epochs']}"
        )
