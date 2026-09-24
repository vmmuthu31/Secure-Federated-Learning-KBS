"""
temporal_tracker.py — Temporal Anomaly State Tracker for FL Clients

Maintains a sliding-window history of per-client anomaly scores and
derives four temporal signals that feed into Rules R15-R18 and KALT v2:

  ema       : Exponential moving average of anomaly (β=0.7)
              Smoothed current risk estimate.

  trend     : Slope of linear regression on last K anomaly scores.
              Positive trend → escalating attack; negative → de-escalating.

  variance  : Rolling variance of anomaly over window K.
              High variance → intermittent/evasion pattern.

  streak    : Consecutive rounds with anomaly > threshold.
              Long streak → persistent attacker.

Plus a cross-client coordination signal computed at pool level:
  n_simultan: Number of clients simultaneously anomalous this round
              (used for R18: coordinated attack detection).
"""

import numpy as np
from collections import deque


class ClientTemporalState:
    """Temporal anomaly state for one client."""

    def __init__(self, window: int = 5, ema_beta: float = 0.7,
                 high_anomaly_th: float = 0.45):
        self.window       = window
        self.ema_beta     = ema_beta
        self.high_th      = high_anomaly_th
        self.history      = deque(maxlen=window)   # raw anomaly scores
        self.ema          = None                    # lazy init
        self.streak       = 0                       # consecutive high-anomaly rounds

    def update(self, anomaly_score: float) -> dict:
        a = float(np.clip(anomaly_score, 0.0, 1.0))

        # EMA update
        if self.ema is None:
            self.ema = a
        else:
            self.ema = self.ema_beta * self.ema + (1.0 - self.ema_beta) * a

        # Streak
        if a >= self.high_th:
            self.streak += 1
        else:
            self.streak  = 0

        self.history.append(a)
        return self._compute_signals(a)

    def _compute_signals(self, current: float) -> dict:
        h = list(self.history)
        n = len(h)

        # Trend: slope of linear fit on window (0 if < 2 points)
        if n >= 2:
            xs    = np.arange(n, dtype=float)
            xs   -= xs.mean()
            ys    = np.array(h, dtype=float)
            ys   -= ys.mean()
            denom = float(np.dot(xs, xs))
            trend = float(np.dot(xs, ys) / denom) if denom > 1e-9 else 0.0
        else:
            trend = 0.0

        # Variance
        variance = float(np.var(h)) if n >= 2 else 0.0

        return {
            "ema":         float(self.ema),
            "trend":       trend,           # positive = escalating
            "variance":    variance,        # high = evasion/intermittent
            "streak":      self.streak,     # consecutive high-anomaly rounds
            "history_len": n,
            "current":     current,
        }

    def signals(self, current: float = None) -> dict:
        """Return signals without updating. current overrides last entry if given."""
        a = float(self.history[-1]) if self.history else 0.0
        if current is not None:
            a = float(current)
        return self._compute_signals(a)


class TemporalTrackerPool:
    """
    Manages per-client temporal states + cross-client coordination signal.

    Usage
    -----
        pool  = TemporalTrackerPool(n_clients=10)
        sigs  = pool.update(client_id=3, anomaly_score=0.72)
        coord = pool.coordination_signal(threshold=0.45)
    """

    def __init__(self, n_clients: int, window: int = 5,
                 ema_beta: float = 0.7, high_anomaly_th: float = 0.45):
        self.trackers = {
            i: ClientTemporalState(window=window, ema_beta=ema_beta,
                                   high_anomaly_th=high_anomaly_th)
            for i in range(n_clients)
        }
        self._last_anomalies: dict = {}   # cid → last anomaly score

    def update(self, client_id: int, anomaly_score: float) -> dict:
        self._last_anomalies[client_id] = anomaly_score
        return self.trackers[client_id].update(anomaly_score)

    def signals(self, client_id: int) -> dict:
        return self.trackers[client_id].signals()

    def coordination_signal(self, threshold: float = 0.45) -> int:
        """
        Number of clients simultaneously above threshold this round.
        > 1 → possible coordinated attack (R18).
        """
        return sum(
            1 for a in self._last_anomalies.values() if a >= threshold
        )

    def all_signals(self) -> dict:
        return {cid: t.signals() for cid, t in self.trackers.items()}


# -----------------------------------------------------------------------
# R15-R18: Temporal Rule Evaluators
# -----------------------------------------------------------------------

def evaluate_temporal_rules(client_id: int, sigs: dict,
                             n_coordinated: int,
                             base_attack: str = "Unknown",
                             base_level: str = "LOW") -> dict:
    """
    Apply temporal rules R15-R18 on top of base KB classification.

    Returns dict:
      {attack, level, rules_fired, upgraded}
    where upgraded=True means temporal rules changed the classification.
    """
    attack  = base_attack
    level   = base_level
    fired   = []
    upgraded = False

    streak   = sigs.get("streak",   0)
    trend    = sigs.get("trend",    0.0)
    variance = sigs.get("variance", 0.0)
    ema      = sigs.get("ema",      0.0)
    current  = sigs.get("current",  0.0)

    # R15 — Persistent attacker: ≥3 consecutive high-anomaly rounds
    if streak >= 3 and ema > 0.40:
        fired.append("R15")
        level   = "CRITICAL"
        upgraded = True
        if attack in ("Unknown", "LOW"):
            attack = "Persistent_Byzantine"

    # R16 — Escalating risk: positive trend + EMA already elevated
    elif trend > 0.08 and ema > 0.30 and level not in ("CRITICAL",):
        fired.append("R16")
        # Escalate one level
        if level == "LOW":
            level = "MEDIUM"
        elif level == "MEDIUM":
            level = "HIGH"
        upgraded = True
        if attack in ("Unknown",):
            attack = "Escalating_Threat"

    # R17 — Evasion pattern: high variance + moderate mean
    # Intermittent anomaly suggests an adaptive adversary hiding behind benign rounds
    if variance > 0.04 and 0.25 < ema < 0.65 and level not in ("CRITICAL", "HIGH"):
        fired.append("R17")
        if level == "LOW":
            level = "MEDIUM"
        upgraded = True
        if attack in ("Unknown", "Escalating_Threat"):
            attack = "Model_Poison"   # evasion is a model-poison signature

    # R18 — Coordinated attack: multiple clients simultaneously anomalous
    if n_coordinated >= 2 and current > 0.45 and level not in ("CRITICAL",):
        fired.append("R18")
        if level in ("LOW", "MEDIUM"):
            level = "HIGH"
        upgraded = True
        if attack in ("Unknown", "Escalating_Threat"):
            attack = "Byzantine"

    return {
        "attack":   attack,
        "level":    level,
        "rules_fired": fired,
        "upgraded":  upgraded,
    }
