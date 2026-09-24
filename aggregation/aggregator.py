"""
aggregator.py
Risk-Adaptive FL Aggregation driven by KB trust and risk scores.

Strategies:
  - fedavg          : standard FedAvg (baseline)
  - trust_weighted  : weight by trust score
  - risk_penalty    : weight by (1 - risk_score)
  - ka_aggregate    : knowledge-aware (our method) — quarantine critical,
                      reduce high-risk, trust-weight the rest
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class AggStrategy(str, Enum):
    FEDAVG         = "fedavg"
    TRUST_WEIGHTED = "trust_weighted"
    RISK_PENALTY   = "risk_penalty"
    KA_AGGREGATE   = "ka_aggregate"   # our method

@dataclass
class ClientUpdate:
    client_id    : int
    weights      : np.ndarray
    n_samples    : int
    trust_score  : float = 1.0
    anomaly_score: float = 0.0
    risk_level   : str   = "low"
    quarantined  : bool  = False


class FLAggregator:
    def __init__(self, strategy: AggStrategy = AggStrategy.KA_AGGREGATE,
                 quarantine_threshold: str = "critical",
                 reduce_threshold: str = "high",
                 reduce_factor: float = 0.1):
        self.strategy             = strategy
        self.quarantine_threshold = quarantine_threshold   # risk level → exclude
        self.reduce_threshold     = reduce_threshold       # risk level → down-weight
        self.reduce_factor        = reduce_factor          # weight for high-risk

        # Risk level ordering
        self._risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        self._quar_level = self._risk_order[quarantine_threshold]
        self._red_level  = self._risk_order[reduce_threshold]

    def aggregate(self, updates: List[ClientUpdate],
                  round_id: int = 0) -> Tuple[np.ndarray, dict]:
        if self.strategy == AggStrategy.FEDAVG:
            return self._fedavg(updates)
        if self.strategy == AggStrategy.TRUST_WEIGHTED:
            return self._trust_weighted(updates)
        if self.strategy == AggStrategy.RISK_PENALTY:
            return self._risk_penalty(updates)
        if self.strategy == AggStrategy.KA_AGGREGATE:
            return self._ka_aggregate(updates, round_id)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    # ── Standard FedAvg ───────────────────────────────────────────────────────
    def _fedavg(self, updates: List[ClientUpdate]):
        total = sum(u.n_samples for u in updates)
        agg   = sum(u.weights * (u.n_samples / total) for u in updates)
        info  = {"strategy": "fedavg", "included": len(updates),
                 "excluded": 0, "weights": {u.client_id: u.n_samples/total
                                             for u in updates}}
        return agg, info

    # ── Trust-weighted ────────────────────────────────────────────────────────
    def _trust_weighted(self, updates: List[ClientUpdate]):
        weights = np.array([u.n_samples * u.trust_score for u in updates])
        total   = weights.sum()
        if total < 1e-9: total = 1.0
        agg = sum(u.weights * (w / total)
                  for u, w in zip(updates, weights))
        info = {"strategy": "trust_weighted", "included": len(updates),
                "excluded": 0,
                "weights": {u.client_id: round(float(w/total),4)
                             for u,w in zip(updates, weights)}}
        return agg, info

    # ── Risk-penalty ──────────────────────────────────────────────────────────
    def _risk_penalty(self, updates: List[ClientUpdate]):
        weights = np.array([u.n_samples * (1.0 - u.anomaly_score)
                            for u in updates])
        total   = weights.sum()
        if total < 1e-9: total = 1.0
        agg = sum(u.weights * (w / total)
                  for u, w in zip(updates, weights))
        info = {"strategy": "risk_penalty", "included": len(updates),
                "excluded": 0,
                "weights": {u.client_id: round(float(w/total),4)
                             for u,w in zip(updates, weights)}}
        return agg, info

    # ── Knowledge-Aware Aggregation (our method) ──────────────────────────────
    def _ka_aggregate(self, updates: List[ClientUpdate], round_id: int):
        included, excluded, reduced = [], [], []

        for u in updates:
            rl = self._risk_order.get(u.risk_level, 0)
            if rl >= self._quar_level:
                u.quarantined = True
                excluded.append(u)
            elif rl >= self._red_level:
                reduced.append(u)
                included.append(u)
            else:
                included.append(u)

        if not included:
            # Fallback: use all with very low weights if everyone excluded
            included = updates
            excluded = []

        # Compute weights: trust_score × n_samples, halved for reduced clients
        raw_w = []
        for u in included:
            rl = self._risk_order.get(u.risk_level, 0)
            w  = u.n_samples * u.trust_score
            if rl >= self._red_level:
                w *= self.reduce_factor   # penalise high-risk
            raw_w.append(w)

        total = sum(raw_w)
        if total < 1e-9: total = 1.0
        norm_w = [w / total for w in raw_w]

        agg = sum(u.weights * w for u, w in zip(included, norm_w))

        info = {
            "strategy"   : "ka_aggregate",
            "round"      : round_id,
            "included"   : len(included),
            "excluded"   : len(excluded),
            "reduced"    : len(reduced),
            "quarantined": [u.client_id for u in excluded],
            "weights"    : {u.client_id: round(float(w), 4)
                            for u, w in zip(included, norm_w)},
        }
        return agg, info

    def print_aggregation_info(self, info: dict, round_id: int):
        print(f"\n  [Aggregation R{round_id}] strategy={info['strategy']} | "
              f"included={info['included']} | excluded={info['excluded']}")
        if info.get("quarantined"):
            print(f"  → Quarantined clients: {info['quarantined']}")
        print(f"  → Weights: {info.get('weights', {})}")
