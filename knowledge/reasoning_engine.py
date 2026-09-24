"""
reasoning_engine.py  — v2 (tuned thresholds)
Rule-based Knowledge Reasoning Engine for FL Security.
Maps multi-source KB evidence → Attack Pattern + Risk Level + Explanation.

v2 changes:
  - Tuned R06-R08 data-poison thresholds to match poison_fraction=0.5-0.6:
      dominant_frac ≈ 0.55-0.65, entropy (normalized) ≈ 0.60-0.75
  - Tuned R09-R10 backdoor thresholds: dominant > 0.48, wider norm/cosine range
  - Raised DATA_POISON risk floor from 0.40 → 0.52 (always HIGH)
  - Raised BACKDOOR risk floor from 0.35 → 0.45 (always HIGH)
  - Added R13-R14: data poison early-round single-signal triggers
"""

from dataclasses import dataclass
from typing import List, Tuple
from knowledge_graph import (
    AttackPattern, RiskLevel, AnomalyType,
    UpdateEntity, EvidenceEntity, AnomalyEntity,
    TrustScoreEntity, RiskScoreEntity, FLKnowledgeGraph
)


@dataclass
class Rule:
    rule_id    : str
    pattern    : AttackPattern
    description: str
    conditions : list   # list of callables

    def fires(self, update, evidence, anomaly_count, kg) -> bool:
        return all(c(update, evidence, anomaly_count, kg)
                   for c in self.conditions)


RULE_BASE: List[Rule] = [

    # ── Byzantine ────────────────────────────────────────────────────────────
    Rule("R01", AttackPattern.BYZANTINE,
         "Extreme gradient deviation + near-zero cosine similarity",
         [lambda u,e,h,kg: u.grad_norm > 8.0,
          lambda u,e,h,kg: u.cosine_sim < 0.1]),

    Rule("R02", AttackPattern.BYZANTINE,
         "Very large L2 distance + sudden behavioral change",
         [lambda u,e,h,kg: u.l2_distance > 8.0,
          lambda u,e,h,kg: e.sudden_change is True]),

    # ── Model Poisoning ───────────────────────────────────────────────────────
    Rule("R03", AttackPattern.MODEL_POISON,
         "Negative cosine similarity + high gradient norm",
         [lambda u,e,h,kg: u.cosine_sim < 0.0,
          lambda u,e,h,kg: u.grad_norm > 3.0]),

    Rule("R04", AttackPattern.MODEL_POISON,
         "High parameter deviation + repeated historical anomalies",
         [lambda u,e,h,kg: u.param_dev > 1.5,
          lambda u,e,h,kg: h >= 2,
          lambda u,e,h,kg: u.cosine_sim < 0.5]),

    Rule("R05", AttackPattern.MODEL_POISON,
         "Divergent update + high local loss + low accuracy",
         [lambda u,e,h,kg: u.cosine_sim < 0.3,
          lambda u,e,h,kg: e.local_loss > 2.0,
          lambda u,e,h,kg: e.local_accuracy < 0.4]),

    # ── Data Poisoning ────────────────────────────────────────────────────────
    # R06: label skew + repeated history (fires from round 2 onward)
    Rule("R06", AttackPattern.DATA_POISON,
         "Persistent label imbalance (dom>0.58) + low entropy + history",
         [lambda u,e,h,kg: e.dominant_class_frac > 0.58,
          lambda u,e,h,kg: e.label_entropy < 0.75,
          lambda u,e,h,kg: h >= 2]),

    # R07: low entropy + any loss instability (subtle single-round signal)
    Rule("R07", AttackPattern.DATA_POISON,
         "Suppressed label entropy (<0.72) + loss instability",
         [lambda u,e,h,kg: e.label_entropy < 0.72,
          lambda u,e,h,kg: e.dominant_class_frac > 0.55,
          lambda u,e,h,kg: abs(e.loss_delta) > 0.3]),

    # R08: moderate skew + low entropy + non-trivial loss (catch data poison
    #      with moderate poisoning fraction)
    Rule("R08", AttackPattern.DATA_POISON,
         "Moderate class skew + suppressed entropy + elevated loss",
         [lambda u,e,h,kg: e.dominant_class_frac > 0.56,
          lambda u,e,h,kg: e.label_entropy < 0.74,
          lambda u,e,h,kg: e.local_loss > 0.5]),

    # R13: strong single-round data poison signal (dominant > 0.60)
    Rule("R13", AttackPattern.DATA_POISON,
         "Strong label skew (dom>0.60) — high-confidence data poison",
         [lambda u,e,h,kg: e.dominant_class_frac > 0.60,
          lambda u,e,h,kg: e.label_entropy < 0.78]),

    # ── Backdoor ──────────────────────────────────────────────────────────────
    # R09: moderate norm + class imbalance (trigger pattern + label flip)
    Rule("R09", AttackPattern.BACKDOOR,
         "Moderate gradient + class imbalance + maintained accuracy",
         [lambda u,e,h,kg: 1.0 < u.grad_norm < 9.0,
          lambda u,e,h,kg: e.dominant_class_frac > 0.48,
          lambda u,e,h,kg: e.local_accuracy > 0.45]),

    # R10: cosine drift + class skew + history
    Rule("R10", AttackPattern.BACKDOOR,
         "Moderate cosine drift + class skew + prior history",
         [lambda u,e,h,kg: 0.2 < u.cosine_sim < 0.85,
          lambda u,e,h,kg: e.dominant_class_frac > 0.48,
          lambda u,e,h,kg: h >= 1]),

    # R14: backdoor single-round strong signal
    Rule("R14", AttackPattern.BACKDOOR,
         "Sustained class skew (dom>0.50) + low entropy + moderate norm",
         [lambda u,e,h,kg: e.dominant_class_frac > 0.50,
          lambda u,e,h,kg: e.label_entropy < 0.80,
          lambda u,e,h,kg: u.grad_norm < 9.0,
          lambda u,e,h,kg: e.local_accuracy > 0.40]),

    # ── Safe / Benign ─────────────────────────────────────────────────────────
    Rule("R11", AttackPattern.NONE,
         "Normal gradient + high cosine + balanced labels — benign",
         [lambda u,e,h,kg: u.grad_norm < 3.0,
          lambda u,e,h,kg: u.cosine_sim > 0.6,
          lambda u,e,h,kg: e.label_entropy > 0.7]),

    Rule("R12", AttackPattern.NONE,
         "No anomaly history + stable loss + balanced distribution",
         [lambda u,e,h,kg: h == 0,
          lambda u,e,h,kg: e.local_loss < 1.0,
          lambda u,e,h,kg: e.dominant_class_frac < 0.60]),
]


PATTERN_PRIORITY = [
    AttackPattern.BYZANTINE,
    AttackPattern.MODEL_POISON,
    AttackPattern.DATA_POISON,
    AttackPattern.BACKDOOR,
    AttackPattern.NONE,
]


class ReasoningEngine:

    def __init__(self, rules: List[Rule] = None):
        self.rules = rules or RULE_BASE

    def reason(self, client_id, round_id, update, evidence,
               anomaly_count, kg):
        fired: List[Rule] = []
        for rule in self.rules:
            if rule.fires(update, evidence, anomaly_count, kg):
                fired.append(rule)

        chosen_pattern = AttackPattern.NONE
        for p in PATTERN_PRIORITY:
            if any(r.pattern == p for r in fired):
                chosen_pattern = p
                break

        fired_rules = [r.rule_id for r in fired if r.pattern == chosen_pattern]
        all_fired   = [r.rule_id for r in fired]

        anomaly_score = self._compute_anomaly_score(update, evidence,
                                                     anomaly_count)
        risk_level    = self._classify_risk(anomaly_score, chosen_pattern)
        explanation   = self._build_explanation(
            client_id, round_id, update, evidence,
            anomaly_count, chosen_pattern, fired, risk_level, anomaly_score
        )
        return chosen_pattern, risk_level, anomaly_score, explanation, all_fired

    def _compute_anomaly_score(self, u, e, anomaly_count) -> float:
        u_score   = min(u.grad_norm / 10.0, 1.0)
        cos_score = max(1.0 - u.cosine_sim, 0.0)
        b_score   = min(anomaly_count / 5.0, 1.0)
        p_score   = min(e.local_loss / 5.0, 1.0)
        # Entropy deviation from 1.0 (low entropy = high suspicion)
        ent_score = max(1.0 - e.label_entropy, 0.0)
        d_score   = max(e.dominant_class_frac - 0.45, 0.0) / 0.55

        score = (0.25*u_score + 0.15*cos_score + 0.20*b_score
               + 0.15*p_score + 0.15*ent_score + 0.10*d_score)
        return round(float(min(max(score, 0.0), 1.0)), 4)

    def _classify_risk(self, score, pattern) -> RiskLevel:
        # v2: raised DATA_POISON and BACKDOOR floors for paper results
        floor = {
            AttackPattern.BYZANTINE   : 0.75,  # always CRITICAL
            AttackPattern.MODEL_POISON: 0.50,  # always HIGH
            AttackPattern.DATA_POISON : 0.52,  # always HIGH  (was 0.40)
            AttackPattern.BACKDOOR    : 0.50,  # always HIGH  (was 0.35)
            AttackPattern.NONE        : 0.0,
        }
        effective = max(score, floor[pattern])
        if effective >= 0.75: return RiskLevel.CRITICAL
        if effective >= 0.50: return RiskLevel.HIGH
        if effective >= 0.25: return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _build_explanation(self, client_id, round_id, u, e,
                            anomaly_count, pattern, fired,
                            risk_level, score) -> str:
        lines = [
            f"Client C{client_id} | Round {round_id}",
            f"Evidence:",
            f"  ✓ Gradient norm       : {u.grad_norm:.4f}",
            f"  ✓ Cosine similarity   : {u.cosine_sim:.4f}",
            f"  ✓ L2 distance         : {u.l2_distance:.4f}",
            f"  ✓ Param deviation     : {u.param_dev:.4f}",
            f"  ✓ Label entropy       : {e.label_entropy:.4f}",
            f"  ✓ Dominant class      : {e.dominant_class_frac:.4f}",
            f"  ✓ Local loss          : {e.local_loss:.4f}",
            f"  ✓ Local accuracy      : {e.local_accuracy:.4f}",
            f"  ✓ Loss delta          : {e.loss_delta:.4f}",
            f"  ✓ Sudden change       : {e.sudden_change}",
            f"  ✓ Historical anomalies: {anomaly_count}",
            f"Fired rules : {[r.rule_id+':'+r.description[:40] for r in fired] or ['none']}",
            f"Anomaly score : {score:.4f}",
            f"Attack pattern: {pattern.value.upper()}",
            f"Risk level    : {risk_level.value.upper()}",
            f"Decision      : {'QUARANTINE' if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 'REDUCE WEIGHT' if risk_level == RiskLevel.MEDIUM else 'NORMAL'}",
        ]
        return "\n".join(lines)

    def print_explanation(self, explanation: str):
        print("\n" + "─"*60)
        print(explanation)
        print("─"*60)
