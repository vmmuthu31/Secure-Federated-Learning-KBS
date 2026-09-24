"""
evidence_logger.py
Captures per-client, per-round evidence for the knowledge layer.

Evidence categories logged every FL round:
  A. Update-level  : gradient norm, cosine similarity to global, L2 distance
  B. Behavioural   : historical anomaly count, sudden change flag
  C. Data-level    : label entropy, dominant class fraction
  D. Performance   : local loss, local accuracy
  E. Knowledge     : composite anomaly score, risk pattern match flags
"""

import json
import os
import time
import numpy as np
from typing import Dict, Any, List, Optional


LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ── Evidence record for one client in one round ───────────────────────────────

class EvidenceRecord:
    def __init__(self, round_id: int, client_id: int):
        self.round_id   = round_id
        self.client_id  = client_id
        self.timestamp  = time.time()

        # A. Update-level
        self.grad_norm: float          = 0.0
        self.cosine_sim: float         = 1.0   # similarity to global model
        self.l2_distance: float        = 0.0
        self.param_deviation: float    = 0.0

        # B. Behavioural
        self.prev_anomaly_count: int   = 0
        self.sudden_change: bool       = False
        self.participation_rounds: int = 0

        # C. Data-level
        self.label_entropy: float      = 0.0   # higher = more uniform
        self.dominant_class_frac: float= 0.0   # fraction of majority class

        # D. Performance
        self.local_loss: float         = 0.0
        self.local_accuracy: float     = 0.0
        self.loss_delta: float         = 0.0   # change from previous round

        # E. Knowledge / composite
        self.anomaly_score: float      = 0.0   # 0–1
        self.risk_pattern: str         = "none"  # none | data_poison | model_poison | backdoor | byzantine
        self.trust_score: float        = 1.0   # 0–1
        self.risk_level: str           = "low"  # low | medium | high | critical

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ── Evidence computation helpers ──────────────────────────────────────────────

def compute_update_evidence(global_weights: np.ndarray,
                             local_weights: np.ndarray) -> Dict[str, float]:
    delta = local_weights - global_weights
    grad_norm    = float(np.linalg.norm(delta))
    l2_distance  = grad_norm
    gn = np.linalg.norm(global_weights)
    ln = np.linalg.norm(local_weights)
    if gn > 1e-9 and ln > 1e-9:
        cosine_sim = float(np.dot(global_weights.flatten(), local_weights.flatten())
                          / (gn * ln))
    else:
        cosine_sim = 1.0
    param_deviation = float(np.std(delta))
    return {
        "grad_norm":       grad_norm,
        "cosine_sim":      cosine_sim,
        "l2_distance":     l2_distance,
        "param_deviation": param_deviation,
    }


def compute_data_evidence(y: np.ndarray, num_classes: int = 10) -> Dict[str, float]:
    counts = np.bincount(y.astype(int), minlength=num_classes)
    probs = counts / counts.sum()
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log2(probs_nonzero)))
    max_entropy = np.log2(num_classes)
    label_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    dominant_class_frac = float(probs.max())
    return {
        "label_entropy":       label_entropy,
        "dominant_class_frac": dominant_class_frac,
    }


def compute_anomaly_score(grad_norm: float, cosine_sim: float,
                          prev_anomaly_count: int, local_loss: float,
                          dominant_class_frac: float,
                          norm_threshold: float = 5.0) -> float:
    """
    Composite anomaly score [0, 1].
    Weights match the trust formula in the paper roadmap:
      U (update quality)  + B (behaviour) + P (performance) + D (data)
    """
    u_score = min(grad_norm / norm_threshold, 1.0)          # high norm → suspicious
    b_score = min(prev_anomaly_count / 10.0, 1.0)           # history of anomalies
    p_score = min(local_loss / 5.0, 1.0)                    # unusually high loss
    d_score = max(dominant_class_frac - 0.5, 0.0) * 2.0    # class imbalance flag
    cos_score = max(1.0 - cosine_sim, 0.0)                  # low cosine → divergence

    score = 0.30 * u_score + 0.20 * b_score + 0.20 * p_score \
          + 0.15 * d_score + 0.15 * cos_score
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def classify_risk(anomaly_score: float) -> str:
    if anomaly_score >= 0.75:  return "critical"
    if anomaly_score >= 0.50:  return "high"
    if anomaly_score >= 0.25:  return "medium"
    return "low"


def classify_trust(anomaly_score: float) -> float:
    return round(float(1.0 - anomaly_score), 4)


def match_risk_pattern(grad_norm: float, cosine_sim: float,
                        dominant_class_frac: float,
                        prev_anomaly_count: int) -> str:
    """
    Simple rule-based pattern matching — mirrors the knowledge reasoning layer.
    Returns the most likely attack pattern or 'none'.
    """
    if dominant_class_frac > 0.85 and prev_anomaly_count > 2:
        return "data_poison"
    if cosine_sim < 0.0 and grad_norm > 3.0:
        return "model_poison"
    if grad_norm > 6.0 and cosine_sim < 0.3:
        return "byzantine"
    if dominant_class_frac > 0.7 and grad_norm > 2.0:
        return "backdoor"
    return "none"


# ── Evidence Logger ───────────────────────────────────────────────────────────

class EvidenceLogger:
    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self.log_path = os.path.join(LOG_DIR, f"{experiment_name}_evidence.jsonl")
        # In-memory history: {client_id: [list of past records]}
        self.history: Dict[int, List[Dict]] = {}

    def log(self, record: EvidenceRecord) -> None:
        cid = record.client_id
        self.history.setdefault(cid, []).append(record.to_dict())
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def get_history(self, client_id: int) -> List[Dict]:
        return self.history.get(client_id, [])

    def prev_anomaly_count(self, client_id: int, window: int = 5) -> int:
        hist = self.get_history(client_id)[-window:]
        return sum(1 for r in hist if r.get("anomaly_score", 0) > 0.5)

    def build_record(self, round_id: int, client_id: int,
                     global_weights: np.ndarray, local_weights: np.ndarray,
                     y_local: np.ndarray,
                     local_loss: float, local_accuracy: float,
                     num_classes: int = 10) -> EvidenceRecord:

        rec = EvidenceRecord(round_id, client_id)

        # A. Update
        upd = compute_update_evidence(global_weights.flatten(),
                                       local_weights.flatten())
        rec.grad_norm       = upd["grad_norm"]
        rec.cosine_sim      = upd["cosine_sim"]
        rec.l2_distance     = upd["l2_distance"]
        rec.param_deviation = upd["param_deviation"]

        # B. Behaviour
        rec.prev_anomaly_count   = self.prev_anomaly_count(client_id)
        rec.participation_rounds = len(self.get_history(client_id))
        hist = self.get_history(client_id)
        if hist:
            prev_norm = hist[-1].get("grad_norm", rec.grad_norm)
            rec.sudden_change = abs(rec.grad_norm - prev_norm) > 2.0 * prev_norm

        # C. Data
        dat = compute_data_evidence(y_local, num_classes)
        rec.label_entropy       = dat["label_entropy"]
        rec.dominant_class_frac = dat["dominant_class_frac"]

        # D. Performance
        rec.local_loss     = local_loss
        rec.local_accuracy = local_accuracy
        if hist:
            rec.loss_delta = local_loss - hist[-1].get("local_loss", local_loss)

        # E. Knowledge
        rec.anomaly_score = compute_anomaly_score(
            rec.grad_norm, rec.cosine_sim,
            rec.prev_anomaly_count, rec.local_loss,
            rec.dominant_class_frac
        )
        rec.risk_level   = classify_risk(rec.anomaly_score)
        rec.trust_score  = classify_trust(rec.anomaly_score)
        rec.risk_pattern = match_risk_pattern(
            rec.grad_norm, rec.cosine_sim,
            rec.dominant_class_frac, rec.prev_anomaly_count
        )
        return rec

    def summary(self, round_id: int) -> None:
        print(f"\n{'─'*72}")
        print(f"Round {round_id} — Evidence Summary")
        print(f"{'Client':>8} {'Trust':>7} {'Risk':>10} {'Pattern':>14} "
              f"{'GradNorm':>10} {'CosSim':>8} {'AnoScore':>10}")
        print(f"{'─'*72}")
        for cid, records in sorted(self.history.items()):
            r = records[-1]
            if r["round_id"] != round_id:
                continue
            print(f"{cid:>8} {r['trust_score']:>7.3f} {r['risk_level']:>10} "
                  f"{r['risk_pattern']:>14} {r['grad_norm']:>10.4f} "
                  f"{r['cosine_sim']:>8.4f} {r['anomaly_score']:>10.4f}")
        print(f"{'─'*72}")


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = EvidenceLogger("smoke_test")
    rng = np.random.default_rng(0)
    global_w = rng.normal(0, 1, 500)

    for rnd in range(3):
        for cid in range(5):
            # Simulate benign client
            noise_scale = 0.1 if cid < 4 else 3.5   # client 4 is malicious
            local_w = global_w + rng.normal(0, noise_scale, 500)
            y_local = rng.integers(0, 10, size=100)
            if cid == 4:                             # poisoned label distribution
                y_local[:] = rng.integers(0, 2, size=100)

            rec = logger.build_record(
                round_id=rnd, client_id=cid,
                global_weights=global_w, local_weights=local_w,
                y_local=y_local,
                local_loss=rng.uniform(0.1, 0.5) if cid < 4 else rng.uniform(2.0, 4.5),
                local_accuracy=rng.uniform(0.75, 0.95) if cid < 4 else rng.uniform(0.2, 0.4)
            )
            logger.log(rec)
        logger.summary(rnd)
    print(f"\nLog saved to: {logger.log_path}")
