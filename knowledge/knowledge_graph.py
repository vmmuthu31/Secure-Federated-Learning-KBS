"""
knowledge_graph.py
Formal Knowledge Base / Ontology for FL Security.

Entity types  : Client, Update, Evidence, Anomaly, Attack, TrustScore, RiskLevel, Round
Relation types: generated, exhibits, matches_pattern, triggers, has_trust,
                has_risk, participated_in, connected_to, history_of
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

KB_DIR = os.path.join(os.path.dirname(__file__), "../../logs/kb")
os.makedirs(KB_DIR, exist_ok=True)


# ── Enumerations (typed ontology values) ─────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class AttackPattern(str, Enum):
    NONE         = "none"
    DATA_POISON  = "data_poison"
    MODEL_POISON = "model_poison"
    BACKDOOR     = "backdoor"
    BYZANTINE    = "byzantine"

class AnomalyType(str, Enum):
    NONE             = "none"
    HIGH_GRAD_NORM   = "high_grad_norm"
    LOW_COSINE_SIM   = "low_cosine_sim"
    LABEL_SKEW       = "label_skew"
    HIGH_LOSS        = "high_loss"
    SUDDEN_CHANGE    = "sudden_change"
    MULTI_ANOMALY    = "multi_anomaly"


# ── Entity definitions ────────────────────────────────────────────────────────

@dataclass
class ClientEntity:
    client_id       : int
    is_malicious    : bool  = False          # ground truth (for eval only)
    participation   : int   = 0              # rounds participated
    flagged_rounds  : List[int] = field(default_factory=list)
    quarantined     : bool  = False
    quarantine_since: Optional[int] = None

    def node_id(self) -> str:
        return f"client_{self.client_id}"


@dataclass
class UpdateEntity:
    update_id   : str        # e.g. "update_c3_r7"
    client_id   : int
    round_id    : int
    grad_norm   : float
    cosine_sim  : float
    l2_distance : float
    param_dev   : float

    def node_id(self) -> str:
        return self.update_id


@dataclass
class EvidenceEntity:
    evidence_id        : str   # e.g. "ev_c3_r7"
    client_id          : int
    round_id           : int
    label_entropy      : float
    dominant_class_frac: float
    local_loss         : float
    local_accuracy     : float
    loss_delta         : float
    prev_anomaly_count : int
    sudden_change      : bool

    def node_id(self) -> str:
        return self.evidence_id


@dataclass
class AnomalyEntity:
    anomaly_id   : str   # e.g. "anom_c3_r7"
    client_id    : int
    round_id     : int
    anomaly_type : AnomalyType
    severity     : float  # 0–1

    def node_id(self) -> str:
        return self.anomaly_id


@dataclass
class TrustScoreEntity:
    ts_id       : str
    client_id   : int
    round_id    : int
    trust_score : float   # 0–1
    trust_class : str     # highly_trusted / trusted / uncertain / suspicious / malicious

    def node_id(self) -> str:
        return self.ts_id

    @staticmethod
    def classify(score: float) -> str:
        if score >= 0.80: return "highly_trusted"
        if score >= 0.60: return "trusted"
        if score >= 0.40: return "uncertain"
        if score >= 0.20: return "suspicious"
        return "malicious"


@dataclass
class RiskScoreEntity:
    rs_id          : str
    client_id      : int
    round_id       : int
    anomaly_score  : float
    risk_level     : RiskLevel
    attack_pattern : AttackPattern
    explanation    : str   # human-readable reasoning trace

    def node_id(self) -> str:
        return self.rs_id


@dataclass
class RoundEntity:
    round_id          : int
    timestamp         : str
    num_clients       : int
    malicious_detected: int = 0
    avg_trust         : float = 0.0
    global_accuracy   : float = 0.0

    def node_id(self) -> str:
        return f"round_{self.round_id}"


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class FLKnowledgeGraph:
    """
    Lightweight in-memory knowledge graph.
    Nodes  : typed entities (Client, Update, Evidence, Anomaly, Trust, Risk, Round)
    Edges  : typed relations stored as adjacency list
    """

    def __init__(self, experiment: str = "default"):
        self.experiment = experiment
        # Node stores
        self.clients   : Dict[int,  ClientEntity]    = {}
        self.updates   : Dict[str,  UpdateEntity]    = {}
        self.evidences : Dict[str,  EvidenceEntity]  = {}
        self.anomalies : Dict[str,  AnomalyEntity]   = {}
        self.trusts    : Dict[str,  TrustScoreEntity] = {}
        self.risks     : Dict[str,  RiskScoreEntity]  = {}
        self.rounds    : Dict[int,  RoundEntity]      = {}
        # Edge store: {(src_node_id, relation, tgt_node_id)}
        self.edges     : List[Tuple[str,str,str]]    = []

    # ── Node adders ───────────────────────────────────────────────────────────

    def add_client(self, c: ClientEntity):
        self.clients[c.client_id] = c

    def add_round(self, r: RoundEntity):
        self.rounds[r.round_id] = r

    def add_update(self, u: UpdateEntity):
        self.updates[u.update_id] = u
        self._add_edge(f"client_{u.client_id}", "generated", u.node_id())
        self._add_edge(u.node_id(), "participated_in", f"round_{u.round_id}")

    def add_evidence(self, e: EvidenceEntity):
        self.evidences[e.evidence_id] = e
        self._add_edge(f"client_{e.client_id}", "has_evidence", e.node_id())

    def add_anomaly(self, a: AnomalyEntity):
        self.anomalies[a.anomaly_id] = a
        upd_id = f"update_c{a.client_id}_r{a.round_id}"
        self._add_edge(upd_id, "exhibits", a.node_id())

    def add_trust(self, t: TrustScoreEntity):
        self.trusts[t.ts_id] = t
        self._add_edge(f"client_{t.client_id}", "has_trust", t.node_id())

    def add_risk(self, r: RiskScoreEntity):
        self.risks[r.rs_id] = r
        self._add_edge(f"client_{r.client_id}", "has_risk", r.node_id())
        if r.attack_pattern != AttackPattern.NONE:
            self._add_edge(r.node_id(), "matches_pattern",
                           r.attack_pattern.value)

    def _add_edge(self, src: str, rel: str, tgt: str):
        self.edges.append((src, rel, tgt))

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_client_risk_history(self, client_id: int) -> List[RiskScoreEntity]:
        return sorted(
            [r for r in self.risks.values() if r.client_id == client_id],
            key=lambda x: x.round_id
        )

    def get_client_trust_history(self, client_id: int) -> List[TrustScoreEntity]:
        return sorted(
            [t for t in self.trusts.values() if t.client_id == client_id],
            key=lambda x: x.round_id
        )

    def get_flagged_clients(self, round_id: int) -> List[int]:
        return [
            r.client_id for r in self.risks.values()
            if r.round_id == round_id and r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]

    def get_client_anomaly_count(self, client_id: int, window: int = 5) -> int:
        hist = self.get_client_risk_history(client_id)[-window:]
        return sum(1 for r in hist if r.anomaly_score > 0.5)

    def get_avg_trust(self, round_id: int) -> float:
        scores = [t.trust_score for t in self.trusts.values()
                  if t.round_id == round_id]
        return round(sum(scores)/len(scores), 4) if scores else 0.0

    def client_summary(self, client_id: int) -> dict:
        hist = self.get_client_risk_history(client_id)
        trust_hist = self.get_client_trust_history(client_id)
        return {
            "client_id"      : client_id,
            "is_quarantined" : self.clients.get(client_id,
                               ClientEntity(client_id)).quarantined,
            "rounds_flagged" : len([h for h in hist
                                    if h.risk_level in (RiskLevel.HIGH,
                                                        RiskLevel.CRITICAL)]),
            "avg_trust"      : round(sum(t.trust_score for t in trust_hist)
                                     / max(len(trust_hist),1), 4),
            "latest_risk"    : hist[-1].risk_level if hist else "none",
            "latest_pattern" : hist[-1].attack_pattern if hist else "none",
            "latest_explanation": hist[-1].explanation if hist else "",
        }

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self):
        path = os.path.join(KB_DIR, f"{self.experiment}_kb.json")
        data = {
            "clients"  : {k: asdict(v) for k,v in self.clients.items()},
            "updates"  : {k: asdict(v) for k,v in self.updates.items()},
            "evidences": {k: asdict(v) for k,v in self.evidences.items()},
            "anomalies": {k: asdict(v) for k,v in self.anomalies.items()},
            "trusts"   : {k: asdict(v) for k,v in self.trusts.items()},
            "risks"    : {k: asdict(v) for k,v in self.risks.items()},
            "rounds"   : {k: asdict(v) for k,v in self.rounds.items()},
            "edges"    : self.edges,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def stats(self):
        print(f"\n{'='*55}")
        print(f"  Knowledge Graph — {self.experiment}")
        print(f"{'='*55}")
        print(f"  Nodes : Clients={len(self.clients)} | Rounds={len(self.rounds)}")
        print(f"          Updates={len(self.updates)} | Evidences={len(self.evidences)}")
        print(f"          Anomalies={len(self.anomalies)} | Trusts={len(self.trusts)}")
        print(f"          Risks={len(self.risks)}")
        print(f"  Edges : {len(self.edges)} typed relations")
        print(f"{'='*55}")
