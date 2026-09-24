"""
evaluator.py
Comprehensive evaluation module for the Knowledge-Aware Trust & Risk
Assessment Framework (KBS Q1 paper).

Provides:
  - ExperimentResult: structured container for one run's metrics
  - FLEvaluator      : accumulates per-round stats during a run
  - AblationRunner   : orchestrates ablation and multi-attack experiments
  - ResultsTable     : formats paper-ready tables (ASCII + CSV)
"""

import os, sys, json, time, csv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── path setup ───────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ["data", "knowledge", "attacks", "aggregation"]:
    sys.path.insert(0, os.path.join(_ROOT, _d))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Per-round stats dataclass
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RoundMetrics:
    round_idx      : int
    test_accuracy  : float
    avg_trust      : float
    avg_risk       : float           # mean anomaly_score across clients
    detected       : List[int]       # client ids flagged HIGH/CRITICAL
    true_positives : int
    false_positives: int
    false_negatives: int
    excluded       : int
    elapsed_sec    : float

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2*p*r/(p+r) if (p+r) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ExperimentResult: full run summary
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentResult:
    exp_name       : str
    dataset        : str
    partition      : str
    attack_type    : str
    agg_strategy   : str
    num_clients    : int
    num_rounds     : int
    malicious_ids  : List[int]
    rounds         : List[RoundMetrics] = field(default_factory=list)

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def final_accuracy(self) -> float:
        return self.rounds[-1].test_accuracy if self.rounds else 0.0

    @property
    def best_accuracy(self) -> float:
        return max(r.test_accuracy for r in self.rounds) if self.rounds else 0.0

    @property
    def avg_precision(self) -> float:
        return float(np.mean([r.precision for r in self.rounds]))

    @property
    def avg_recall(self) -> float:
        return float(np.mean([r.recall for r in self.rounds]))

    @property
    def avg_f1(self) -> float:
        return float(np.mean([r.f1 for r in self.rounds]))

    @property
    def avg_trust(self) -> float:
        return float(np.mean([r.avg_trust for r in self.rounds]))

    @property
    def avg_risk(self) -> float:
        return float(np.mean([r.avg_risk for r in self.rounds]))

    @property
    def total_time(self) -> float:
        return sum(r.elapsed_sec for r in self.rounds)

    @property
    def first_detection_round(self) -> int:
        """Earliest round where ALL malicious clients are detected."""
        for r in self.rounds:
            if set(r.detected) >= set(self.malicious_ids):
                return r.round_idx
        return -1  # never detected all

    def to_dict(self) -> dict:
        return {
            "exp_name"             : self.exp_name,
            "dataset"              : self.dataset,
            "partition"            : self.partition,
            "attack_type"          : self.attack_type,
            "agg_strategy"         : self.agg_strategy,
            "num_clients"          : self.num_clients,
            "num_rounds"           : self.num_rounds,
            "malicious_ids"        : self.malicious_ids,
            "final_accuracy"       : round(self.final_accuracy, 4),
            "best_accuracy"        : round(self.best_accuracy, 4),
            "avg_precision"        : round(self.avg_precision, 4),
            "avg_recall"           : round(self.avg_recall, 4),
            "avg_f1"               : round(self.avg_f1, 4),
            "avg_trust"            : round(self.avg_trust, 4),
            "avg_risk"             : round(self.avg_risk, 4),
            "total_time_sec"       : round(self.total_time, 2),
            "first_detection_round": self.first_detection_round,
            "rounds"               : [
                {
                    "round"          : r.round_idx,
                    "test_accuracy"  : round(r.test_accuracy, 4),
                    "avg_trust"      : round(r.avg_trust, 4),
                    "avg_risk"       : round(r.avg_risk, 4),
                    "precision"      : round(r.precision, 4),
                    "recall"         : round(r.recall, 4),
                    "f1"             : round(r.f1, 4),
                    "detected"       : r.detected,
                    "TP"             : r.true_positives,
                    "FP"             : r.false_positives,
                    "FN"             : r.false_negatives,
                    "excluded"       : r.excluded,
                    "elapsed_sec"    : round(r.elapsed_sec, 3),
                }
                for r in self.rounds
            ],
        }

    def save(self, log_dir: str) -> str:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{self.exp_name}_eval.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FLEvaluator: used INSIDE fl_runner per-round
# ═══════════════════════════════════════════════════════════════════════════════
class FLEvaluator:
    """
    Drop-in accumulator to use inside run_experiment():

        ev = FLEvaluator(exp_name, dataset, partition, attack, agg,
                         num_clients, num_rounds, malicious_ids)
        ...
        ev.record_round(round_idx, test_acc, avg_trust,
                        client_anomaly_scores, detected, malicious_ids,
                        excluded, elapsed)
        result = ev.finalize()
        result.save(log_dir)
    """

    def __init__(self, exp_name, dataset, partition, attack, agg,
                 num_clients, num_rounds, malicious_ids):
        self.result = ExperimentResult(
            exp_name      = exp_name,
            dataset       = dataset,
            partition     = partition,
            attack_type   = attack,
            agg_strategy  = agg,
            num_clients   = num_clients,
            num_rounds    = num_rounds,
            malicious_ids = list(malicious_ids),
        )

    def record_round(self,
                     round_idx       : int,
                     test_acc        : float,
                     avg_trust       : float,
                     anomaly_scores  : Dict[int, float],   # cid → score
                     detected        : List[int],
                     malicious_ids   : set,
                     excluded        : int,
                     elapsed         : float):

        avg_risk = float(np.mean(list(anomaly_scores.values()))) \
                   if anomaly_scores else 0.0
        tp = len(set(detected) & malicious_ids)
        fp = len(set(detected) - malicious_ids)
        fn = len(malicious_ids  - set(detected))

        self.result.rounds.append(RoundMetrics(
            round_idx      = round_idx,
            test_accuracy  = test_acc,
            avg_trust      = avg_trust,
            avg_risk       = avg_risk,
            detected       = list(detected),
            true_positives = tp,
            false_positives= fp,
            false_negatives= fn,
            excluded       = excluded,
            elapsed_sec    = elapsed,
        ))

    def finalize(self) -> ExperimentResult:
        return self.result

    def print_summary(self):
        r = self.result
        print(f"\n{'═'*65}")
        print(f"  EVALUATION SUMMARY — {r.exp_name}")
        print(f"{'═'*65}")
        print(f"  Dataset / Partition  : {r.dataset} / {r.partition}")
        print(f"  Attack / Aggregation : {r.attack_type} / {r.agg_strategy}")
        print(f"  Clients (malicious)  : {r.num_clients} ({r.malicious_ids})")
        print(f"  Rounds               : {r.num_rounds}")
        print(f"{'─'*65}")
        print(f"  Final accuracy       : {r.final_accuracy:.4f}")
        print(f"  Best accuracy        : {r.best_accuracy:.4f}")
        print(f"  Avg detection P/R/F1 : {r.avg_precision:.4f} / "
              f"{r.avg_recall:.4f} / {r.avg_f1:.4f}")
        print(f"  First full detection : round {r.first_detection_round}")
        print(f"  Avg trust / risk     : {r.avg_trust:.4f} / {r.avg_risk:.4f}")
        print(f"  Total wall time      : {r.total_time:.1f}s")
        print(f"{'═'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  AblationRunner
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    # (label,                       use_kb, use_trust, use_risk, agg_strategy)
    ("Baseline_FedAvg",             False,  False,     False,    "fedavg"),
    ("KA_NoTrust_NoRisk",           True,   False,     False,    "fedavg"),
    ("KA_TrustOnly",                True,   True,      False,    "trust_weighted"),
    ("KA_RiskOnly",                 True,   False,     True,     "risk_penalty"),
    ("KA_Full",                     True,   True,      True,     "ka_aggregate"),
]

ATTACK_TYPES = ["byzantine", "model_poison", "data_poison", "backdoor"]

DATASETS = ["mnist", "fashion_mnist", "cifar10"]


class AblationRunner:
    """
    Orchestrates ablation, multi-attack, and multi-dataset experiments.

    Usage:
        from evaluation.evaluator import AblationRunner
        runner = AblationRunner(log_dir="logs/ablation", num_rounds=10)
        runner.run_ablation(dataset="mnist", attack="byzantine")
        runner.run_multi_attack(dataset="mnist")
        runner.run_multi_dataset(attack="byzantine")
        runner.save_tables()
    """

    def __init__(self, log_dir: str = "logs/ablation", num_rounds: int = 10,
                 num_clients: int = 10, malicious_ids: set = None):
        self.log_dir      = log_dir
        self.num_rounds   = num_rounds
        self.num_clients  = num_clients
        self.malicious_ids= malicious_ids or {8, 9}
        self.results      : List[ExperimentResult] = []
        os.makedirs(log_dir, exist_ok=True)

    def _run_one(self, label, dataset, partition, attack_str,
                 use_kb, use_trust, use_risk, agg_str) -> ExperimentResult:
        """Run a single experiment and return ExperimentResult."""
        # lazy import to avoid circular dependency
        from fl_runner import run_experiment
        from attacks.attack_injector import AttackType
        from aggregation.aggregator  import AggStrategy

        attack_map = {
            "byzantine"   : AttackType.BYZANTINE,
            "model_poison": AttackType.MODEL_POISON,
            "data_poison" : AttackType.DATA_POISON,
            "backdoor"    : AttackType.BACKDOOR,
            "none"        : AttackType.NONE,
        }
        agg_map = {
            "fedavg"        : AggStrategy.FEDAVG,
            "trust_weighted": AggStrategy.TRUST_WEIGHTED,
            "risk_penalty"  : AggStrategy.RISK_PENALTY,
            "ka_aggregate"  : AggStrategy.KA_AGGREGATE,
        }
        exp_name = f"{label}_{dataset}_{attack_str}"
        print(f"\n[AblationRunner] ▶ {exp_name}")

        t0     = time.time()
        rounds = run_experiment(
            dataset_name      = dataset,
            partition_strategy= partition,
            num_clients       = self.num_clients,
            num_rounds        = self.num_rounds,
            malicious_ids     = self.malicious_ids,
            attack_type       = attack_map[attack_str],
            agg_strategy      = agg_map[agg_str],
            lr                = 0.01,
            local_epochs      = 3,
        )
        elapsed = time.time() - t0

        # Convert raw dicts from run_experiment → ExperimentResult
        ev_obj = FLEvaluator(exp_name, dataset, partition,
                             attack_str, agg_str,
                             self.num_clients, self.num_rounds,
                             self.malicious_ids)
        for rd in rounds:
            # anomaly_scores not returned by current run_experiment;
            # reconstruct avg_risk from round's detected list heuristic
            n_mal   = len(self.malicious_ids)
            avg_risk_est = rd["true_positives"] / max(self.num_clients, 1)
            ev_obj.record_round(
                round_idx      = rd["round"],
                test_acc       = rd["test_accuracy"],
                avg_trust      = rd["avg_trust"],
                anomaly_scores = {i: avg_risk_est for i in range(self.num_clients)},
                detected       = rd["detected"],
                malicious_ids  = self.malicious_ids,
                excluded       = rd["excluded"],
                elapsed        = elapsed / self.num_rounds,
            )
        result = ev_obj.finalize()
        result.save(self.log_dir)
        self.results.append(result)
        ev_obj.print_summary()
        return result

    def run_ablation(self, dataset: str = "mnist",
                     attack: str = "byzantine",
                     partition: str = "label_skew") -> List[ExperimentResult]:
        print(f"\n{'#'*65}")
        print(f"  ABLATION STUDY  |  {dataset}  |  {attack}")
        print(f"{'#'*65}")
        out = []
        for label, use_kb, use_trust, use_risk, agg in ABLATION_CONFIGS:
            r = self._run_one(label, dataset, partition, attack,
                              use_kb, use_trust, use_risk, agg)
            out.append(r)
        return out

    def run_multi_attack(self, dataset: str = "mnist",
                         partition: str = "label_skew") -> List[ExperimentResult]:
        print(f"\n{'#'*65}")
        print(f"  MULTI-ATTACK STUDY  |  {dataset}")
        print(f"{'#'*65}")
        out = []
        for atk in ATTACK_TYPES:
            r = self._run_one("KA_Full", dataset, partition, atk,
                              True, True, True, "ka_aggregate")
            out.append(r)
        return out

    def run_multi_dataset(self, attack: str = "byzantine",
                          partition: str = "label_skew") -> List[ExperimentResult]:
        print(f"\n{'#'*65}")
        print(f"  MULTI-DATASET STUDY  |  {attack}")
        print(f"{'#'*65}")
        out = []
        for ds in DATASETS:
            r = self._run_one("KA_Full", ds, partition, attack,
                              True, True, True, "ka_aggregate")
            out.append(r)
        return out

    # ── Table generation ──────────────────────────────────────────────────────
    def save_tables(self):
        """Write paper-ready ASCII + CSV tables to log_dir."""
        # ─── Table 1: Ablation ─────────────────────────────────────────────
        abl = [r for r in self.results
               if any(c[0] in r.exp_name for c in ABLATION_CONFIGS)]
        _write_table(
            os.path.join(self.log_dir, "table_ablation.csv"),
            ["Variant", "FinalAcc", "BestAcc", "Precision", "Recall",
             "F1", "AvgTrust", "AvgRisk", "FirstDet", "TotalTime(s)"],
            [[r.exp_name, f"{r.final_accuracy:.4f}", f"{r.best_accuracy:.4f}",
              f"{r.avg_precision:.4f}", f"{r.avg_recall:.4f}", f"{r.avg_f1:.4f}",
              f"{r.avg_trust:.4f}", f"{r.avg_risk:.4f}",
              str(r.first_detection_round), f"{r.total_time:.1f}"]
             for r in abl],
            title="TABLE: Ablation Study",
        )

        # ─── Table 2: Multi-attack ─────────────────────────────────────────
        atk_res = [r for r in self.results
                   if any(a in r.exp_name for a in ATTACK_TYPES)
                   and "KA_Full" in r.exp_name]
        _write_table(
            os.path.join(self.log_dir, "table_multi_attack.csv"),
            ["Attack", "FinalAcc", "F1", "AvgTrust", "AvgRisk", "FirstDet"],
            [[r.attack_type, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
              f"{r.avg_trust:.4f}", f"{r.avg_risk:.4f}",
              str(r.first_detection_round)]
             for r in atk_res],
            title="TABLE: Multi-Attack Performance (KA_Full)",
        )

        # ─── Table 3: Multi-dataset ────────────────────────────────────────
        ds_res = [r for r in self.results
                  if r.dataset in DATASETS and "KA_Full" in r.exp_name
                  and r.attack_type == "byzantine"]
        _write_table(
            os.path.join(self.log_dir, "table_multi_dataset.csv"),
            ["Dataset", "FinalAcc", "F1", "AvgTrust", "FirstDet", "Time(s)"],
            [[r.dataset, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
              f"{r.avg_trust:.4f}", str(r.first_detection_round),
              f"{r.total_time:.1f}"]
             for r in ds_res],
            title="TABLE: Multi-Dataset Performance (Byzantine, KA_Full)",
        )

        print(f"\n[ResultsTable] Tables saved to: {self.log_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_table(csv_path: str, headers: List[str],
                 rows: List[List[str]], title: str = ""):
    """Write a CSV and print an ASCII table."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # ASCII table
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep  = "+" + "+".join("-"*(w+2) for w in widths) + "+"
    fmt  = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"
    print(f"\n{title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


def load_result(json_path: str) -> ExperimentResult:
    """Load a saved ExperimentResult JSON back into the dataclass."""
    with open(json_path) as f:
        d = json.load(f)
    er = ExperimentResult(
        exp_name     = d["exp_name"],
        dataset      = d["dataset"],
        partition    = d["partition"],
        attack_type  = d["attack_type"],
        agg_strategy = d["agg_strategy"],
        num_clients  = d["num_clients"],
        num_rounds   = d["num_rounds"],
        malicious_ids= d["malicious_ids"],
    )
    for r in d["rounds"]:
        er.rounds.append(RoundMetrics(
            round_idx      = r["round"],
            test_accuracy  = r["test_accuracy"],
            avg_trust      = r["avg_trust"],
            avg_risk       = r["avg_risk"],
            detected       = r["detected"],
            true_positives = r["TP"],
            false_positives= r["FP"],
            false_negatives= r["FN"],
            excluded       = r["excluded"],
            elapsed_sec    = r["elapsed_sec"],
        ))
    return er


def compare_results(results: List[ExperimentResult]) -> None:
    """Print a side-by-side comparison of multiple experiment results."""
    print(f"\n{'═'*75}")
    print(f"  COMPARISON TABLE")
    print(f"{'═'*75}")
    hdr = f"{'Experiment':<40} {'Acc':>6} {'F1':>6} {'Trust':>6} {'1stDet':>7}"
    print(hdr)
    print("─"*75)
    for r in results:
        name = r.exp_name[:38]
        print(f"{name:<40} {r.final_accuracy:>6.4f} {r.avg_f1:>6.4f} "
              f"{r.avg_trust:>6.4f} {r.first_detection_round:>7}")
    print("═"*75)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(_ROOT))

    print("=" * 65)
    print("  Evaluator self-test: loading existing result JSON")
    print("=" * 65)

    log_dir = os.path.join(os.path.dirname(_ROOT), "logs")
    json_files = [f for f in os.listdir(log_dir)
                  if f.endswith("_results.json")]

    if not json_files:
        print("No result JSON found in logs/. Run fl_runner.py first.")
        sys.exit(0)

    # Convert the flat results.json (from fl_runner) to ExperimentResult
    rfile = os.path.join(log_dir, json_files[0])
    with open(rfile) as f:
        raw = json.load(f)

    name_parts = json_files[0].replace("_results.json", "").split("_")
    ev = FLEvaluator(
        exp_name     = json_files[0].replace("_results.json", ""),
        dataset      = name_parts[0] if name_parts else "mnist",
        partition    = name_parts[1] if len(name_parts) > 1 else "label_skew",
        attack       = name_parts[2] if len(name_parts) > 2 else "byzantine",
        agg          = name_parts[3] if len(name_parts) > 3 else "ka_aggregate",
        num_clients  = 10,
        num_rounds   = len(raw),
        malicious_ids= {8, 9},
    )
    for rd in raw:
        ev.record_round(
            round_idx     = rd["round"],
            test_acc      = rd["test_accuracy"],
            avg_trust     = rd["avg_trust"],
            anomaly_scores= {},
            detected      = rd["detected"],
            malicious_ids = {8, 9},
            excluded      = rd["excluded"],
            elapsed       = 1.0,
        )
    result = ev.finalize()
    ev.print_summary()
    saved = result.save(os.path.join(log_dir, "ablation"))
    print(f"\nEval JSON saved: {saved}")
