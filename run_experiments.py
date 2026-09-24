"""
run_experiments.py
Master script: ablation + multi-attack + multi-dataset experiments.
Saves per-run JSON + paper-ready CSV tables to logs/ablation/.
"""

import os, sys, json, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "knowledge"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "attacks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aggregation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))

from fl_runner    import run_experiment
from attack_injector import AttackType
from aggregator      import AggStrategy
from evaluator    import FLEvaluator, compare_results, _write_table

# ─── Config ────────────────────────────────────────────────────────────────────
NUM_CLIENTS   = 10
MALICIOUS_IDS = {8, 9}
NUM_ROUNDS    = 10
LOG_DIR       = os.path.join(os.path.dirname(__file__), "../logs/ablation")
os.makedirs(LOG_DIR, exist_ok=True)

all_results = []

def run_one(label, dataset, partition, attack_enum, agg_enum,
            attack_str, agg_str):
    exp_name = f"{label}_{dataset}_{attack_str}"
    print(f"\n{'▶'*3}  {exp_name}")
    t0    = time.time()
    rounds = run_experiment(
        dataset_name      = dataset,
        partition_strategy= partition,
        num_clients       = NUM_CLIENTS,
        num_rounds        = NUM_ROUNDS,
        malicious_ids     = MALICIOUS_IDS,
        attack_type       = attack_enum,
        agg_strategy      = agg_enum,
        lr                = 0.01,
        local_epochs      = 3,
    )
    elapsed = time.time() - t0

    ev = FLEvaluator(exp_name, dataset, partition, attack_str, agg_str,
                     NUM_CLIENTS, NUM_ROUNDS, MALICIOUS_IDS)
    for rd in rounds:
        per_round_elapsed = elapsed / NUM_ROUNDS
        ev.record_round(
            round_idx     = rd["round"],
            test_acc      = rd["test_accuracy"],
            avg_trust     = rd["avg_trust"],
            anomaly_scores= {},          # reconstructed inside record_round
            detected      = rd["detected"],
            malicious_ids = MALICIOUS_IDS,
            excluded      = rd["excluded"],
            elapsed       = per_round_elapsed,
        )
    result = ev.finalize()
    result.save(LOG_DIR)
    all_results.append(result)
    ev.print_summary()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# A. ABLATION STUDY  (MNIST, label_skew, Byzantine)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*65)
print("  A. ABLATION STUDY — MNIST / label_skew / Byzantine")
print("█"*65)

ABLATION = [
    # label                    use_kb  agg_enum                  agg_str
    ("A1_Baseline_FedAvg",     False,  AggStrategy.FEDAVG,       "fedavg"),
    ("A2_KB_NoTrust_NoRisk",   True,   AggStrategy.FEDAVG,       "fedavg"),
    ("A3_KB_TrustOnly",        True,   AggStrategy.TRUST_WEIGHTED,"trust_weighted"),
    ("A4_KB_RiskOnly",         True,   AggStrategy.RISK_PENALTY, "risk_penalty"),
    ("A5_KB_Full_KA",          True,   AggStrategy.KA_AGGREGATE, "ka_aggregate"),
]

ablation_results = []
for label, use_kb, agg_enum, agg_str in ABLATION:
    r = run_one(label, "mnist", "label_skew",
                AttackType.BYZANTINE, agg_enum,
                "byzantine", agg_str)
    ablation_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# B. MULTI-ATTACK STUDY  (MNIST, label_skew, KA_Full)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*65)
print("  B. MULTI-ATTACK STUDY — MNIST / KA_Full")
print("█"*65)

ATTACKS = [
    ("byzantine",    AttackType.BYZANTINE),
    ("model_poison", AttackType.MODEL_POISON),
    ("data_poison",  AttackType.DATA_POISON),
    ("backdoor",     AttackType.BACKDOOR),
]

attack_results = []
for atk_str, atk_enum in ATTACKS:
    r = run_one("KA_Full", "mnist", "label_skew",
                atk_enum, AggStrategy.KA_AGGREGATE,
                atk_str, "ka_aggregate")
    attack_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# C. MULTI-DATASET STUDY  (Byzantine, KA_Full, label_skew)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*65)
print("  C. MULTI-DATASET STUDY — Byzantine / KA_Full")
print("█"*65)

DATASETS = [
    ("mnist",        "label_skew"),
    ("fashion","label_skew"),
    ("cifar10",      "label_skew"),
]

dataset_results = []
for ds, part in DATASETS:
    r = run_one("KA_Full", ds, part,
                AttackType.BYZANTINE, AggStrategy.KA_AGGREGATE,
                "byzantine", "ka_aggregate")
    dataset_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# D. PAPER TABLES
# ══════════════════════════════════════════════════════════════════════════════

# Table 1 — Ablation
_write_table(
    os.path.join(LOG_DIR, "table1_ablation.csv"),
    ["Variant","FinalAcc","BestAcc","Precision","Recall","F1",
     "AvgTrust","FirstDet","Time(s)"],
    [[r.exp_name, f"{r.final_accuracy:.4f}", f"{r.best_accuracy:.4f}",
      f"{r.avg_precision:.4f}", f"{r.avg_recall:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round),
      f"{r.total_time:.1f}"]
     for r in ablation_results],
    title="TABLE 1: Ablation Study (MNIST, Byzantine)",
)

# Table 2 — Multi-attack
_write_table(
    os.path.join(LOG_DIR, "table2_multi_attack.csv"),
    ["Attack","FinalAcc","F1","AvgTrust","FirstDet"],
    [[r.attack_type, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round)]
     for r in attack_results],
    title="TABLE 2: Multi-Attack (MNIST, KA_Full)",
)

# Table 3 — Multi-dataset
_write_table(
    os.path.join(LOG_DIR, "table3_multi_dataset.csv"),
    ["Dataset","FinalAcc","F1","AvgTrust","FirstDet","Time(s)"],
    [[r.dataset, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round),
      f"{r.total_time:.1f}"]
     for r in dataset_results],
    title="TABLE 3: Multi-Dataset (Byzantine, KA_Full)",
)

# Overall comparison
compare_results(all_results)

print(f"\n✓ All tables saved to: {LOG_DIR}")
