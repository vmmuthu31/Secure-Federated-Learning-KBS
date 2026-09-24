"""
run_final_tables.py
Full ablation study with v2 reasoning engine + compile all paper tables.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
for d in ["data","knowledge","attacks","aggregation","evaluation"]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), d))

from fl_runner       import run_experiment
from attack_injector import AttackType
from aggregator      import AggStrategy
from evaluator       import FLEvaluator, _write_table, load_result

NUM_CLIENTS   = 10
MALICIOUS_IDS = {8, 9}
NUM_ROUNDS    = 10
LOG_DIR       = os.path.join(os.path.dirname(__file__), "../logs/ablation")
os.makedirs(LOG_DIR, exist_ok=True)

def run_one(label, dataset, partition, atk_enum, agg_enum, atk_str, agg_str):
    exp_name = label
    print(f"\n▶ {exp_name}")
    t0 = time.time()
    rounds = run_experiment(
        dataset_name=dataset, partition_strategy=partition,
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        malicious_ids=MALICIOUS_IDS,
        attack_type=atk_enum, agg_strategy=agg_enum,
        lr=0.01, local_epochs=3,
    )
    elapsed = time.time() - t0
    ev = FLEvaluator(exp_name, dataset, partition, atk_str, agg_str,
                     NUM_CLIENTS, NUM_ROUNDS, MALICIOUS_IDS)
    for rd in rounds:
        ev.record_round(rd["round"], rd["test_accuracy"], rd["avg_trust"],
                        {}, rd["detected"], MALICIOUS_IDS,
                        rd["excluded"], elapsed/NUM_ROUNDS)
    r = ev.finalize()
    r.save(LOG_DIR)
    print(f"   → acc={r.final_accuracy:.4f}  F1={r.avg_f1:.4f}  "
          f"trust={r.avg_trust:.4f}  firstDet={r.first_detection_round}")
    return r

# ── TABLE 1: Ablation (MNIST, Byzantine) ─────────────────────────────────────
print("\n" + "█"*65)
print("  A. ABLATION — MNIST / label_skew / Byzantine (v2 rules)")
print("█"*65)

ABLATION = [
    ("A1_Baseline_FedAvg",      AttackType.BYZANTINE, AggStrategy.FEDAVG,         "byzantine","fedavg"),
    ("A2_KB_NoTrust_NoRisk",    AttackType.BYZANTINE, AggStrategy.FEDAVG,         "byzantine","fedavg"),
    ("A3_KB_TrustOnly",         AttackType.BYZANTINE, AggStrategy.TRUST_WEIGHTED, "byzantine","trust_weighted"),
    ("A4_KB_RiskOnly",          AttackType.BYZANTINE, AggStrategy.RISK_PENALTY,   "byzantine","risk_penalty"),
    ("A5_KB_Full_KA",           AttackType.BYZANTINE, AggStrategy.KA_AGGREGATE,   "byzantine","ka_aggregate"),
]
abl = [run_one(n,"mnist","label_skew",a,g,as_,gs_) for n,a,g,as_,gs_ in ABLATION]

# ── TABLE 2: Multi-Attack (v2 already in LOG_DIR) ────────────────────────────
v2_files = [f for f in os.listdir(LOG_DIR) if f.startswith("v2_KA_Full_mnist_") and f.endswith("_eval.json")]
atk_res  = sorted([load_result(os.path.join(LOG_DIR,f)) for f in v2_files],
                   key=lambda r: r.attack_type)

# ── TABLE 3: Multi-Dataset (already in LOG_DIR) ───────────────────────────────
ds_files = ["KA_Full_mnist_byzantine_eval.json",
            "KA_Full_fashion_byzantine_eval.json",
            "KA_Full_cifar10_byzantine_eval.json"]
ds_res = [load_result(os.path.join(LOG_DIR,f)) for f in ds_files if os.path.exists(os.path.join(LOG_DIR,f))]

# ── Print + save tables ───────────────────────────────────────────────────────
_write_table(
    os.path.join(LOG_DIR, "FINAL_table1_ablation.csv"),
    ["Variant","FinalAcc","Precision","Recall","F1","AvgTrust","FirstDet"],
    [[r.exp_name, f"{r.final_accuracy:.4f}", f"{r.avg_precision:.4f}",
      f"{r.avg_recall:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round)]
     for r in abl],
    title="TABLE 1 (FINAL): Ablation Study — MNIST / label_skew / Byzantine",
)

_write_table(
    os.path.join(LOG_DIR, "FINAL_table2_multi_attack.csv"),
    ["Attack","FinalAcc","Precision","Recall","F1","AvgTrust","FirstDet"],
    [[r.attack_type, f"{r.final_accuracy:.4f}", f"{r.avg_precision:.4f}",
      f"{r.avg_recall:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round)]
     for r in atk_res],
    title="TABLE 2 (FINAL): Multi-Attack — MNIST / KA_Full (v2 rules)",
)

if ds_res:
    _write_table(
        os.path.join(LOG_DIR, "FINAL_table3_multi_dataset.csv"),
        ["Dataset","FinalAcc","F1","AvgTrust","FirstDet","Time(s)"],
        [[r.dataset, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
          f"{r.avg_trust:.4f}", str(r.first_detection_round),
          f"{r.total_time:.1f}"]
         for r in ds_res],
        title="TABLE 3 (FINAL): Multi-Dataset — Byzantine / KA_Full",
    )

print(f"\n✓ All FINAL tables in: {LOG_DIR}")
