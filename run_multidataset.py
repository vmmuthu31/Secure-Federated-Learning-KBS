"""Run only the multi-dataset study + re-generate tables from saved evals."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
for d in ["data","knowledge","attacks","aggregation","evaluation"]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), d))

from fl_runner        import run_experiment
from attack_injector  import AttackType
from aggregator       import AggStrategy
from evaluator        import FLEvaluator, compare_results, _write_table, load_result

NUM_CLIENTS   = 10
MALICIOUS_IDS = {8, 9}
NUM_ROUNDS    = 10
LOG_DIR       = os.path.join(os.path.dirname(__file__), "../logs/ablation")
os.makedirs(LOG_DIR, exist_ok=True)

all_results = []

def run_one(label, dataset, partition, attack_enum, agg_enum, attack_str, agg_str):
    exp_name = f"{label}_{dataset}_{attack_str}"
    print(f"\n▶▶▶  {exp_name}")
    t0 = time.time()
    rounds = run_experiment(
        dataset_name=dataset, partition_strategy=partition,
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        malicious_ids=MALICIOUS_IDS, attack_type=attack_enum,
        agg_strategy=agg_enum, lr=0.01, local_epochs=3,
    )
    elapsed = time.time() - t0
    ev = FLEvaluator(exp_name, dataset, partition, attack_str, agg_str,
                     NUM_CLIENTS, NUM_ROUNDS, MALICIOUS_IDS)
    for rd in rounds:
        ev.record_round(rd["round"], rd["test_accuracy"], rd["avg_trust"],
                        {}, rd["detected"], MALICIOUS_IDS,
                        rd["excluded"], elapsed/NUM_ROUNDS)
    result = ev.finalize()
    result.save(LOG_DIR)
    all_results.append(result)
    ev.print_summary()
    return result

# C. MULTI-DATASET
print("\n" + "█"*65)
print("  C. MULTI-DATASET STUDY — Byzantine / KA_Full")
print("█"*65)
DATASETS = [("mnist","label_skew"), ("fashion","label_skew"), ("cifar10","label_skew")]
ds_results = []
for ds, part in DATASETS:
    r = run_one("KA_Full", ds, part,
                AttackType.BYZANTINE, AggStrategy.KA_AGGREGATE,
                "byzantine", "ka_aggregate")
    ds_results.append(r)

# Reload ablation results saved earlier
abl_dir = LOG_DIR
abl_files = sorted([f for f in os.listdir(abl_dir) if "_eval.json" in f and "A" in f])
ablation_results = [load_result(os.path.join(abl_dir,f)) for f in abl_files]

atk_files = sorted([f for f in os.listdir(abl_dir) if "_eval.json" in f and "KA_Full_mnist" in f])
attack_results = [load_result(os.path.join(abl_dir,f)) for f in atk_files]

# Re-print tables
print("\n=== Loaded ablation results:", [r.exp_name for r in ablation_results])
print("=== Loaded attack results:  ", [r.exp_name for r in attack_results])
print("=== Dataset results:        ", [r.exp_name for r in ds_results])

_write_table(
    os.path.join(LOG_DIR, "table1_ablation.csv"),
    ["Variant","FinalAcc","BestAcc","Precision","Recall","F1","AvgTrust","FirstDet","Time(s)"],
    [[r.exp_name, f"{r.final_accuracy:.4f}", f"{r.best_accuracy:.4f}",
      f"{r.avg_precision:.4f}", f"{r.avg_recall:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round), f"{r.total_time:.1f}"]
     for r in ablation_results],
    title="TABLE 1: Ablation Study (MNIST, Byzantine)",
)
_write_table(
    os.path.join(LOG_DIR, "table2_multi_attack.csv"),
    ["Attack","FinalAcc","F1","AvgTrust","FirstDet"],
    [[r.attack_type, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round)]
     for r in attack_results],
    title="TABLE 2: Multi-Attack (MNIST, KA_Full)",
)
_write_table(
    os.path.join(LOG_DIR, "table3_multi_dataset.csv"),
    ["Dataset","FinalAcc","F1","AvgTrust","FirstDet","Time(s)"],
    [[r.dataset, f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round), f"{r.total_time:.1f}"]
     for r in ds_results],
    title="TABLE 3: Multi-Dataset (Byzantine, KA_Full)",
)
compare_results(ablation_results + attack_results + ds_results)
print(f"\n✓ Tables saved to: {LOG_DIR}")
