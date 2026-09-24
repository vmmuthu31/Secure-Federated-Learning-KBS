"""Re-run multi-attack study with tuned reasoning engine v2."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
for d in ["data","knowledge","attacks","aggregation","evaluation"]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), d))

from fl_runner       import run_experiment
from attack_injector import AttackType
from aggregator      import AggStrategy
from evaluator       import FLEvaluator, _write_table

NUM_CLIENTS   = 10
MALICIOUS_IDS = {8, 9}
NUM_ROUNDS    = 10
LOG_DIR       = os.path.join(os.path.dirname(__file__), "../logs/ablation")
os.makedirs(LOG_DIR, exist_ok=True)

ATTACKS = [
    ("byzantine",    AttackType.BYZANTINE),
    ("model_poison", AttackType.MODEL_POISON),
    ("data_poison",  AttackType.DATA_POISON),
    ("backdoor",     AttackType.BACKDOOR),
]

results = []
for atk_str, atk_enum in ATTACKS:
    exp_name = f"v2_KA_Full_mnist_{atk_str}"
    print(f"\n▶▶▶  {exp_name}")
    t0 = time.time()
    rounds = run_experiment(
        dataset_name="mnist", partition_strategy="label_skew",
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
        malicious_ids=MALICIOUS_IDS,
        attack_type=atk_enum, agg_strategy=AggStrategy.KA_AGGREGATE,
        lr=0.01, local_epochs=3,
    )
    elapsed = time.time() - t0
    ev = FLEvaluator(exp_name, "mnist", "label_skew", atk_str,
                     "ka_aggregate", NUM_CLIENTS, NUM_ROUNDS, MALICIOUS_IDS)
    for rd in rounds:
        ev.record_round(rd["round"], rd["test_accuracy"], rd["avg_trust"],
                        {}, rd["detected"], MALICIOUS_IDS,
                        rd["excluded"], elapsed/NUM_ROUNDS)
    r = ev.finalize()
    r.save(LOG_DIR)
    results.append(r)
    ev.print_summary()

_write_table(
    os.path.join(LOG_DIR, "table2_multi_attack_v2.csv"),
    ["Attack","FinalAcc","F1","AvgTrust","FirstDet","Excluded(avg)"],
    [[r.attack_type,
      f"{r.final_accuracy:.4f}", f"{r.avg_f1:.4f}",
      f"{r.avg_trust:.4f}", str(r.first_detection_round),
      f"{sum(rd.excluded for rd in r.rounds)/len(r.rounds):.1f}"]
     for r in results],
    title="TABLE 2 v2: Multi-Attack (MNIST, KA_Full, tuned rules)",
)
