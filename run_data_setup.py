"""
run_data_setup.py
Entry point — loads datasets, creates partitions, runs evidence logger smoke test.
Usage:  python run_data_setup.py [mnist|fashion|cifar10]  [iid|label_skew|quantity_skew|class_shards]
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from dataset_loader  import load_mnist, load_fashion_mnist, load_cifar10
from partitioner     import (partition_iid, partition_label_skew,
                              partition_quantity_skew, partition_class_shards,
                              partition_stats)
from evidence_logger import EvidenceLogger

NUM_CLIENTS   = 10
NUM_ROUNDS    = 3
MALICIOUS_IDS = {8, 9}   # clients 8 & 9 will behave maliciously in simulation

# ── Dataset selection ─────────────────────────────────────────────────────────
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
strategy     = sys.argv[2] if len(sys.argv) > 2 else "label_skew"

loaders = {"mnist": load_mnist, "fashion": load_fashion_mnist, "cifar10": load_cifar10}
if dataset_name not in loaders:
    print(f"Unknown dataset '{dataset_name}'. Choose: mnist | fashion | cifar10"); sys.exit(1)

print(f"\n{'='*60}")
print(f"  Dataset  : {dataset_name}")
print(f"  Strategy : {strategy}")
print(f"  Clients  : {NUM_CLIENTS}  (malicious: {sorted(MALICIOUS_IDS)})")
print(f"  Rounds   : {NUM_ROUNDS}")
print(f"{'='*60}\n")

(X_train, y_train), (X_test, y_test) = loaders[dataset_name]()

# ── Partition ─────────────────────────────────────────────────────────────────
strategy_fns = {
    "iid":           lambda: partition_iid(X_train, y_train, NUM_CLIENTS),
    "label_skew":    lambda: partition_label_skew(X_train, y_train, NUM_CLIENTS, alpha=0.5),
    "quantity_skew": lambda: partition_quantity_skew(X_train, y_train, NUM_CLIENTS),
    "class_shards":  lambda: partition_class_shards(X_train, y_train, NUM_CLIENTS),
}
if strategy not in strategy_fns:
    print(f"Unknown strategy '{strategy}'. Choose: iid | label_skew | quantity_skew | class_shards")
    sys.exit(1)

client_data = strategy_fns[strategy]()
partition_stats(client_data)

# ── Simulated FL rounds with evidence logging ─────────────────────────────────
rng    = np.random.default_rng(42)
logger = EvidenceLogger(f"{dataset_name}_{strategy}")

# Simulate flat global model (placeholder — replace with real model weights later)
feature_dim  = X_train.shape[1] if X_train.ndim == 2 else int(np.prod(X_train.shape[1:]))
num_classes  = len(np.unique(y_train))
global_w     = rng.normal(0, 0.1, feature_dim * num_classes)

print(f"\nRunning {NUM_ROUNDS} simulated FL rounds …")
for rnd in range(NUM_ROUNDS):
    for cid in range(NUM_CLIENTS):
        X_c, y_c = client_data[cid]
        is_malicious = cid in MALICIOUS_IDS

        # Simulated local update
        if is_malicious:
            noise = rng.normal(0, 4.0, global_w.shape)          # large deviation
            local_loss     = rng.uniform(2.0, 5.0)
            local_accuracy = rng.uniform(0.1, 0.35)
            # poison label distribution: push to 1–2 dominant classes
            y_poisoned = np.where(rng.random(len(y_c)) < 0.85,
                                  rng.integers(0, 2, len(y_c)), y_c)
        else:
            noise = rng.normal(0, 0.1, global_w.shape)
            local_loss     = rng.uniform(0.05, 0.4)
            local_accuracy = rng.uniform(0.80, 0.97)
            y_poisoned = y_c

        local_w = global_w + noise

        rec = logger.build_record(
            round_id       = rnd,
            client_id      = cid,
            global_weights = global_w,
            local_weights  = local_w,
            y_local        = y_poisoned,
            local_loss     = local_loss,
            local_accuracy = local_accuracy,
            num_classes    = num_classes,
        )
        logger.log(rec)

    logger.summary(rnd)

    # Naive aggregation (benign only, trust-weighted placeholder)
    updates = []
    weights_agg = []
    for cid in range(NUM_CLIENTS):
        hist = logger.get_history(cid)
        if not hist: continue
        last = hist[-1]
        trust = last["trust_score"]
        X_c, _ = client_data[cid]
        updates.append(trust)
        weights_agg.append(trust * len(X_c))

    total = sum(weights_agg)
    agg_weight = sum(weights_agg) / (total if total > 0 else 1)
    print(f"  → Round {rnd} avg trust-weighted contribution factor: {agg_weight:.4f}")

print(f"\nEvidence log saved to: {logger.log_path}")
print("Setup complete. Next step: plug real model weights into build_record().")
