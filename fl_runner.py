"""
fl_runner.py  (v2 — MLP + KALT)
Complete FL simulation with Knowledge-Aware Trust & Risk Framework.

What's new in v2:
  • 2-layer MLP backbone (pure NumPy) replaces logistic regression
  • KALT adapter feeds KB trust/risk back into local training hyperparams
    (adaptive LR, L2, gradient clipping, epochs) — the core novelty claim

Wires: dataset → partition → attack injection → KB → reasoning →
       trust/risk → KALT adaptation → local_train_mlp → risk-adaptive agg
"""

import sys, os, json, time
import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "data"))
sys.path.insert(0, os.path.join(_HERE, "knowledge"))
sys.path.insert(0, os.path.join(_HERE, "attacks"))
sys.path.insert(0, os.path.join(_HERE, "aggregation"))
sys.path.insert(0, os.path.join(_HERE, "models"))

from dataset_loader   import load_mnist, load_fashion_mnist, load_cifar10
from partitioner      import partition_label_skew, partition_iid
from knowledge_graph  import (FLKnowledgeGraph, ClientEntity, UpdateEntity,
                               EvidenceEntity, AnomalyEntity, TrustScoreEntity,
                               RiskScoreEntity, RoundEntity,
                               AttackPattern, RiskLevel, AnomalyType)
from reasoning_engine import ReasoningEngine
from attack_injector  import AttackInjector, AttackType
from aggregator       import FLAggregator, AggStrategy, ClientUpdate
from mlp              import (init_params, local_train_mlp,
                               params_to_flat, flat_to_params, accuracy_mlp)
from ka_trainer       import KALTAdapter, KALTConfig


# ── Architecture config per dataset ──────────────────────────────────────────
# MNIST / Fashion-MNIST : 784-in → 256 → 128 → 10-out
# CIFAR-10              : 3072-in → 512 → 256 → 10-out
ARCH = {
    "mnist"   : (256, 128),
    "fashion" : (256, 128),
    "cifar10" : (512, 256),
}


# ── Main FL experiment ────────────────────────────────────────────────────────

def run_experiment(
    dataset_name      : str   = "mnist",
    partition_strategy: str   = "label_skew",
    num_clients       : int   = 10,
    num_rounds        : int   = 10,
    malicious_ids     : set   = None,
    attack_type       : AttackType   = AttackType.BYZANTINE,
    agg_strategy      : AggStrategy  = AggStrategy.KA_AGGREGATE,
    lr                : float = 0.01,
    local_epochs      : int   = 3,
    use_kalt          : bool  = True,    # NEW: toggle KALT on/off for ablation
):
    if malicious_ids is None:
        malicious_ids = {8, 9}

    exp_name = (f"{dataset_name}_{partition_strategy}_"
                f"{attack_type.value}_{agg_strategy.value}"
                + ("_KALT" if use_kalt else "_noKALT"))

    print(f"\n{'='*70}")
    print(f"  Experiment : {exp_name}")
    print(f"  Clients    : {num_clients}  | Malicious: {sorted(malicious_ids)}")
    print(f"  Rounds     : {num_rounds}  | Attack: {attack_type.value}")
    print(f"  Aggregation: {agg_strategy.value}  | KALT: {use_kalt}")
    print(f"{'='*70}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    loaders = {"mnist": load_mnist, "fashion": load_fashion_mnist,
               "cifar10": load_cifar10}
    (X_tr, y_tr), (X_te, y_te) = loaders[dataset_name]()
    n_features = int(np.prod(X_tr.shape[1:]))
    n_classes  = len(np.unique(y_tr))
    X_tr = X_tr.reshape(len(X_tr), n_features).astype(np.float32)
    X_te = X_te.reshape(len(X_te), n_features).astype(np.float32)

    h1, h2 = ARCH.get(dataset_name, (256, 128))
    print(f"  Dataset    : {dataset_name} | n_features={n_features} "
          f"n_classes={n_classes} | arch={n_features}→{h1}→{h2}→{n_classes}\n")

    # ── Partition ─────────────────────────────────────────────────────────────
    if partition_strategy == "label_skew":
        client_data = partition_label_skew(X_tr, y_tr, num_clients, alpha=0.5)
    else:
        client_data = partition_iid(X_tr, y_tr, num_clients)

    # ── Init global MLP (flat param vector) ───────────────────────────────────
    init_p       = init_params(n_features, h1, h2, n_classes, seed=42)
    params_flat_g = params_to_flat(init_p)          # 1-D vector shared via FL

    # ── Init KB, reasoning engine, aggregator, KALT adapter ──────────────────
    kg      = FLKnowledgeGraph(exp_name)
    eng     = ReasoningEngine()
    agg     = FLAggregator(strategy=agg_strategy)
    adapter = KALTAdapter(KALTConfig(
        base_lr     = lr,
        base_epochs = local_epochs,
    ))

    for cid in range(num_clients):
        kg.add_client(ClientEntity(cid, is_malicious=(cid in malicious_ids)))

    # ── Attack injectors per malicious client ─────────────────────────────────
    injectors = {
        cid: AttackInjector(attack_type, poison_fraction=0.6,
                            target_label=0, scale_factor=-8.0, seed=cid)
        for cid in malicious_ids
    }

    results = []

    for rnd in range(num_rounds):
        t0        = time.time()
        round_ent = RoundEntity(rnd, str(time.time()), num_clients)
        kg.add_round(round_ent)

        client_updates: list[ClientUpdate] = []
        detected_malicious = []

        for cid in range(num_clients):
            X_c, y_c = client_data[cid]

            # ── Inject data-level attack ──────────────────────────────────────
            if cid in malicious_ids:
                X_c, y_c = injectors[cid].inject_data(X_c, y_c)

            # ── KALT: fetch KB scores from previous round ─────────────────────
            if use_kalt and rnd > 0:
                prev_ts = kg.trusts.get(f"ts_c{cid}_r{rnd-1}")
                prev_rs = kg.risks.get(f"rs_c{cid}_r{rnd-1}")
                prev_us = kg.updates.get(f"update_c{cid}_r{rnd-1}")
                kb_trust   = prev_ts.trust_score           if prev_ts else 1.0
                kb_anomaly = prev_rs.anomaly_score      if prev_rs else 0.0
                kb_gnorm   = prev_us.grad_norm        if prev_us else None
            else:
                # Round 0: full trust until proven otherwise
                kb_trust, kb_anomaly, kb_gnorm = 1.0, 0.0, None

            # ── Adapt hyperparams via KALT ────────────────────────────────────
            hp = adapter.adapt(kb_trust, kb_anomaly, kb_gnorm, rnd)
            if rnd > 0 and cid < 3:   # light logging for first 3 clients
                print(f"  {adapter.adapt_log(cid, rnd, kb_trust, kb_anomaly, kb_gnorm, hp)}")

            # ── Local MLP training ────────────────────────────────────────────
            params_flat_l, loss, acc = local_train_mlp(
                params_flat_g, X_c, y_c, n_classes,
                n_features, h1, h2,
                lr           = hp["lr"],
                epochs       = hp["epochs"],
                batch_size   = hp["batch_size"],
                l2_lambda    = hp["l2_lambda"],
                grad_clip_norm = hp["grad_clip"],
                seed         = cid * 100 + rnd,
            )

            # ── Inject update-level attack ────────────────────────────────────
            if cid in malicious_ids:
                params_flat_l = injectors[cid].inject_update(
                    params_flat_g, params_flat_l)

            # ── Build KB evidence entities ────────────────────────────────────
            delta      = params_flat_l - params_flat_g
            grad_norm  = float(np.linalg.norm(delta))
            gn = np.linalg.norm(params_flat_g)
            ln = np.linalg.norm(params_flat_l)
            cosine_sim = float(np.dot(params_flat_g, params_flat_l)
                               / (gn * ln + 1e-9))
            l2_dist    = grad_norm
            param_dev  = float(np.std(delta))

            counts    = np.bincount(y_c.astype(int), minlength=n_classes)
            probs     = counts / counts.sum()
            probs_nz  = probs[probs > 0]
            entropy   = float(-np.sum(probs_nz * np.log2(probs_nz + 1e-9)))
            max_ent   = np.log2(n_classes)
            label_ent = round(entropy / max_ent, 4) if max_ent > 0 else 0.0
            dom_frac  = float(probs.max())

            prev_us2   = kg.updates.get(f"update_c{cid}_r{rnd-1}", None)
            prev_norm  = prev_us2.grad_norm if prev_us2 else grad_norm
            sudden_chg = abs(grad_norm - prev_norm) > 2.0 * max(prev_norm, 1e-6)

            upd_id = f"update_c{cid}_r{rnd}"
            ev_id  = f"ev_c{cid}_r{rnd}"

            upd = UpdateEntity(upd_id, cid, rnd, grad_norm, cosine_sim,
                               l2_dist, param_dev)
            ev  = EvidenceEntity(ev_id, cid, rnd, label_ent, dom_frac,
                                 loss, acc, 0.0, 0, sudden_chg)

            if rnd > 0:
                prev_ev = kg.evidences.get(f"ev_c{cid}_r{rnd-1}", None)
                ev.loss_delta = (loss - prev_ev.local_loss) if prev_ev else 0.0

            anomaly_cnt          = kg.get_client_anomaly_count(cid, window=5)
            ev.prev_anomaly_count = anomaly_cnt

            kg.add_update(upd)
            kg.add_evidence(ev)

            # ── Reasoning engine ──────────────────────────────────────────────
            pattern, risk_lvl, a_score, explanation, fired = eng.reason(
                cid, rnd, upd, ev, anomaly_cnt, kg)
            trust_score = round(1.0 - a_score, 4)
            trust_class = TrustScoreEntity.classify(trust_score)

            kg.add_trust(TrustScoreEntity(
                f"ts_c{cid}_r{rnd}", cid, rnd, trust_score, trust_class))
            kg.add_risk(RiskScoreEntity(
                f"rs_c{cid}_r{rnd}", cid, rnd, a_score,
                risk_lvl, pattern, explanation))

            if risk_lvl in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                detected_malicious.append(cid)

            client_updates.append(ClientUpdate(
                client_id     = cid,
                weights       = params_flat_l,
                n_samples     = len(X_c),
                trust_score   = trust_score,
                anomaly_score = a_score,
                risk_level    = risk_lvl.value,
            ))

        # ── Aggregate ─────────────────────────────────────────────────────────
        new_flat, agg_info = agg.aggregate(client_updates, rnd)
        params_flat_g = new_flat     # new global params (flat MLP vector)

        # ── Evaluate on test set ──────────────────────────────────────────────
        g_params  = flat_to_params(params_flat_g, n_features, h1, h2, n_classes)
        test_acc  = accuracy_mlp(g_params, X_te, y_te, n_classes)
        avg_trust = kg.get_avg_trust(rnd)

        true_mal  = set(detected_malicious) & malicious_ids
        false_pos = set(detected_malicious) - malicious_ids
        missed    = malicious_ids - set(detected_malicious)

        round_ent.malicious_detected = len(true_mal)
        round_ent.avg_trust          = avg_trust
        round_ent.global_accuracy    = test_acc

        elapsed = time.time() - t0
        print(f"R{rnd:02d} | acc={test_acc:.4f} | trust={avg_trust:.3f}"
              f" | det={sorted(detected_malicious)}"
              f" | TP={len(true_mal)} FP={len(false_pos)} FN={len(missed)}"
              f" | excl={agg_info.get('excluded',0)} | {elapsed:.1f}s")

        results.append({
            "round"          : rnd,
            "test_accuracy"  : test_acc,
            "avg_trust"      : avg_trust,
            "detected"       : list(detected_malicious),
            "true_positives" : len(true_mal),
            "false_positives": len(false_pos),
            "false_negatives": len(missed),
            "excluded"       : agg_info.get("excluded", 0),
        })

    # ── Save results ──────────────────────────────────────────────────────────
    kb_path  = kg.save()
    log_dir  = os.path.join(_HERE, "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    res_path = os.path.join(log_dir, f"{exp_name}_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    kg.stats()
    print(f"\nKB saved     : {kb_path}")
    print(f"Results saved: {res_path}")

    final_acc = results[-1]["test_accuracy"]
    avg_tp  = np.mean([r["true_positives"]   for r in results])
    avg_fp  = np.mean([r["false_positives"]  for r in results])
    avg_fn  = np.mean([r["false_negatives"]  for r in results])
    prec = avg_tp / (avg_tp + avg_fp + 1e-9)
    rec  = avg_tp / (avg_tp + avg_fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)

    print(f"\n{'─'*70}")
    print(f"  FINAL — {exp_name}")
    print(f"{'─'*70}")
    print(f"  Final test accuracy     : {final_acc:.4f}")
    print(f"  Avg detection precision : {prec:.4f}")
    print(f"  Avg detection recall    : {rec:.4f}")
    print(f"  Avg detection F1        : {f1:.4f}")
    print(f"{'─'*70}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiment(
        dataset_name       = "mnist",
        partition_strategy = "label_skew",
        num_clients        = 10,
        num_rounds         = 10,
        malicious_ids      = {8, 9},
        attack_type        = AttackType.BYZANTINE,
        agg_strategy       = AggStrategy.KA_AGGREGATE,
        lr                 = 0.01,
        local_epochs       = 3,
        use_kalt           = True,
    )
