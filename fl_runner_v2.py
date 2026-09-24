"""
fl_runner_v2.py — Full upgrade: Bayesian Trust + Temporal Reasoning + KALT v2

What's new vs v1:
  • BayesianTrustPool     : Beta-Binomial per-client trust posterior (τ_hat, σ)
  • TemporalTrackerPool   : per-client EMA, trend, variance, streak
  • Temporal rules R15-R18: persistent, escalating, evasion, coordinated attack
  • KALT v2               : uses (τ_hat, σ, trend) — not just deterministic τ
"""

import sys, os, json, time
import numpy as np

_HERE = os.path.dirname(__file__)
for sub in ("data", "knowledge", "attacks", "aggregation", "models"):
    sys.path.insert(0, os.path.join(_HERE, sub))

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
from bayesian_trust   import BayesianTrustPool
from temporal_tracker import TemporalTrackerPool, evaluate_temporal_rules

ARCH = {
    "mnist"  : (128, 64),
    "fashion": (128, 64),
    "cifar10": (128, 64),
}


def run_experiment(
    dataset_name      : str  = "mnist",
    partition_strategy: str  = "label_skew",
    num_clients       : int  = 10,
    num_rounds        : int  = 10,
    malicious_ids     : set  = None,
    attack_type       : AttackType  = AttackType.BYZANTINE,
    agg_strategy      : AggStrategy = AggStrategy.KA_AGGREGATE,
    lr                : float = 0.01,
    local_epochs      : int  = 3,
    use_kalt          : bool = True,
):
    if malicious_ids is None:
        malicious_ids = {8, 9}

    exp_name = (f"{dataset_name}_{partition_strategy}_"
                f"{attack_type.value}_{agg_strategy.value}"
                + ("_v2KALT" if use_kalt else "_v2noKALT"))

    print(f"\n{'='*70}")
    print(f"  Experiment : {exp_name}")
    print(f"  Malicious  : {sorted(malicious_ids)}  | Attack: {attack_type.value}")
    print(f"  Agg        : {agg_strategy.value}  | KALT-v2: {use_kalt}")
    print(f"{'='*70}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    loaders = {"mnist": load_mnist, "fashion": load_fashion_mnist,
               "cifar10": load_cifar10}
    (X_tr, y_tr), (X_te, y_te) = loaders[dataset_name]()
    n_features = int(np.prod(X_tr.shape[1:]))
    n_classes  = len(np.unique(y_tr))
    X_tr = X_tr.reshape(len(X_tr), n_features).astype(np.float32)
    X_te = X_te.reshape(len(X_te), n_features).astype(np.float32)

    h1, h2 = ARCH.get(dataset_name, (128, 64))
    print(f"  Arch: {n_features}→{h1}→{h2}→{n_classes}\n")

    # ── Partition ─────────────────────────────────────────────────────────────
    client_data = (partition_label_skew(X_tr, y_tr, num_clients, alpha=0.5)
                   if partition_strategy == "label_skew"
                   else partition_iid(X_tr, y_tr, num_clients))

    # ── Init global model ─────────────────────────────────────────────────────
    params_flat_g = params_to_flat(init_params(n_features, h1, h2, n_classes, seed=42))

    # ── Init KB + reasoning + aggregator ──────────────────────────────────────
    kg      = FLKnowledgeGraph(exp_name)
    eng     = ReasoningEngine()
    agg     = FLAggregator(strategy=agg_strategy)
    adapter = KALTAdapter(KALTConfig(base_lr=lr, base_epochs=local_epochs))

    for cid in range(num_clients):
        kg.add_client(ClientEntity(cid, is_malicious=(cid in malicious_ids)))

    # ── NEW v2: Bayesian trust pool + temporal tracker ────────────────────────
    bay_pool  = BayesianTrustPool(n_clients=num_clients, alpha0=2.0, beta0=1.0, decay=0.95)
    temp_pool = TemporalTrackerPool(n_clients=num_clients, window=5, ema_beta=0.7,
                                    high_anomaly_th=0.45)

    # ── Attack injectors ──────────────────────────────────────────────────────
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

        client_updates      = []
        detected_malicious  = []
        temporal_upgrades   = []

        # ── Per-client loop ───────────────────────────────────────────────────
        for cid in range(num_clients):
            X_c, y_c = client_data[cid]
            if cid in malicious_ids:
                X_c, y_c = injectors[cid].inject_data(X_c, y_c)

            # ── Fetch v2 KB signals for KALT ──────────────────────────────────
            if use_kalt and rnd > 0:
                # Bayesian posterior from previous round
                bay_stats  = bay_pool.stats(cid)
                tau_hat    = bay_stats["tau_hat"]
                sigma      = bay_stats["sigma"]
                # Temporal signals
                temp_sigs  = temp_pool.signals(cid)
                trend      = temp_sigs["trend"]
                # KB grad norm
                prev_us    = kg.updates.get(f"update_c{cid}_r{rnd-1}")
                kb_gnorm   = prev_us.grad_norm if prev_us else None
                # Fallback KB anomaly for KALT L2 eq
                prev_rs    = kg.risks.get(f"rs_c{cid}_r{rnd-1}")
                kb_anomaly = prev_rs.anomaly_score if prev_rs else 0.0
            else:
                tau_hat, sigma, trend, kb_gnorm, kb_anomaly = 0.667, 0.20, 0.0, None, 0.0

            # ── KALT v2 adaptation ────────────────────────────────────────────
            hp = adapter.adapt(
                tau_hat       = tau_hat,
                sigma         = sigma,
                anomaly_score = kb_anomaly,
                trend         = trend,
                kb_grad_norm  = kb_gnorm,
                round_idx     = rnd,
            )

            # ── Local training ────────────────────────────────────────────────
            params_flat_l, loss, acc = local_train_mlp(
                params_flat_g, X_c, y_c, n_classes,
                n_features, h1, h2,
                lr             = hp["lr"],
                epochs         = hp["epochs"],
                batch_size     = hp["batch_size"],
                l2_lambda      = hp["l2_lambda"],
                grad_clip_norm = hp["grad_clip"],
                seed           = cid * 100 + rnd,
            )

            if cid in malicious_ids:
                params_flat_l = injectors[cid].inject_update(
                    params_flat_g, params_flat_l)

            # ── Build KB entities ─────────────────────────────────────────────
            delta     = params_flat_l - params_flat_g
            grad_norm = float(np.linalg.norm(delta))
            gn = np.linalg.norm(params_flat_g)
            ln = np.linalg.norm(params_flat_l)
            cosine_sim = float(np.dot(params_flat_g, params_flat_l)
                               / (gn * ln + 1e-9))
            l2_dist   = grad_norm
            param_dev = float(np.std(delta))

            counts   = np.bincount(y_c.astype(int), minlength=n_classes)
            probs    = counts / counts.sum()
            probs_nz = probs[probs > 0]
            entropy  = float(-np.sum(probs_nz * np.log2(probs_nz + 1e-9)))
            max_ent  = np.log2(n_classes)
            label_ent = round(entropy / max_ent, 4) if max_ent > 0 else 0.0
            dom_frac  = float(probs.max())

            prev_us2   = kg.updates.get(f"update_c{cid}_r{rnd-1}", None)
            prev_norm  = prev_us2.grad_norm if prev_us2 else grad_norm
            sudden_chg = abs(grad_norm - prev_norm) > 2.0 * max(prev_norm, 1e-6)

            upd = UpdateEntity(f"update_c{cid}_r{rnd}", cid, rnd,
                               grad_norm, cosine_sim, l2_dist, param_dev)
            ev  = EvidenceEntity(f"ev_c{cid}_r{rnd}", cid, rnd,
                                 label_ent, dom_frac, loss, acc, 0.0, 0, sudden_chg)

            if rnd > 0:
                prev_ev = kg.evidences.get(f"ev_c{cid}_r{rnd-1}", None)
                ev.loss_delta = (loss - prev_ev.local_loss) if prev_ev else 0.0

            anomaly_cnt           = kg.get_client_anomaly_count(cid, window=5)
            ev.prev_anomaly_count = anomaly_cnt

            kg.add_update(upd)
            kg.add_evidence(ev)

            # ── Base reasoning (R01-R14) ──────────────────────────────────────
            pattern, risk_lvl, a_score, explanation, fired = eng.reason(
                cid, rnd, upd, ev, anomaly_cnt, kg)

            # ── NEW v2: Bayesian trust update ─────────────────────────────────
            bay_stats = bay_pool.update(cid, anomaly_score=a_score)
            tau_hat_new = bay_stats["tau_hat"]
            sigma_new   = bay_stats["sigma"]

            # ── NEW v2: Temporal tracker update ───────────────────────────────
            temp_sigs_new = temp_pool.update(cid, a_score)

            # ── NEW v2: Temporal rules R15-R18 ────────────────────────────────
            n_coord = temp_pool.coordination_signal(threshold=0.45)
            t_result = evaluate_temporal_rules(
                cid, temp_sigs_new, n_coord,
                base_attack = pattern.value if hasattr(pattern, "value") else str(pattern),
                base_level  = risk_lvl.value if hasattr(risk_lvl, "value") else str(risk_lvl),
            )

            # Upgrade classification if temporal rules fired
            final_risk_lvl = risk_lvl
            if t_result["upgraded"]:
                try:
                    final_risk_lvl = RiskLevel[t_result["level"]]
                except KeyError:
                    final_risk_lvl = risk_lvl
                temporal_upgrades.append((cid, t_result["rules_fired"]))

            # Use Bayesian τ_hat as the stored trust score
            trust_score = round(tau_hat_new, 4)
            trust_class = TrustScoreEntity.classify(trust_score)

            kg.add_trust(TrustScoreEntity(
                f"ts_c{cid}_r{rnd}", cid, rnd, trust_score, trust_class))
            kg.add_risk(RiskScoreEntity(
                f"rs_c{cid}_r{rnd}", cid, rnd, a_score,
                final_risk_lvl, pattern, explanation))

            if final_risk_lvl in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                detected_malicious.append(cid)

            client_updates.append(ClientUpdate(
                client_id     = cid,
                weights       = params_flat_l,
                n_samples     = len(X_c),
                trust_score   = trust_score,
                anomaly_score = a_score,
                risk_level    = final_risk_lvl.value,
            ))

        # ── Aggregate ─────────────────────────────────────────────────────────
        new_flat, agg_info = agg.aggregate(client_updates, rnd)
        params_flat_g = new_flat

        # ── Evaluate ──────────────────────────────────────────────────────────
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
        t_info  = f" [T-rules: {temporal_upgrades}]" if temporal_upgrades else ""
        print(f"R{rnd:02d} | acc={test_acc:.4f} | trust={avg_trust:.3f}"
              f" | det={sorted(detected_malicious)}"
              f" | TP={len(true_mal)} FP={len(false_pos)} FN={len(missed)}"
              f" | excl={agg_info.get('excluded',0)} | {elapsed:.1f}s{t_info}")

        results.append({
            "round"           : rnd,
            "test_accuracy"   : test_acc,
            "avg_trust"       : avg_trust,
            "detected"        : list(detected_malicious),
            "true_positives"  : len(true_mal),
            "false_positives" : len(false_pos),
            "false_negatives" : len(missed),
            "excluded"        : agg_info.get("excluded", 0),
            "temporal_upgrades": len(temporal_upgrades),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    log_dir  = os.path.join(_HERE, "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    res_path = os.path.join(log_dir, f"{exp_name}_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    kg.save()

    final_acc = results[-1]["test_accuracy"]
    avg_tp    = np.mean([r["true_positives"]  for r in results])
    avg_fp    = np.mean([r["false_positives"] for r in results])
    avg_fn    = np.mean([r["false_negatives"] for r in results])
    prec = avg_tp / (avg_tp + avg_fp + 1e-9)
    rec  = avg_tp / (avg_tp + avg_fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)

    print(f"\n{'─'*70}")
    print(f"  FINAL — {exp_name}")
    print(f"  Final acc={final_acc:.4f} | P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print(f"{'─'*70}")
    return results


if __name__ == "__main__":
    run_experiment()
