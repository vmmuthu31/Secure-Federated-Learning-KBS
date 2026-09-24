"""
partitioner.py
Creates IID and non-IID client data partitions for FL experiments.

Supported strategies:
  - iid           : uniform random split
  - label_skew    : Dirichlet(alpha) over class labels  (lower alpha = more skewed)
  - quantity_skew : unequal sample counts per client
  - class_shards  : each client gets exactly `shards_per_client` classes
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


ClientData = Dict[int, Tuple[np.ndarray, np.ndarray]]   # {client_id: (X, y)}


# ── IID ──────────────────────────────────────────────────────────────────────

def partition_iid(X: np.ndarray, y: np.ndarray,
                  num_clients: int, seed: int = 42) -> ClientData:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, num_clients)
    return {i: (X[s], y[s]) for i, s in enumerate(splits)}


# ── Label skew (Dirichlet) ────────────────────────────────────────────────────

def partition_label_skew(X: np.ndarray, y: np.ndarray,
                         num_clients: int, alpha: float = 0.5,
                         seed: int = 42) -> ClientData:
    """
    alpha → 0   : extreme skew (each client ~1 class)
    alpha = 0.5 : moderate skew  (default for FL papers)
    alpha → inf : approaches IID
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices: Dict[int, List[int]] = defaultdict(list)

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # scale proportions to actual counts
        proportions = (proportions * len(cls_idx)).astype(int)
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()   # fix rounding
        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(cls_idx[start:start + count].tolist())
            start += count

    return {
        cid: (X[np.array(idxs)], y[np.array(idxs)])
        for cid, idxs in client_indices.items()
    }


# ── Quantity skew ─────────────────────────────────────────────────────────────

def partition_quantity_skew(X: np.ndarray, y: np.ndarray,
                             num_clients: int, min_frac: float = 0.05,
                             seed: int = 42) -> ClientData:
    """
    Each client gets a random fraction of the data; fractions are drawn from
    a Dirichlet distribution so they sum to 1, then clipped to [min_frac, ...].
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    fractions = rng.dirichlet(np.ones(num_clients))
    fractions = np.clip(fractions, min_frac / num_clients, None)
    fractions /= fractions.sum()
    counts = (fractions * len(X)).astype(int)
    counts[-1] = len(X) - counts[:-1].sum()

    client_data: ClientData = {}
    start = 0
    for cid, cnt in enumerate(counts):
        idx = indices[start:start + cnt]
        client_data[cid] = (X[idx], y[idx])
        start += cnt
    return client_data


# ── Class shards (original FedAvg non-IID) ───────────────────────────────────

def partition_class_shards(X: np.ndarray, y: np.ndarray,
                            num_clients: int, shards_per_client: int = 2,
                            seed: int = 42) -> ClientData:
    """
    Sort data by label, split into (num_clients * shards_per_client) shards,
    assign shards_per_client shards to each client (original McMahan et al. approach).
    """
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(y)
    num_shards = num_clients * shards_per_client
    shard_size = len(X) // num_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size]
              for i in range(num_shards)]
    shard_ids = rng.permutation(num_shards)
    client_data: ClientData = {}
    for cid in range(num_clients):
        assigned = np.concatenate(
            [shards[shard_ids[cid * shards_per_client + s]]
             for s in range(shards_per_client)]
        )
        client_data[cid] = (X[assigned], y[assigned])
    return client_data


# ── Stats helper ─────────────────────────────────────────────────────────────

def partition_stats(client_data: ClientData) -> None:
    print(f"{'Client':>8} {'Samples':>8} {'Classes':>8}  Label distribution")
    print("-" * 60)
    for cid, (X, y) in sorted(client_data.items()):
        classes, counts = np.unique(y, return_counts=True)
        dist = {int(c): int(n) for c, n in zip(classes, counts)}
        print(f"{cid:>8} {len(X):>8} {len(classes):>8}  {dist}")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_loader import load_mnist

    (X, y), _ = load_mnist()
    NUM_CLIENTS = 10

    for strategy, kwargs in [
        ("iid",           {}),
        ("label_skew",    {"alpha": 0.5}),
        ("label_skew",    {"alpha": 0.1}),
        ("quantity_skew", {}),
        ("class_shards",  {"shards_per_client": 2}),
    ]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}  {kwargs}")
        print('='*60)
        fn = {
            "iid":           partition_iid,
            "label_skew":    partition_label_skew,
            "quantity_skew": partition_quantity_skew,
            "class_shards":  partition_class_shards,
        }[strategy]
        data = fn(X, y, num_clients=NUM_CLIENTS, **kwargs)
        partition_stats(data)
