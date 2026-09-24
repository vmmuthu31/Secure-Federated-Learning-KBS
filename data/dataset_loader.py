"""
dataset_loader.py
Loads MNIST, Fashion-MNIST, and CIFAR-10 from local .gz / .tar.gz files.
No torchvision required at load time — pure numpy/pickle.
"""

import os
import gzip
import pickle
import struct
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "../../datasets")


# ── MNIST / Fashion-MNIST (IDX format) ──────────────────────────────────────

def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(normalize: bool = True):
    root = os.path.join(BASE, "mnist")
    X_train = _read_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    X_test  = _read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    y_test  = _read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    if normalize:
        X_train, X_test = X_train / 255.0, X_test / 255.0
    print(f"[MNIST] train={X_train.shape}, test={X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist(normalize: bool = True):
    root = os.path.join(BASE, "fashion_mnist")
    X_train = _read_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    X_test  = _read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    y_test  = _read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    if normalize:
        X_train, X_test = X_train / 255.0, X_test / 255.0
    print(f"[FashionMNIST] train={X_train.shape}, test={X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


# ── CIFAR-10 (pickle batches) ────────────────────────────────────────────────

def _unpickle(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10(normalize: bool = True, extracted_dir: str = None):
    if extracted_dir is None:
        extracted_dir = os.path.join(BASE, "cifar10", "cifar-10-batches-py")
    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(
            f"CIFAR-10 not extracted yet.\n"
            f"Run: tar -xzf datasets/cifar10/cifar-10-python.tar.gz -C datasets/cifar10/"
        )
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch = _unpickle(os.path.join(extracted_dir, f"data_batch_{i}"))
        X_train_list.append(batch[b"data"])
        y_train_list.extend(batch[b"labels"])
    X_train = np.vstack(X_train_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train_list)

    test_batch = _unpickle(os.path.join(extracted_dir, "test_batch"))
    X_test = np.array(test_batch[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b"labels"])

    if normalize:
        X_train, X_test = X_train / 255.0, X_test / 255.0
    print(f"[CIFAR-10] train={X_train.shape}, test={X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    (Xtr, ytr), (Xte, yte) = load_mnist()
    print(f"  Labels: {np.unique(ytr)}, classes: {len(np.unique(ytr))}")

    (Xtr, ytr), (Xte, yte) = load_fashion_mnist()
    print(f"  Labels: {np.unique(ytr)}, classes: {len(np.unique(ytr))}")

    try:
        (Xtr, ytr), (Xte, yte) = load_cifar10()
        print(f"  Labels: {np.unique(ytr)}, classes: {len(np.unique(ytr))}")
    except FileNotFoundError as e:
        print(f"  CIFAR-10: {e}")
