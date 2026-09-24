"""
mlp.py
Two-layer MLP in pure NumPy — backbone for the KALT framework.
Supports variable hidden sizes per dataset (MNIST vs CIFAR-10).
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-9)

def cross_entropy(y_hat, y, n_classes):
    y_oh = np.eye(n_classes)[y.astype(int)]
    return float(-np.mean(np.sum(y_oh * np.log(y_hat + 1e-9), axis=1)))

def accuracy_mlp(params, X, y, n_classes):
    _, _, _, y_hat = _forward(params, X)
    return float(np.mean(np.argmax(y_hat, axis=1) == y.astype(int)))


def init_params(n_features, hidden1, hidden2, n_classes, seed=0):
    """He-style initialisation."""
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / n_features)
    scale2 = np.sqrt(2.0 / hidden1)
    scale3 = np.sqrt(2.0 / hidden2)
    return {
        "W1": rng.normal(0, scale1, (n_features, hidden1)),
        "b1": np.zeros(hidden1),
        "W2": rng.normal(0, scale2, (hidden1, hidden2)),
        "b2": np.zeros(hidden2),
        "W3": rng.normal(0, scale3, (hidden2, n_classes)),
        "b3": np.zeros(n_classes),
    }


def params_to_flat(p):
    return np.concatenate([p["W1"].flatten(), p["b1"],
                           p["W2"].flatten(), p["b2"],
                           p["W3"].flatten(), p["b3"]])

def flat_to_params(flat, n_features, h1, h2, n_classes):
    idx = 0
    def _take(shape):
        nonlocal idx
        n = int(np.prod(shape))
        v = flat[idx:idx+n].reshape(shape)
        idx += n
        return v
    return {
        "W1": _take((n_features, h1)),
        "b1": _take(h1),
        "W2": _take((h1, h2)),
        "b2": _take(h2),
        "W3": _take((h2, n_classes)),
        "b3": _take(n_classes),
    }


def _forward(p, X):
    z1   = X  @ p["W1"] + p["b1"]
    a1   = relu(z1)
    z2   = a1 @ p["W2"] + p["b2"]
    a2   = relu(z2)
    z3   = a2 @ p["W3"] + p["b3"]
    y_hat= softmax(z3)
    return a1, a2, z1, y_hat   # return pre-activations for backprop


def _backward(p, X, y, a1, a2, z1, y_hat, n_classes, l2_lambda=1e-4):
    n     = X.shape[0]
    y_oh  = np.eye(n_classes)[y.astype(int)]

    # Output layer delta
    dz3   = (y_hat - y_oh) / n
    dW3   = a2.T @ dz3 + l2_lambda * p["W3"]
    db3   = dz3.sum(axis=0)

    # Hidden layer 2 delta
    da2   = dz3 @ p["W3"].T
    dz2   = da2 * relu_grad(a2)          # reuse a2 as z2 ≥0 ↔ a2≥0
    dW2   = a1.T @ dz2 + l2_lambda * p["W2"]
    db2   = dz2.sum(axis=0)

    # Hidden layer 1 delta
    da1   = dz2 @ p["W2"].T
    dz1   = da1 * relu_grad(a1)
    dW1   = X.T  @ dz1 + l2_lambda * p["W1"]
    db1   = dz1.sum(axis=0)

    return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}


def train_one_epoch(p, X, y, n_classes, lr, batch_size, l2_lambda,
                    grad_clip_norm, rng):
    """One pass over the data with mini-batch SGD + gradient clipping."""
    n   = len(X)
    idx = rng.permutation(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        xb = X[idx[start:start+batch_size]]
        yb = y[idx[start:start+batch_size]].astype(int)

        a1, a2, z1, y_hat = _forward(p, xb)
        total_loss += cross_entropy(y_hat, yb, n_classes) * len(xb)

        grads = _backward(p, xb, yb, a1, a2, z1, y_hat, n_classes, l2_lambda)

        # Gradient clipping (by global norm)
        if grad_clip_norm > 0:
            all_grads = np.concatenate([v.flatten() for v in grads.values()])
            gnorm = np.linalg.norm(all_grads)
            if gnorm > grad_clip_norm:
                scale = grad_clip_norm / gnorm
                grads = {k: v*scale for k, v in grads.items()}

        for key in p:
            p[key] -= lr * grads[key]

    return p, total_loss / n


def local_train_mlp(params_flat, X, y, n_classes,
                    n_features, h1, h2,
                    lr=0.01, epochs=5, batch_size=64,
                    l2_lambda=1e-4, grad_clip_norm=5.0, seed=0):
    """
    Full local training loop. Returns (flat_weights, loss, accuracy).
    """
    p   = flat_to_params(params_flat, n_features, h1, h2, n_classes)
    rng = np.random.default_rng(seed)
    loss = 0.0
    for _ in range(epochs):
        p, loss = train_one_epoch(p, X, y, n_classes, lr, batch_size,
                                   l2_lambda, grad_clip_norm, rng)
    acc = accuracy_mlp(p, X, y, n_classes)
    return params_to_flat(p), float(loss), float(acc)
