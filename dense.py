"""Dense ReLU MLP baseline (d -> H -> C) with manual backprop, plus the shared data generator and Adam."""
from __future__ import annotations

from typing import Dict

import numpy as np

Params = Dict[str, np.ndarray]


def make_task(n: int, seed: int, d: int = 16, n_clusters: int = 8, n_classes: int = 4, spread: float = 1.0):
    """Multi-cluster task: inputs come from well-separated clusters and each cluster labels its points with
    its *own* random linear rule, argmax(A_c (x - mu_c)). One expert per cluster is the ideal solution."""
    rng = np.random.default_rng(2024)                   # task definition shared by train/test
    mu = rng.normal(0, 4.0, (n_clusters, d))
    A = rng.normal(0, 1.0, (n_clusters, n_classes, d))
    r = np.random.default_rng(seed)
    c = r.integers(0, n_clusters, n)
    X = mu[c] + r.normal(0, spread, (n, d))
    y = np.einsum("nkd,nd->nk", A[c], X - mu[c]).argmax(1)
    return X, y, c


def init_dense(d: int, h: int, c: int, rng) -> Params:
    return {"W1": rng.normal(0, np.sqrt(2 / d), (d, h)), "b1": np.zeros(h),
            "W2": rng.normal(0, np.sqrt(2 / h), (h, c)), "b2": np.zeros(c)}


def dense_forward(p: Params, X):
    H = np.maximum(X @ p["W1"] + p["b1"], 0)
    return H @ p["W2"] + p["b2"], {"X": X, "H": H}


def dense_backward(p: Params, cache, dy) -> Params:
    dH = dy @ p["W2"].T * (cache["H"] > 0)
    return {"W1": cache["X"].T @ dH, "b1": dH.sum(0), "W2": cache["H"].T @ dy, "b2": dy.sum(0)}


class Adam:
    def __init__(self, p: Params, lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}

    def step(self, p: Params, g: Params) -> None:
        self.t += 1
        for k in p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            p[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def ce_grad(logits, y):
    """Mean cross-entropy and its gradient w.r.t. logits."""
    z = logits - logits.max(1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(1, keepdims=True)
    loss = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
    P[np.arange(len(y)), y] -= 1
    return loss, P / len(y)
