"""Training loops for the dense baseline and the MoE (with optional load-balancing loss)."""
from __future__ import annotations

import numpy as np

from dense import Adam, ce_grad, dense_backward, dense_forward, init_dense
from moe import aux_loss, init_moe, moe_backward, moe_forward


def train_dense(X, y, Xt, yt, h, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed)
    p = init_dense(X.shape[1], h, int(y.max()) + 1, rng)
    opt, hist = Adam(p, lr), []
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            out, cache = dense_forward(p, X[b])
            _, dy = ce_grad(out, y[b])
            opt.step(p, dense_backward(p, cache, dy))
        hist.append(float((dense_forward(p, Xt)[0].argmax(1) == yt).mean()))
    return p, hist


def train_moe(X, y, Xt, yt, h, n_experts, k, aux_coef, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed)
    p = init_moe(X.shape[1], h, int(y.max()) + 1, n_experts, rng)
    opt, hist = Adam(p, lr), {"test_acc": [], "aux": [], "max_load": []}
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            out, cache = moe_forward(p, X[b], k)
            _, dy = ce_grad(out, y[b])
            la, dz_aux, f, _ = aux_loss(cache, n_experts, k)
            opt.step(p, moe_backward(p, cache, dy, aux_coef * dz_aux if aux_coef else None))
        out, cache = moe_forward(p, Xt, k)
        la, _, f, _ = aux_loss(cache, n_experts, k)
        hist["test_acc"].append(float((out.argmax(1) == yt).mean()))
        hist["aux"].append(la)
        hist["max_load"].append(float(f.max()))
    return p, hist
