"""Mixture-of-Experts layer from scratch in NumPy: top-k softmax router, MLP experts, Switch-style load-balancing loss.

Forward for a batch X (n, d):
    router logits  z = X Wg + bg                     (n, E)
    top-k experts  S = argtopk(z)                    k per token
    gates          g = softmax(z restricted to S)    (renormalised over the k chosen experts, 0 elsewhere)
    output         y = sum_e g[:, e] * expert_e(X)   (n, C)   expert_e = ReLU MLP d -> h -> C
Aux loss (Fedus et al., Switch Transformer): L_aux = E * sum_e f_e * P_e
    f_e = fraction of top-k slots routed to expert e (no gradient),  P_e = mean router softmax prob of e.
It is minimised (= 1) when routing is uniform.

For clarity every expert runs on every token here (E is tiny), but the gates are exactly zero for experts that
were not selected, so outputs and gradients equal those of sparse dispatch.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

Params = Dict[str, np.ndarray]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def init_moe(d: int, h: int, c: int, n_experts: int, rng) -> Params:
    return {"Wg": rng.normal(0, 0.01, (d, n_experts)), "bg": np.zeros(n_experts),
            "W1": rng.normal(0, np.sqrt(2 / d), (n_experts, d, h)), "b1": np.zeros((n_experts, h)),
            "W2": rng.normal(0, np.sqrt(2 / h), (n_experts, h, c)), "b2": np.zeros((n_experts, c))}


def moe_forward(p: Params, X: np.ndarray, k: int):
    z = X @ p["Wg"] + p["bg"]
    top = np.argsort(-z, axis=1)[:, :k]
    mask = np.zeros_like(z, dtype=bool)
    np.put_along_axis(mask, top, True, axis=1)
    g = softmax(np.where(mask, z, -np.inf))                       # (n, E), zero outside top-k
    H = np.maximum(np.einsum("nd,edh->enh", X, p["W1"]) + p["b1"][:, None], 0)   # (E, n, h)
    O = np.einsum("enh,ehc->enc", H, p["W2"]) + p["b2"][:, None]                 # (E, n, C)
    y = np.einsum("ne,enc->nc", g, O)
    return y, {"X": X, "z": z, "mask": mask, "g": g, "H": H, "O": O, "top": top}


def aux_loss(cache, n_experts: int, k: int):
    """Switch load-balancing loss and its gradient w.r.t. router logits z."""
    probs = softmax(cache["z"])
    f = cache["mask"].mean(0) / k                     # fraction of routed slots per expert (sums to 1)
    P = probs.mean(0)
    loss = n_experts * float(f @ P)
    dP = n_experts * f                                # dL/dP_e
    n = probs.shape[0]
    dz = probs * (dP - (probs * dP).sum(1, keepdims=True)) / n
    return loss, dz, f, P


def moe_backward(p: Params, cache, dy: np.ndarray, dz_extra: np.ndarray | None = None) -> Params:
    X, g, H, O, mask = cache["X"], cache["g"], cache["H"], cache["O"], cache["mask"]
    dg = np.einsum("nc,enc->ne", dy, O)
    dz = g * (dg - (g * dg).sum(1, keepdims=True))            # softmax over selected experts
    dz = np.where(mask, dz, 0.0)
    if dz_extra is not None:
        dz = dz + dz_extra
    dO = np.einsum("ne,nc->enc", g, dy)
    dH = np.einsum("enc,ehc->enh", dO, p["W2"]) * (H > 0)
    return {"Wg": X.T @ dz, "bg": dz.sum(0),
            "W2": np.einsum("enh,enc->ehc", H, dO), "b2": dO.sum(1),
            "W1": np.einsum("nd,enh->edh", X, dH), "b1": dH.sum(1)}


def active_params(p: Params, k: int) -> int:
    E = p["W1"].shape[0]
    expert = (p["W1"].size + p["b1"].size + p["W2"].size + p["b2"].size) // E
    return int(p["Wg"].size + p["bg"].size + k * expert)


def total_params(p: Params) -> int:
    return int(sum(v.size for v in p.values()))
