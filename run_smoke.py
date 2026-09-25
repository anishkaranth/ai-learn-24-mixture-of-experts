#!/usr/bin/env python3
"""MoE smoke: dense MLP (equal active params / equal total params) vs top-2 MoE with an aux-loss sweep -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from dense import make_task
from moe import active_params, aux_loss, moe_forward, total_params
from smoke_plots import make_plots, write_results_md
from train import train_dense, train_moe

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {"d": 16, "n_clusters": 8, "n_classes": 4, "n_train": 4000, "n_test": 2000, "n_experts": 8, "top_k": 2,
       "expert_hidden": 16, "epochs": 40, "lr": 3e-3, "batch": 64, "aux_coefs": [0.0, 0.01, 0.1, 1.0],
       "dead_expert_threshold": 0.02}


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def r4(x) -> float:
    return round(float(x), 4)


def dense_hidden_for(n_params: int, d: int, c: int) -> int:
    """Hidden width H so that d*H + H + H*c + c is as close as possible to n_params."""
    return max(1, round((n_params - c) / (d + 1 + c)))


def main() -> None:
    t0 = time.perf_counter()
    X, y, _ = make_task(CFG["n_train"], 1, CFG["d"], CFG["n_clusters"], CFG["n_classes"])
    Xt, yt, ct = make_task(CFG["n_test"], 2, CFG["d"], CFG["n_clusters"], CFG["n_classes"])
    E, k, ep, lr, bs = CFG["n_experts"], CFG["top_k"], CFG["epochs"], CFG["lr"], CFG["batch"]

    moe_runs, routing = [], {}
    for a in CFG["aux_coefs"]:
        p, hist = train_moe(X, y, Xt, yt, CFG["expert_hidden"], E, k, a, ep, lr, bs, SEED)
        out, cache = moe_forward(p, Xt, k)
        la, _, f, P = aux_loss(cache, E, k)
        top1 = cache["top"][:, 0]
        M = np.zeros((CFG["n_clusters"], E), dtype=int)
        np.add.at(M, (ct, top1), 1)
        ent = float(-(f[f > 0] * np.log(f[f > 0])).sum() / np.log(E))
        moe_runs.append({"aux_coef": a, "test_acc": r4((out.argmax(1) == yt).mean()), "aux_loss_value": r4(la),
                         "utilization": [r4(v) for v in f], "max_load": r4(f.max()), "load_cv": r4(f.std() / f.mean()),
                         "utilization_entropy": r4(ent), "dead_experts": int((f < CFG["dead_expert_threshold"]).sum()),
                         "cluster_purity_top1": r4(M.max(1).sum() / M.sum()),
                         "test_acc_curve": [r4(v) for v in hist["test_acc"]]})
        routing[str(a)] = M.tolist()
    n_active, n_total = active_params(p, k), total_params(p)

    dense_runs = []
    for name, target in (("dense_equal_active", n_active), ("dense_equal_total", n_total)):
        H = dense_hidden_for(target, CFG["d"], CFG["n_classes"])
        pd, hist = train_dense(X, y, Xt, yt, H, ep, lr, bs, SEED)
        dense_runs.append({"name": name, "hidden": H, "params": int(sum(v.size for v in pd.values())),
                           "test_acc": r4(hist[-1]), "test_acc_curve": [r4(v) for v in hist]})

    by = {r["aux_coef"]: r for r in moe_runs}
    da = dense_runs[0]
    m = {"project": "ai-learn-24-mixture-of-experts", "seed": SEED, "config": CFG,
         "moe_params": {"active_per_token": n_active, "total": n_total}, "moe_runs": moe_runs, "dense_runs": dense_runs,
         "routing_cluster_x_top1_expert": routing,
         "headline": {
             "dense_equal_active_acc": da["test_acc"], "dense_equal_active_params": da["params"],
             "dense_equal_total_acc": dense_runs[1]["test_acc"], "dense_equal_total_params": dense_runs[1]["params"],
             "moe_active_params": n_active, "moe_total_params": n_total,
             "moe_noaux_acc": by[0.0]["test_acc"], "moe_noaux_dead_experts": by[0.0]["dead_experts"],
             "moe_noaux_min_util": min(by[0.0]["utilization"]), "moe_noaux_load_cv": by[0.0]["load_cv"],
             "moe_aux0.1_acc": by[0.1]["test_acc"], "moe_aux0.1_dead_experts": by[0.1]["dead_experts"],
             "moe_aux0.1_min_util": min(by[0.1]["utilization"]), "moe_aux0.1_load_cv": by[0.1]["load_cv"],
             "moe_aux1.0_acc": by[1.0]["test_acc"], "moe_aux1.0_load_cv": by[1.0]["load_cv"],
             "moe_aux0.1_cluster_purity": by[0.1]["cluster_purity_top1"],
         }}
    m["wall_time_s"] = round(time.perf_counter() - t0, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)) + "\n")
    shot = {"project": m["project"], "seed": SEED, "config": CFG, "moe_params": m["moe_params"], "headline": m["headline"]}
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(m["headline"], indent=2), "\nwall", m["wall_time_s"])


if __name__ == "__main__":
    main()
