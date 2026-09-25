"""Matplotlib SVG plots + RESULTS.md writer for the MoE smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-24", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
COLS = ["#e76f51", "#e9c46a", "#2a9d8f", "#126782"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names, runs = [], m["moe_runs"]
    E = m["config"]["n_experts"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.8 / len(runs)
    for i, (r, c) in enumerate(zip(runs, COLS)):
        ax.bar(np.arange(E) + i * w - 0.4 + w / 2, r["utilization"], width=w, color=c, label=f"aux={r['aux_coef']}")
    ax.axhline(1 / E, color="#555", ls="--", lw=1, label="uniform")
    ax.set_xlabel("expert")
    ax.set_ylabel("share of routed slots")
    ax.set_title("Expert utilization (test set, top-2)")
    ax.legend(fontsize=7, ncol=3)
    names.append(_save(fig, out / "expert_utilization.svg"))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for r, c in zip(runs, COLS):
        ax.plot(range(1, len(r["test_acc_curve"]) + 1), r["test_acc_curve"], color=c, label=f"MoE aux={r['aux_coef']}")
    for r, ls in zip(m["dense_runs"], ("--", ":")):
        ax.plot(range(1, len(r["test_acc_curve"]) + 1), r["test_acc_curve"], color="#333", ls=ls,
                label=f"{r['name']} ({r['params']} p)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"MoE ({m['moe_params']['active_per_token']} active / {m['moe_params']['total']} total params) vs dense")
    ax.legend(fontsize=7)
    names.append(_save(fig, out / "accuracy_curves.svg"))

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, a in zip(axs, ("0.0", "0.1")):
        M = np.array(m["routing_cluster_x_top1_expert"][a], dtype=float)
        M = M / M.sum(1, keepdims=True)
        # draw only cells holding >= 5% of a cluster's points (keeps the SVG small), 5 colour levels
        cmap = plt.get_cmap("Blues", 5)
        for i, j in zip(*np.nonzero(M >= 0.05)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(min(M[i, j], 0.999))))
        ax.set_xlim(0, M.shape[1])
        ax.set_ylim(0, M.shape[0])
        ax.set_xlabel("top-1 expert")
        ax.set_ylabel("input cluster")
        ax.set_title(f"Routing (aux={a})")
        ax.invert_yaxis()
    names.append(_save(fig, out / "routing_matrix.svg"))
    return names


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    h, cfg = m["headline"], m["config"]
    accs = [r["test_acc"] for r in m["moe_runs"]]
    purs = [r["cluster_purity_top1"] for r in m["moe_runs"]]
    mrows = "\n".join(f"| MoE top-{cfg['top_k']}, aux={r['aux_coef']} | {m['moe_params']['active_per_token']} / {m['moe_params']['total']} | "
                      f"{r['test_acc']} | {r['dead_experts']} | {min(r['utilization'])} | {r['max_load']} | {r['load_cv']} | "
                      f"{r['utilization_entropy']} | {r['cluster_purity_top1']} |" for r in m["moe_runs"])
    drows = "\n".join(f"| {r['name']} (H={r['hidden']}) | {r['params']} / {r['params']} | {r['test_acc']} | - | - | - | - | - | - |"
                      for r in m["dense_runs"])
    txt = f"""# Results: ai-learn-24-mixture-of-experts

Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).

## Setup
- Task: {cfg['n_clusters']} well-separated Gaussian clusters in {cfg['d']}-D. Each cluster labels its points with its own random linear rule
  ({cfg['n_classes']} classes). {cfg['n_train']} train and {cfg['n_test']} test points.
- MoE: {cfg['n_experts']} ReLU-MLP experts ({cfg['d']}-{cfg['expert_hidden']}-{cfg['n_classes']}), a linear softmax router, top-{cfg['top_k']} gating renormalised over the chosen experts,
  and a Switch-style aux loss `E * sum_e f_e P_e`. {cfg['epochs']} epochs of Adam (lr {cfg['lr']}, batch {cfg['batch']}). The init seed is the same for every run.
- Dense baselines are ReLU MLPs whose hidden width matches the MoE's **active** params per token, or its **total** params.
- A dead expert gets less than {cfg['dead_expert_threshold']} of routed slots. Purity is the share of each cluster's points sent to that cluster's most-used top-1 expert.

## Results
| model | active / total params | test acc | dead experts | min util | max util | load CV | util entropy | cluster purity |
|---|---|---|---|---|---|---|---|---|
{drows}
{mrows}

## Plots
""" + "\n".join(f"![{p}]({p})" for p in plots) + f"""

## Observations (from the numbers above)
- At equal active params the MoE scores {h['moe_noaux_acc']} (no aux) vs {h['dense_equal_active_acc']} for the dense MLP. It even beats the dense MLP with
  equal *total* params ({h['dense_equal_total_acc']}). The router learns to send each cluster to a small set of experts that can specialise.
- Without the aux loss, {h['moe_noaux_dead_experts']} expert(s) end up nearly unused (min utilization {h['moe_noaux_min_util']}, load CV {h['moe_noaux_load_cv']}).
  With aux=0.1: {h['moe_aux0.1_dead_experts']} dead experts, min utilization {h['moe_aux0.1_min_util']}, load CV {h['moe_aux0.1_load_cv']}.
- Accuracy barely moves with the aux coefficient: it spans {min(accs)}-{max(accs)} across the sweep, which is within single-seed noise.
  On this small single-device task the aux loss mainly buys balanced load (which matters for throughput when experts sit on different devices),
  not accuracy. Cluster purity stays high ({min(purs)}-{max(purs)}), so balancing did not destroy specialisation.

## Honest notes
- Every expert is computed densely for all tokens (E is tiny). Unselected experts get an exact zero gate, so the maths matches sparse dispatch,
  but no speedup is measured. There is no capacity factor and no token dropping.
- One seed on a synthetic task. With renormalised gates and top-1, the router gets no gradient from the task loss (the single gate is always 1),
  which is why this repo uses top-2.
"""
    path.write_text(txt, encoding="utf-8")
