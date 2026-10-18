"""Keep-sets for GGUF expert pruning from native/moe_stats routing statistics.

Criteria come from moep.scoring.criteria_matrix applied one layer at a time, so they are the
same formulas as the safetensors pipeline while allowing a different expert count per layer.
Each layer keeps round(keep_ratio * its expert count) experts, never fewer than it routes to.
Split-half reliability compares the criterion computed on the two document halves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .scoring import criteria_matrix, keep_sets

STAT_KEYS = ("counts", "gate_sum", "norm_sum", "norm_sq_sum", "reap_sum")


def load_stats(path: str | Path) -> dict[int, list[dict]]:
    """Layer index -> [half0, half1], each {stat: float64 tensor [E], "tokens": float}."""
    d = json.loads(Path(path).read_text())
    out = {}
    for il, layer in d["layers"].items():
        out[int(il)] = [{**{k: torch.tensor(h[k], dtype=torch.float64) for k in STAT_KEYS},
                         "tokens": float(h["tokens"])} for h in layer["halves"]]
    return out


def _scores(halves: list[dict], criterion: str) -> torch.Tensor:
    stats = {k: sum(h[k] for h in halves)[None] for k in STAT_KEYS}
    stats["total_tokens"] = sum(h["tokens"] for h in halves)
    stats["prob_mass"] = torch.zeros_like(stats["counts"])
    return criteria_matrix(stats)[criterion]


def select(stats: dict[int, list[dict]], criterion: str, keep_ratio: float,
           n_used: dict[int, int] | int) -> tuple[dict[int, list[int]], dict]:
    keep, per_layer = {}, {}
    for il in sorted(stats):
        scores = _scores(stats[il], criterion)
        n = scores.shape[1]
        k = max(n_used[il] if isinstance(n_used, dict) else n_used, round(keep_ratio * n))
        keep[il] = keep_sets(scores, k)[0]
        per_layer[il] = {"n_expert": n, "kept": k}
    info = {"criterion": criterion, "keep_ratio": keep_ratio, "per_layer": per_layer,
            "experts_kept": sum(v["kept"] for v in per_layer.values()),
            "experts_total": sum(v["n_expert"] for v in per_layer.values())}
    return keep, info


def split_half(stats: dict[int, list[dict]], criterion: str, keep_ratio: float,
               n_used: dict[int, int] | int) -> dict:
    """Per-layer Spearman rho between the two halves' scores, and Jaccard overlap of the keep
    sets each half would choose on its own."""
    rho, jac = {}, {}
    for il in sorted(stats):
        a, b = (_scores([h], criterion)[0].numpy() for h in stats[il])
        # rank correlation is undefined when a half scores every expert the same
        rho[il] = float(spearmanr(a, b).statistic) if a.std() > 0 and b.std() > 0 else None
        k = max(n_used[il] if isinstance(n_used, dict) else n_used, round(keep_ratio * a.size))
        ka, kb = set(np.argsort(-a, kind="stable")[:k]), set(np.argsort(-b, kind="stable")[:k])
        jac[il] = len(ka & kb) / len(ka | kb)
    r = np.array([v for v in rho.values() if v is not None])
    j = np.array(list(jac.values()))
    return {"criterion": criterion, "keep_ratio": keep_ratio,
            "spearman_median": float(np.median(r)) if r.size else None,
            "spearman_min": float(r.min()) if r.size else None,
            "spearman_undefined_layers": sum(v is None for v in rho.values()),
            "jaccard_median": float(np.median(j)), "jaccard_min": float(j.min()),
            "spearman": rho, "jaccard": jac}
