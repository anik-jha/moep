"""Expert scoring criteria and keep-set selection.

Criteria follow the unified formulation of arXiv:2606.15716 (freq^b * gate^a * norm^c):
task-specific pruning (our coding case) benefits from retaining frequency and gate
signals (REAP-style); MAN/MSAN are the gate-free task-agnostic controls. REAP saliency
(arXiv:2510.13999): mean over routed tokens of gate * ||expert_output||.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

EPS = 1e-12
GUARD_RE = re.compile(r"^guard(\d+)$")


def criteria_matrix(stats: dict) -> dict[str, torch.Tensor]:
    """Compute all criteria as [n_layers, E] tensors from accumulated stats."""
    counts = stats["counts"]
    total = float(stats["total_tokens"])
    routed = counts.clamp_min(1.0)
    return {
        "freq": counts / total,
        "gate_mass": stats["gate_sum"] / total,
        "prob_mass": stats["prob_mass"] / total,
        "man": stats["norm_sum"] / routed,
        "msan": stats["norm_sq_sum"] / routed,
        "reap": stats["reap_sum"] / routed,  # mean g*||f|| over routed tokens (paper's S)
        "reap_flow": stats["reap_sum"] / total,  # frequency-weighted variant (per-token flow)
    }


def contrastive(primary: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                base: str = "reap_flow") -> torch.Tensor:
    """Domain-contrastive score: importance on the target domain relative to a reference
    domain. High = domain-specialist worth keeping under domain-focused pruning.
    CAUTION (measured 2026-07-12): used alone this destroys the model at 50% keep. It
    strips high-magnitude domain-neutral experts (score ~1) that every input needs.
    Use the magnitude-gated hybrids below instead."""
    return primary[base] / (reference[base] + EPS)


def hybrid_share(primary: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                 base: str = "reap_flow") -> torch.Tensor:
    """Flow times domain-share: S = F_T * F_T/(F_T+F_R) = F_T^2/(F_T+F_R).
    Parameter-free. Keeps magnitude in charge; the share term breaks ties toward
    domain specialists and demotes experts whose flow is mostly off-domain."""
    ft, fr = primary[base], reference[base]
    return ft * ft / (ft + fr + EPS)


def hybrid_rank(primary: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                alpha: float = 0.5, base: str = "reap_flow") -> torch.Tensor:
    """Per-layer rank blend: alpha * rank(F_T) + (1-alpha) * rank(F_T/F_R), scale-free
    so neither signal dominates by unit choice. alpha=0.5 weights them equally."""
    ft = primary[base]
    ratio = ft / (reference[base] + EPS)

    def ranks(x):
        order = x.argsort(dim=-1)
        r = torch.empty_like(order)
        idx = torch.arange(x.shape[-1], dtype=order.dtype).expand_as(order)
        r.scatter_(-1, order, idx)
        return r.float() / (x.shape[-1] - 1)

    return alpha * ranks(ft) + (1 - alpha) * ranks(ratio)


def guard_union(primary: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                guard_n: int) -> torch.Tensor:
    """Union keep: the top guard_n experts per layer by REFERENCE (general-domain) gate
    mass are unprunable; the remaining budget is filled by primary REAP saliency.
    Motivated by the 2026-07-16 diagnostic: reap-0.25 prunes ~half of each layer's
    top-32 general-mass experts and general coverage falls to 0.35. Mirror image of
    contrastive (which demotes domain-neutral experts and collapses): protect them.
    Scores are per-layer normalized ranks in [0, 1]; guarded experts get 2.0, so any
    keep_per_layer >= guard_n keeps every guarded expert."""
    ranks = primary["reap"].argsort(dim=-1).argsort(dim=-1).float()
    scores = ranks / max(1, ranks.shape[-1] - 1)
    guard_idx = reference["gate_mass"].topk(guard_n, dim=-1).indices
    return scores.scatter(-1, guard_idx, 2.0)


def keep_sets(scores: torch.Tensor, keep_per_layer: int) -> list[list[int]]:
    """Per layer, indices of the top-scoring experts, ascending (ascending order keeps
    surgery renumbering order-preserving, which the equivalence tests rely on)."""
    k = scores.topk(keep_per_layer, dim=-1).indices
    return [sorted(row.tolist()) for row in k]


def select(stats: dict, criterion: str, keep_ratio: float,
           reference_stats: dict | None = None, seed: int = 17) -> tuple[list[list[int]], dict]:
    mats = criteria_matrix(stats)
    guard_m = GUARD_RE.match(criterion)
    if criterion in ("contrastive", "hybrid_share", "hybrid_rank") or guard_m:
        if reference_stats is None:
            raise ValueError(f"criterion={criterion} requires reference stats")
        if guard_m:
            scores = guard_union(mats, criteria_matrix(reference_stats), int(guard_m.group(1)))
        else:
            fn = {"contrastive": contrastive, "hybrid_share": hybrid_share,
                  "hybrid_rank": hybrid_rank}[criterion]
            scores = fn(mats, criteria_matrix(reference_stats))
    elif criterion == "random":
        # Seeded random keep: the baseline every reviewer asks for.
        g = torch.Generator().manual_seed(seed)
        scores = torch.rand(mats["freq"].shape, generator=g)
    else:
        scores = mats[criterion]
    E = scores.shape[1]
    keep_n = max(1, round(E * keep_ratio))
    if guard_m and int(guard_m.group(1)) > keep_n:
        raise ValueError(f"guard_n={guard_m.group(1)} exceeds keep_per_layer={keep_n}")
    keeps = keep_sets(scores, keep_n)
    info = {
        "criterion": criterion,
        "keep_ratio": keep_ratio,
        "keep_per_layer": keep_n,
        "num_experts": E,
        "seed": seed if criterion == "random" else None,
        "score_stats": {
            "min": float(scores.min()),
            "median": float(scores.median()),
            "max": float(scores.max()),
        },
    }
    return keeps, info


def save_selection(path: str | Path, keeps: list[list[int]], layer_ids: list[int], info: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"info": info, "layer_ids": layer_ids,
                                "keep": {str(l): k for l, k in zip(layer_ids, keeps)}}, indent=2))


def load_selection(path: str | Path) -> tuple[dict[int, list[int]], dict]:
    d = json.loads(Path(path).read_text())
    return {int(k): v for k, v in d["keep"].items()}, d["info"]
