"""Streaming expert-usage profiler.

Attaches a forward-pre-hook to each sparse-MoE block's `experts` module, which receives
(hidden_states [T, H], top_k_index [T, k], top_k_weights [T, k]) exactly as the model
computes them. Accumulates per (layer, expert), in fp32 on-device:

  counts        tokens routed to the expert
  gate_sum      sum of final (renormalized) routing weights over routed tokens
  prob_mass     sum over ALL tokens of the full-softmax probability (contrastive signal)
  norm_sum      sum over routed tokens of ||f_e(x)||_2  (unweighted expert output)
  norm_sq_sum   sum of squared norms
  reap_sum      sum of g * ||f_e(x)||_2  (REAP numerator)

The block sums weighted expert outputs in place, so unweighted per-expert norms are not
observable from module outputs; they are recomputed here from the module's own fused
weights (same math as Qwen3_5MoeExperts.forward). That doubles MoE FLOPs during
profiling, which is one-shot and dominated by prompt processing anyway.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


STAT_KEYS = ("counts", "gate_sum", "prob_mass", "norm_sum", "norm_sq_sum", "reap_sum")


class ExpertStats:
    def __init__(self, num_layers: int, num_experts: int, device):
        self.tensors = {k: torch.zeros(num_layers, num_experts, dtype=torch.float32, device=device) for k in STAT_KEYS}
        self.total_tokens = 0

    def to_cpu(self) -> dict:
        out = {k: v.cpu() for k, v in self.tensors.items()}
        out["total_tokens"] = torch.tensor(self.total_tokens)
        return out

    def save(self, path: str | Path, meta: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stats": self.to_cpu(), "meta": meta}, path)


def load_stats(path: str | Path) -> tuple[dict, dict]:
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["stats"], d["meta"]


def _make_hook(stats: ExpertStats, row: int, router_weight: torch.nn.Parameter | None,
               count_tokens: bool, router_prob: str = "softmax"):
    """router_weight None means the adapter accumulates prob_mass through its own router
    hook (gemma4: the router input differs from the experts input, so recomputing logits
    here would be wrong); this hook then covers only counts, gates, and norms."""
    E = stats.tensors["counts"].shape[1]

    @torch.no_grad()
    def hook(experts_module, args):
        hidden, top_k_index, top_k_weights = args[0], args[1], args[2]
        T = hidden.shape[0]
        if count_tokens:
            stats.total_tokens += T
        idx = top_k_index.reshape(-1)
        w = top_k_weights.reshape(-1).float()
        st = stats.tensors
        st["counts"][row] += torch.bincount(idx, minlength=E).float()
        st["gate_sum"][row].scatter_add_(0, idx, w)
        # Full-distribution probability mass over all tokens, matching the router's own
        # normalization. The softmax branch is byte-identical to the original qwen path;
        # glm4_moe adds a per-expert sigmoid branch (DeepSeek-style routing). None means
        # the adapter's own router hook accumulates prob_mass instead (gemma4).
        if router_weight is not None:
            logits = torch.nn.functional.linear(hidden, router_weight)
            if router_prob == "sigmoid":
                st["prob_mass"][row] += logits.float().sigmoid().sum(dim=0)
            else:
                st["prob_mass"][row] += torch.softmax(logits.float(), dim=-1).sum(dim=0)
        # Unweighted expert output norms, recomputed with the module's own weights.
        gup, down, act = experts_module.gate_up_proj, experts_module.down_proj, experts_module.act_fn
        for e in idx.unique():
            pair_rows = (idx == e).nonzero(as_tuple=True)[0]  # (token, slot) pairs routed to e
            tok_rows = pair_rows // top_k_index.shape[1]
            xs = hidden[tok_rows]
            gate, up = torch.nn.functional.linear(xs, gup[e]).chunk(2, dim=-1)
            f = torch.nn.functional.linear(act(gate) * up, down[e])
            n = f.float().norm(dim=-1)
            st["norm_sum"][row, e] += n.sum()
            st["norm_sq_sum"][row, e] += (n * n).sum()
            st["reap_sum"][row, e] += (w[pair_rows] * n).sum()

    return hook


def attach(model, adapter) -> tuple[ExpertStats, list, list[int]]:
    """Install hooks on every sparse-MoE block. Returns (stats, handles, layer_ids)."""
    blocks = list(adapter.moe_blocks(model))
    layer_ids = [i for i, _ in blocks]
    device = next(model.parameters()).device
    stats = ExpertStats(len(blocks), adapter.num_experts, device)
    handles = []
    for row, (_, block) in enumerate(blocks):
        if hasattr(adapter, "attach_prob_hook"):
            handles.append(adapter.attach_prob_hook(block, stats, row))
            router_weight = None
        else:
            router_weight = block.gate.weight
        handles.append(
            block.experts.register_forward_pre_hook(
                _make_hook(stats, row, router_weight, count_tokens=(row == 0),
                           router_prob=getattr(adapter, "router_prob", "softmax"))
            )
        )
    return stats, handles, layer_ids


def pack_token_stream(token_lists: list[list[int]], seq_len: int, eos_id: int) -> torch.Tensor:
    """Concatenate tokenized samples with EOS separators and chunk into [N, seq_len].
    Packing avoids padding entirely, which matters because the hybrid linear-attention
    layers have no attention-mask path we want to rely on for calibration."""
    flat: list[int] = []
    for ids in token_lists:
        flat.extend(ids)
        flat.append(eos_id)
    n = len(flat) // seq_len
    return torch.tensor(flat[: n * seq_len], dtype=torch.long).view(n, seq_len)


@torch.no_grad()
def profile(model, adapter, batches: torch.Tensor, batch_size: int, log_every: int = 20, log=print):
    """Run packed batches [N, seq_len] through the model, accumulating expert stats."""
    stats, handles, layer_ids = attach(model, adapter)
    # Under disk offload the first parameter can be on the meta device; inputs must go to the
    # real execution device, so pick the first non-meta parameter's device (cpu for the
    # synthetic tests, cuda:0 for an offloaded real model).
    device = next((p.device for p in model.parameters() if p.device.type != "meta"),
                  next(model.parameters()).device)
    t0 = time.time()
    n = batches.shape[0]
    try:
        for i in range(0, n, batch_size):
            chunk = batches[i : i + batch_size].to(device)
            model(input_ids=chunk, use_cache=False)
            if log_every and (i // batch_size) % log_every == 0:
                done = min(i + batch_size, n)
                rate = stats.total_tokens / max(time.time() - t0, 1e-9)
                log(f"profile: {done}/{n} seqs, {stats.total_tokens} tokens, {rate:.0f} tok/s")
    finally:
        for h in handles:
            h.remove()
    return stats, layer_ids


def write_summary(stats_path: Path, stats: ExpertStats, layer_ids: list[int], meta: dict):
    stats.save(stats_path, meta | {"layer_ids": layer_ids})
    # Which accumulators are fully finite. counts/gate_sum come from the forward args and
    # are robust even under disk offload; the norm-recomputed stats (norm_sum/reap_sum) read
    # expert weights and can be corrupted if offload de-materializes them mid-hook. Surfacing
    # this prevents a silent bad selection: a criterion whose inputs are non-finite must not
    # be used (score/select should fall back to a finite one, e.g. gate_mass = routing-mass).
    finite = {k: bool(torch.isfinite(v).all()) for k, v in stats.tensors.items()}
    summary = {
        "total_tokens": stats.total_tokens,
        "layers": len(layer_ids),
        "experts": stats.tensors["counts"].shape[1],
        "min_layer_expert_coverage": float((stats.tensors["counts"] > 0).float().mean()),
        "stats_finite": finite,
        "usable_criteria": ["freq", "gate_mass"] + (["reap", "man", "msan"] if finite["reap_sum"] and finite["norm_sum"] else []),
        **{k: v for k, v in meta.items() if isinstance(v, (str, int, float))},
    }
    if not all(finite.values()):
        summary["WARNING"] = f"non-finite accumulators: {[k for k,ok in finite.items() if not ok]}; norm-based criteria unusable"
    Path(str(stats_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return summary
