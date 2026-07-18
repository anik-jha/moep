"""S5 masked-ablation causal importance (method novelty leg).

Routing-statistic criteria (REAP, gate_mass) are proxies; the contrastive coverage
paradox showed proxy saliency can misrank badly. This stage measures importance
causally: draw a random half mask per layer, set masked experts' router logits to the
dtype minimum so top-k renormalizes over survivors (behaviorally identical to pruning
them), and record LM loss on a calibration sequence. Each mask is paired with its
complement on the SAME tokens (antithetic pairing: every expert is dropped in exactly
one run of the pair), so per-expert importance is the mean over pairs of
loss(run with e masked) - loss(run with e kept); other experts' assignments are
balanced across pairs and cancel in expectation.

Checkpointed and resumable: masks and losses append to s5_samples.pt every
checkpoint_every pairs; a rerun continues from the recorded pair count. Scoring emits
a keep file in the same schema as score/select (info + layer_ids + keep{layer: ids})
so surgery/export/eval run unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def mask_router_logits(model, adapter, mask: torch.Tensor) -> list:
    """Forward hooks on each MoE gate re-running the router pipeline with masked
    experts' logits at the dtype minimum, so softmax gives them ~0 and top-k selects
    survivors only - behaviorally identical to pruning them. The recomputation mirrors
    the installed Qwen3_5MoeTopKRouter.forward (modeling_qwen3_5_moe.py:787: linear ->
    float softmax -> topk -> renormalize -> cast), which computes top-k INSIDE the gate
    and returns (router_logits, router_scores, router_indices), so patching logits
    alone would not change routing. `mask` is bool [num_layers, num_experts], True =
    masked (ablated). Returns hook handles; caller removes them."""
    import torch.nn.functional as F

    blocks = list(adapter.moe_blocks(model))
    assert len(blocks) == mask.shape[0], (len(blocks), mask.shape)
    handles = []

    def make(row: int):
        def hook(gate, args, _out):
            hs = args[0].reshape(-1, gate.hidden_dim)
            logits = F.linear(hs, gate.weight)
            logits = logits.masked_fill(mask[row].to(logits.device),
                                        torch.finfo(logits.dtype).min)
            probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
            top_v, top_i = torch.topk(probs, gate.top_k, dim=-1)
            top_v = (top_v / top_v.sum(dim=-1, keepdim=True)).to(logits.dtype)
            return logits, top_v, top_i
        return hook

    for row, (_, block) in enumerate(blocks):
        handles.append(block.gate.register_forward_hook(make(row)))
    return handles


@torch.no_grad()
def seq_loss(model, input_ids: torch.Tensor) -> float:
    device = next(model.parameters()).device
    ids = input_ids.view(1, -1).to(device)
    return float(model(input_ids=ids, labels=ids, use_cache=False).loss)


@torch.no_grad()
def run_pair(model, adapter, input_ids: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Loss under mask and under its complement, on the same tokens."""
    losses = []
    for m in (mask, ~mask):
        handles = mask_router_logits(model, adapter, m)
        try:
            losses.append(seq_loss(model, input_ids))
        finally:
            for h in handles:
                h.remove()
    return losses[0], losses[1]


def draw_mask(num_layers: int, num_experts: int, gen: torch.Generator) -> torch.Tensor:
    """Exactly half the experts of each layer masked (balanced, so the complement is
    also a half mask and top-k always has survivors)."""
    mask = torch.zeros(num_layers, num_experts, dtype=torch.bool)
    for l in range(num_layers):
        perm = torch.randperm(num_experts, generator=gen)
        mask[l, perm[: num_experts // 2]] = True
    return mask


def importance_from_samples(masks: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Per-expert mean loss-when-masked minus mean loss-when-kept.

    masks: bool [S, L, E]; losses: float [S]. The antithetic pairs are just samples
    here; pairing lives in how the samples were generated (complement rows share
    tokens), which cancels token difficulty in the difference of means.

    Estimator property (exercised by the planted-signal test): exact-half masks make
    same-layer co-masking hypergeometric, giving E[imp_e] = true_e * E/(E-1)
    - S_layer/(E-1) - a positive rescale plus per-layer constant, so per-layer
    RANKING is exact in expectation, which is all top-k selection consumes. The
    layer sum S itself is not identifiable (estimates sum to ~0 per layer)."""
    m = masks.float()
    lw = losses.view(-1, 1, 1)
    masked_mean = (m * lw).sum(0) / m.sum(0).clamp(min=1)
    kept_mean = ((1 - m) * lw).sum(0) / (1 - m).sum(0).clamp(min=1)
    return masked_mean - kept_mean


def write_keep(importance: torch.Tensor, layer_ids: list[int], keep_ratio: float,
               out_path: Path, meta: dict):
    """Keep the top keep_ratio experts per layer by causal importance, in the same
    schema as the score/select keep files (verified against keep_reap_0.25.json)."""
    num_experts = importance.shape[1]
    keep_n = round(num_experts * keep_ratio)
    keep = {str(lid): sorted(torch.topk(importance[row], keep_n).indices.tolist())
            for row, lid in enumerate(layer_ids)}
    fin = importance[torch.isfinite(importance)]
    out_path.write_text(json.dumps({
        "info": {"criterion": "ablation", "keep_ratio": keep_ratio,
                 "keep_per_layer": keep_n, "num_experts": num_experts,
                 "score_stats": {"min": float(fin.min()), "median": float(fin.median()),
                                 "max": float(fin.max())},
                 "stats_meta": meta},
        "layer_ids": layer_ids,
        "keep": keep}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3.6-35b-a3b")
    ap.add_argument("--calib", default="data/calib/calib.jsonl")
    ap.add_argument("--out", default="runs/strat/s5_ablation")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--keep-ratio", type=float, default=0.25)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "s5_samples.pt"

    from transformers import AutoTokenizer

    from .adapters import get_adapter
    from .data import read_jsonl, tokenize_rows
    from .loader import load_streaming
    from .profiler import pack_token_stream

    adapter = get_adapter(json.loads((Path(a.model) / "config.json").read_text()))
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = read_jsonl(a.calib)
    batches = pack_token_stream(tokenize_rows(tok, rows), a.seq_len, tok.eos_token_id)
    print(f"s5: {len(rows)} calib samples -> {batches.shape[0]} packed seqs of {a.seq_len}")

    masks_l, losses_l = [], []
    if samples_path.exists():
        st = torch.load(samples_path, weights_only=True)
        masks_l, losses_l = [st["masks"]], [st["losses"]]
        print(f"s5: resuming from {st['losses'].shape[0]} recorded runs")
    done_pairs = sum(x.shape[0] for x in losses_l) // 2

    model = load_streaming(a.model)
    blocks = list(adapter.moe_blocks(model))
    layer_ids = [i for i, _ in blocks]
    L, E = len(blocks), adapter.num_experts
    print(f"s5: {L} MoE layers x {E} experts, top-{adapter.top_k}; "
          f"{a.pairs} pairs, resume at {done_pairs}")

    # deterministic mask stream: regenerate and skip past completed pairs on resume
    gen = torch.Generator().manual_seed(a.seed)
    for _ in range(done_pairs):
        draw_mask(L, E, gen)

    t0, new_masks, new_losses = time.time(), [], []

    def flush():
        nonlocal masks_l, losses_l, new_masks, new_losses
        if not new_losses:
            return
        masks_l.append(torch.stack(new_masks))
        losses_l.append(torch.tensor(new_losses, dtype=torch.float32))
        new_masks, new_losses = [], []
        torch.save({"masks": torch.cat(masks_l), "losses": torch.cat(losses_l)},
                   samples_path)
        masks_l, losses_l = [torch.cat(masks_l)], [torch.cat(losses_l)]

    for p in range(done_pairs, a.pairs):
        ids = batches[p % batches.shape[0]]
        mask = draw_mask(L, E, gen)
        lm, lc = run_pair(model, adapter, ids, mask)
        if not (torch.isfinite(torch.tensor([lm, lc])).all()):
            raise RuntimeError(f"s5: non-finite loss at pair {p}: {lm}, {lc}")
        new_masks += [mask, ~mask]
        new_losses += [lm, lc]
        if (p + 1) % a.checkpoint_every == 0:
            flush()
            rate = (p + 1 - done_pairs) / max(time.time() - t0, 1e-9)
            eta_h = (a.pairs - p - 1) / max(rate, 1e-9) / 3600
            print(f"s5: {p + 1}/{a.pairs} pairs, {rate * 3600:.0f} pairs/h, eta {eta_h:.1f}h")
    flush()

    masks = torch.cat(masks_l)
    losses = torch.cat(losses_l)
    imp = importance_from_samples(masks, losses)
    torch.save({"importance": imp, "layer_ids": layer_ids}, out / "s5_importance.pt")
    keep_path = out / f"keep_ablation_{a.keep_ratio}.json"
    write_keep(imp, layer_ids, a.keep_ratio, keep_path,
               {"model": a.model, "calib": a.calib, "seq_len": a.seq_len,
                "pairs": a.pairs, "seed": a.seed, "estimator": "antithetic-half-mask"})
    print(f"s5: importance spread min={float(imp.min()):.4f} max={float(imp.max()):.4f}; "
          f"keep file {keep_path}")
    print("S5_STATS_DONE fails=0")


if __name__ == "__main__":
    main()
