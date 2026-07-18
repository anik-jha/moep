import json

import pytest
import torch

from moep.ablation import (draw_mask, importance_from_samples, mask_router_logits,
                           run_pair, seq_loss, write_keep)


def _routed_experts(model, adapter, ids):
    """Set of expert indices per layer that top-k actually selected."""
    routed = {}
    handles = []
    for row, (_, block) in enumerate(adapter.moe_blocks(model)):
        def make(r):
            def hook(_m, args):
                routed.setdefault(r, set()).update(args[1].reshape(-1).tolist())
            return hook
        handles.append(block.experts.register_forward_pre_hook(make(row)))
    try:
        model(input_ids=ids.view(1, -1), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return routed


def test_masked_experts_never_routed(tiny_ckpt, tiny_adapter, sample_ids):
    _, model = tiny_ckpt
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, 3:6] = True
    handles = mask_router_logits(model, tiny_adapter, mask)
    try:
        routed = _routed_experts(model, tiny_adapter, sample_ids[0])
    finally:
        for h in handles:
            h.remove()
    assert routed[0] <= {5, 6, 7}
    assert routed[1] <= {0, 1, 2, 6, 7}
    # hooks removed -> routing free again
    routed_free = _routed_experts(model, tiny_adapter, sample_ids[0])
    assert len(routed_free[0] | routed_free[1]) > 3


def test_no_mask_reproduces_base_loss(tiny_ckpt, tiny_adapter, sample_ids):
    _, model = tiny_ckpt
    base = seq_loss(model, sample_ids[0])
    handles = mask_router_logits(model, tiny_adapter, torch.zeros(2, 8, dtype=torch.bool))
    try:
        masked = seq_loss(model, sample_ids[0])
    finally:
        for h in handles:
            h.remove()
    assert masked == pytest.approx(base, abs=1e-6)


def test_run_pair_and_draw_mask(tiny_ckpt, tiny_adapter, sample_ids):
    _, model = tiny_ckpt
    gen = torch.Generator().manual_seed(0)
    mask = draw_mask(2, 8, gen)
    assert mask.shape == (2, 8) and (mask.sum(dim=1) == 4).all()
    lm, lc = run_pair(model, tiny_adapter, sample_ids[0], mask)
    base = seq_loss(model, sample_ids[0])
    assert torch.isfinite(torch.tensor([lm, lc])).all()
    # the mask must actually change the computation (no >= base guarantee exists on a
    # random untrained model, only difference)
    assert lm != pytest.approx(base, abs=1e-9) and lc != pytest.approx(base, abs=1e-9)
    # deterministic mask stream: same seed reproduces, stream advances
    gen2 = torch.Generator().manual_seed(0)
    assert (draw_mask(2, 8, gen2) == mask).all()
    assert not (draw_mask(2, 8, gen2) == mask).all()


def test_importance_recovers_planted_signal():
    torch.manual_seed(7)
    S, L, E = 4000, 2, 8
    masks = torch.zeros(S, L, E, dtype=torch.bool)
    gen = torch.Generator().manual_seed(1)
    for s in range(0, S, 2):
        m = draw_mask(L, E, gen)
        masks[s], masks[s + 1] = m, ~m
    # expert (0, 3) matters a lot, (1, 5) a little; others don't
    losses = (2.0 + 1.0 * masks[:, 0, 3].float() + 0.3 * masks[:, 1, 5].float()
              + 0.05 * torch.randn(S))
    imp = importance_from_samples(masks, losses)
    # E[imp_e] = true_e * E/(E-1) - S_layer/(E-1): per-layer ranking exact; with a
    # single planted effect per layer the causal expert itself is unbiased and the
    # rest sit near the -S/(E-1) leak (-1/7 and -0.3/7 here)
    assert imp[0].argmax() == 3 and imp[1].argmax() == 5
    assert imp[0, 3] == pytest.approx(1.0, abs=0.05)
    assert imp[1, 5] == pytest.approx(0.3, abs=0.05)
    others = imp[0, torch.arange(E) != 3]
    assert others.mean() == pytest.approx(-1.0 / 7, abs=0.05)
    assert (others - others.mean()).abs().max() < 0.075


def test_write_keep_matches_selection_schema(tmp_path):
    imp = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.0, 0.3, 0.8, 0.4],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    out = tmp_path / "keep_ablation_0.25.json"
    write_keep(imp, [0, 1], 0.25, out, {"model": "tiny"})
    k = json.loads(out.read_text())
    ref = json.loads(open("runs/qwen36/keep_reap_0.25.json").read())
    assert set(k) == set(ref) and set(k["info"]) == set(ref["info"])
    assert k["info"]["keep_per_layer"] == 2
    assert k["keep"]["0"] == [2, 6] and k["keep"]["1"] == [6, 7]
