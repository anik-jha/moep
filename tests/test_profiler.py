"""Profiler consistency: accumulated stats must match a brute-force recount taken from
the router's own outputs, and expert-output norms must match direct recomputation."""

import torch

from moep.profiler import pack_token_stream, profile


def _ground_truth(model, adapter, ids):
    """Capture (indices, weights, hidden) per MoE block straight from module IO."""
    captured = {}
    handles = []
    for i, block in adapter.moe_blocks(model):
        def grab(module, args, i=i):
            captured[i] = (args[0].detach().clone(), args[1].detach().clone(), args[2].detach().clone())
        handles.append(block.experts.register_forward_pre_hook(grab))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return captured


def test_counts_and_norms_match_brute_force(tiny_ckpt, tiny_adapter):
    _, model = tiny_ckpt
    torch.manual_seed(7)
    ids = torch.randint(0, 512, (3, 32))

    stats, layer_ids = profile(model, tiny_adapter, ids, batch_size=3, log_every=0)
    truth = _ground_truth(model, tiny_adapter, ids)

    assert stats.total_tokens == 3 * 32
    for row, layer in enumerate(layer_ids):
        hidden, idx, w = truth[layer]
        E = tiny_adapter.num_experts
        counts = torch.bincount(idx.reshape(-1), minlength=E).float()
        assert torch.equal(stats.tensors["counts"][row], counts), f"layer {layer}"
        gate_sum = torch.zeros(E).scatter_add_(0, idx.reshape(-1), w.reshape(-1).float())
        assert torch.allclose(stats.tensors["gate_sum"][row], gate_sum, atol=1e-5)

        block = dict(tiny_adapter.moe_blocks(model))[layer]
        e = int(counts.argmax())
        pairs = (idx.reshape(-1) == e).nonzero(as_tuple=True)[0]
        toks = pairs // idx.shape[1]
        gate, up = torch.nn.functional.linear(hidden[toks], block.experts.gate_up_proj[e]).chunk(2, -1)
        f = torch.nn.functional.linear(block.experts.act_fn(gate) * up, block.experts.down_proj[e])
        norms = f.norm(dim=-1)
        assert torch.allclose(stats.tensors["norm_sum"][row, e], norms.sum(), rtol=1e-4)
        reap = (w.reshape(-1)[pairs] * norms).sum()
        assert torch.allclose(stats.tensors["reap_sum"][row, e], reap, rtol=1e-4)


def test_prob_mass_sums_to_tokens(tiny_ckpt, tiny_adapter):
    _, model = tiny_ckpt
    ids = torch.randint(0, 512, (2, 16))
    stats, _ = profile(model, tiny_adapter, ids, batch_size=2, log_every=0)
    per_layer = stats.tensors["prob_mass"].sum(dim=-1)
    assert torch.allclose(per_layer, torch.full_like(per_layer, 32.0), atol=1e-3)


def test_pack_token_stream():
    packed = pack_token_stream([[1, 2, 3], [4, 5], [6, 7, 8, 9]], seq_len=4, eos_id=0)
    assert packed.shape == (3, 4)
    assert packed.reshape(-1).tolist() == [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 9, 0]
