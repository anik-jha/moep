"""Gemma4 adapter invariants (CLAUDE.md testing rules), mirrored from test_surgery /
test_profiler for the second model family:
1. prune-nothing reproduces base logits exactly;
2. pruned model == original whose router is masked to the kept experts, including the
   per_expert_scale applied after top-k renormalization;
3. router rows (proj + per_expert_scale) and expert slabs are byte-identical for kept ids;
4. adapter classification covers the real 26B checkpoint index when present;
5. profiler stats match a brute-force recount, with prob_mass captured from the router.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from moep.adapters import get_adapter
from moep.profiler import profile
from moep.scoring import select
from moep.surgery import prune_checkpoint

TINY_KW = dict(
    hidden_size=64,
    num_hidden_layers=2,
    layer_types=["full_attention", "full_attention"],
    num_experts=8,
    top_k_experts=2,
    moe_intermediate_size=32,
    intermediate_size=48,
    enable_moe_block=True,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=16,
    vocab_size=512,
    max_position_embeddings=256,
    # the real checkpoint ties lm_head to embed_tokens and ships no lm_head tensor;
    # the tiny model must exercise the same load path (caught 2026-07-14: an untied
    # tiny model masked a load_streaming failure on the real 26B checkpoint)
    tie_word_embeddings=True,
    dtype="float32",
)

ALL = list(range(8))
KEEP = {0: [0, 2, 5, 7], 1: [1, 2, 3, 6]}


@pytest.fixture(scope="session")
def tiny_gemma_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_KW))
    # per_expert_scale initializes to ones; give it structure so the masked-router
    # equivalence test would catch a surgery path that forgets to slice it.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.router.per_expert_scale.copy_(torch.linspace(0.5, 1.5, 8))
    model.eval()
    path = tmp_path_factory.mktemp("tiny_gemma") / "base"
    model.save_pretrained(path)
    return path, model


@pytest.fixture(scope="session")
def gemma_adapter(tiny_gemma_ckpt):
    path, _ = tiny_gemma_ckpt
    return get_adapter(json.loads((path / "config.json").read_text()))


@pytest.fixture()
def sample_ids():
    torch.manual_seed(123)
    return torch.randint(0, 512, (2, 48))


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def _prune_and_load(tiny_gemma_ckpt, gemma_adapter, tmp_path, keep):
    src, _ = tiny_gemma_ckpt
    out = tmp_path / "pruned"
    report = prune_checkpoint(src, out, keep, gemma_adapter, log=lambda *_: None)
    model = Gemma4ForCausalLM.from_pretrained(out)
    model.eval()
    return model, out, report


def test_keep_all_identical(tiny_gemma_ckpt, gemma_adapter, tmp_path, sample_ids):
    _, base = tiny_gemma_ckpt
    pruned, _, report = _prune_and_load(tiny_gemma_ckpt, gemma_adapter, tmp_path, {0: ALL, 1: ALL})
    assert report["kept_experts"] == 8
    assert torch.equal(_logits(base, sample_ids), _logits(pruned, sample_ids))


def test_masked_router_equivalence(tiny_gemma_ckpt, gemma_adapter, tmp_path, sample_ids):
    """Zeroing dropped experts' softmax probabilities, re-selecting top-k, renormalizing
    over the k, then applying per_expert_scale must equal the pruned model exactly: the
    leading renormalization cancels inside the top-k renorm, and per_expert_scale is a
    per-expert multiplier that commutes with the index-select."""
    _, base = tiny_gemma_ckpt
    pruned, _, _ = _prune_and_load(tiny_gemma_ckpt, gemma_adapter, tmp_path, KEEP)

    handles = []
    for i, block in gemma_adapter.moe_blocks(base):
        dropped = torch.tensor([e for e in ALL if e not in KEEP[i]])
        top_k = gemma_adapter.top_k
        scale = block.router.per_expert_scale

        def masked(module, args, output, dropped=dropped, top_k=top_k, scale=scale):
            probs = output[0].clone()
            probs[:, dropped] = 0.0
            val, idx = torch.topk(probs, top_k, dim=-1)
            val = val / val.sum(dim=-1, keepdim=True)
            val = val * scale[idx]
            return probs, val, idx

        handles.append(block.router.register_forward_hook(masked))
    try:
        ref = _logits(base, sample_ids)
    finally:
        for h in handles:
            h.remove()

    got = _logits(pruned, sample_ids)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), f"max diff {(ref - got).abs().max()}"


def test_pruned_tensors_bytewise(tiny_gemma_ckpt, gemma_adapter, tmp_path):
    _, out, _ = _prune_and_load(tiny_gemma_ckpt, gemma_adapter, tmp_path, KEEP)
    src, _ = tiny_gemma_ckpt
    src_shard = next(Path(src).glob("*.safetensors"))
    out_shard = next(Path(out).glob("*.safetensors"))
    with safe_open(src_shard, framework="pt") as fs, safe_open(out_shard, framework="pt") as fo:
        out_names = set(fo.keys())
        for name in fs.keys():
            act = gemma_adapter.classify(name)
            if act.kind not in ("expert_fused", "router"):
                continue
            keep_ids = torch.tensor(KEEP[act.layer])
            expect = fs.get_tensor(name).index_select(0, keep_ids)
            assert name in out_names, name
            assert torch.equal(expect, fo.get_tensor(name)), name


def test_pruned_config(tiny_gemma_ckpt, gemma_adapter, tmp_path):
    _, out, _ = _prune_and_load(tiny_gemma_ckpt, gemma_adapter, tmp_path, KEEP)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["num_experts"] == 4
    assert cfg["top_k_experts"] == 2


def test_classify_kinds(gemma_adapter):
    c = gemma_adapter.classify
    assert c("model.language_model.layers.3.experts.down_proj").kind == "expert_fused"
    assert c("model.layers.3.experts.gate_up_proj").kind == "expert_fused"
    assert c("model.language_model.layers.3.router.proj.weight").kind == "router"
    assert c("model.language_model.layers.3.router.per_expert_scale").kind == "router"
    assert c("model.language_model.layers.3.router.scale").kind == "copy"
    assert c("model.language_model.layers.3.mlp.gate_proj.weight").kind == "copy"
    assert c("model.vision_tower.encoder.layers.3.mlp.gate_proj.linear.weight").kind == "copy"


REAL_INDEX = Path(__file__).resolve().parents[1] / "models/gemma-4-26b-a4b-it/model.safetensors.index.json"


@pytest.mark.skipif(not REAL_INDEX.exists(), reason="real checkpoint index not downloaded")
def test_classify_covers_real_index(gemma_adapter):
    names = json.loads(REAL_INDEX.read_text())["weight_map"]
    kinds = {"expert_fused": 0, "expert_item": 0, "router": 0, "drop": 0, "copy": 0}
    for n in names:
        kinds[gemma_adapter.classify(n).kind] += 1
    assert kinds["expert_fused"] == 60, kinds  # 30 layers x (gate_up_proj + down_proj)
    assert kinds["router"] == 60, kinds        # 30 layers x (proj.weight + per_expert_scale)
    assert kinds["expert_item"] == 0, kinds
    assert kinds["drop"] == 0, kinds


def test_profiler_counts_and_prob_mass(tiny_gemma_ckpt, gemma_adapter):
    _, model = tiny_gemma_ckpt
    torch.manual_seed(7)
    ids = torch.randint(0, 512, (3, 32))

    stats, layer_ids = profile(model, gemma_adapter, ids, batch_size=3, log_every=0)
    assert layer_ids == [0, 1]
    assert stats.total_tokens == 3 * 32

    captured = {}
    handles = []
    for i, block in gemma_adapter.moe_blocks(model):
        def grab(module, args, i=i):
            captured[i] = (args[0].detach().clone(), args[1].detach().clone(), args[2].detach().clone())
        handles.append(block.experts.register_forward_pre_hook(grab))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    for row, layer in enumerate(layer_ids):
        hidden, idx, w = captured[layer]
        counts = torch.bincount(idx.reshape(-1), minlength=8).float()
        assert torch.equal(stats.tensors["counts"][row], counts), f"layer {layer}"
        gate_sum = torch.zeros(8).scatter_add_(0, idx.reshape(-1), w.reshape(-1).float())
        assert torch.allclose(stats.tensors["gate_sum"][row], gate_sum, atol=1e-5)
        block = dict(gemma_adapter.moe_blocks(model))[layer]
        e = int(counts.argmax())
        pairs = (idx.reshape(-1) == e).nonzero(as_tuple=True)[0]
        toks = pairs // idx.shape[1]
        gate, up = torch.nn.functional.linear(hidden[toks], block.experts.gate_up_proj[e]).chunk(2, -1)
        f = torch.nn.functional.linear(block.experts.act_fn(gate) * up, block.experts.down_proj[e])
        norms = f.norm(dim=-1)
        assert torch.allclose(stats.tensors["norm_sum"][row, e], norms.sum(), rtol=1e-4)
        reap = (w.reshape(-1)[pairs] * norms).sum()
        assert torch.allclose(stats.tensors["reap_sum"][row, e], reap, rtol=1e-4)

    # prob_mass comes from the router hook: fp32 softmax over all experts sums to 1/token
    per_layer = stats.tensors["prob_mass"].sum(dim=-1)
    assert torch.allclose(per_layer, torch.full_like(per_layer, 96.0), atol=1e-3)


def test_streaming_loader_handles_tied_embeddings(tiny_gemma_ckpt, sample_ids):
    """A tied checkpoint has no lm_head.weight on disk; load_streaming must re-tie
    instead of reporting a missing parameter, and produce from_pretrained's logits."""
    from moep.loader import load_streaming

    path, base = tiny_gemma_ckpt
    loaded = load_streaming(path, device="cpu", log=lambda *_: None)
    assert torch.equal(_logits(base, sample_ids), _logits(loaded, sample_ids))


def test_scoring_end_to_end(tiny_gemma_ckpt, gemma_adapter):
    _, model = tiny_gemma_ckpt
    ids = torch.randint(0, 512, (2, 32))
    stats, layer_ids = profile(model, gemma_adapter, ids, batch_size=2, log_every=0)
    keeps, info = select(stats.to_cpu(), "reap", 0.5)
    assert info["keep_per_layer"] == 4
    assert len(keeps) == 2 and all(len(k) == 4 for k in keeps)
