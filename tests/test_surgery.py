"""Surgery invariants (CLAUDE.md testing rules):
1. prune-nothing reproduces base logits exactly;
2. pruned model == original model whose router is masked to the kept experts;
3. router rows and expert slabs are byte-identical for kept ids;
4. adapter classification covers the real 35B checkpoint index when present.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import Qwen3_5MoeForCausalLM

from moep.surgery import prune_checkpoint

ALL = list(range(8))
KEEP = {0: [0, 2, 5, 7], 1: [1, 2, 3, 6]}


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def _prune_and_load(tiny_ckpt, tiny_adapter, tmp_path, keep):
    src, _ = tiny_ckpt
    out = tmp_path / "pruned"
    report = prune_checkpoint(src, out, keep, tiny_adapter, log=lambda *_: None)
    model = Qwen3_5MoeForCausalLM.from_pretrained(out)
    model.eval()
    return model, out, report


def test_keep_all_identical(tiny_ckpt, tiny_adapter, tmp_path, sample_ids):
    _, base = tiny_ckpt
    pruned, _, report = _prune_and_load(tiny_ckpt, tiny_adapter, tmp_path, {0: ALL, 1: ALL})
    assert report["kept_experts"] == 8
    assert torch.equal(_logits(base, sample_ids), _logits(pruned, sample_ids))


def test_masked_router_equivalence(tiny_ckpt, tiny_adapter, tmp_path, sample_ids):
    """Pruning must equal restricting the original router to the kept experts:
    fp32 softmax with -inf on dropped ids is exactly the softmax over the kept set."""
    _, base = tiny_ckpt
    pruned, _, _ = _prune_and_load(tiny_ckpt, tiny_adapter, tmp_path, KEEP)

    handles = []
    for i, block in tiny_adapter.moe_blocks(base):
        dropped = torch.tensor([e for e in ALL if e not in KEEP[i]])
        top_k = block.gate.top_k

        def masked(module, args, output, dropped=dropped, top_k=top_k):
            logits = output[0].clone()
            logits[:, dropped] = float("-inf")
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            val, idx = torch.topk(probs, top_k, dim=-1)
            val = val / val.sum(dim=-1, keepdim=True)
            return logits, val.to(logits.dtype), idx

        handles.append(block.gate.register_forward_hook(masked))
    try:
        ref = _logits(base, sample_ids)
    finally:
        for h in handles:
            h.remove()

    got = _logits(pruned, sample_ids)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), f"max diff {(ref - got).abs().max()}"


def test_pruned_tensors_bytewise(tiny_ckpt, tiny_adapter, tmp_path):
    """transformers 5.x saves the legacy per-expert layout, so the tiny checkpoint
    exercises the drop+renumber path; router rows exercise the slice path."""
    src, _ = tiny_ckpt
    _, out, report = _prune_and_load(tiny_ckpt, tiny_adapter, tmp_path, KEEP)
    assert report["renamed"] == (4 + 4) * 3 and report["sliced"] == 2  # experts x proj, routers
    with safe_open(src / "model.safetensors", framework="pt") as fs, \
         safe_open(out / "model.safetensors", framework="pt") as fo:
        out_names = set(fo.keys())
        for layer, kept in KEEP.items():
            gate_old = fs.get_tensor(f"model.layers.{layer}.mlp.gate.weight")
            gate_new = fo.get_tensor(f"model.layers.{layer}.mlp.gate.weight")
            assert gate_new.shape[0] == len(kept)
            for new_i, old_i in enumerate(kept):
                assert torch.equal(gate_new[new_i], gate_old[old_i])
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    old = fs.get_tensor(f"model.layers.{layer}.mlp.experts.{old_i}.{proj}.weight")
                    new = fo.get_tensor(f"model.layers.{layer}.mlp.experts.{new_i}.{proj}.weight")
                    assert torch.equal(old, new)
            for e in range(len(kept), 8):  # renumbered ids are contiguous 0..kept-1
                assert not any(f".layers.{layer}.mlp.experts.{e}." in n for n in out_names)


def test_pruned_config_and_map(tiny_ckpt, tiny_adapter, tmp_path):
    _, out, _ = _prune_and_load(tiny_ckpt, tiny_adapter, tmp_path, KEEP)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["num_experts"] == 4
    pmap = json.loads((out / "pruning_map.json").read_text())
    assert pmap["keep"]["0"] == KEEP[0] and pmap["original_num_experts"] == 8


def test_classify_kinds(tiny_adapter):
    c = tiny_adapter.classify
    assert c("mtp.layers.0.mlp.experts.gate_up_proj").kind == "drop"
    assert c("model.language_model.layers.3.mlp.experts.down_proj").kind == "expert_fused"
    a = c("model.layers.3.mlp.experts.17.up_proj.weight")
    assert (a.kind, a.layer, a.expert) == ("expert_item", 3, 17)
    assert c("model.language_model.layers.3.mlp.gate.weight").kind == "router"
    assert c("model.language_model.layers.3.mlp.shared_expert.gate_proj.weight").kind == "copy"
    assert c("model.language_model.layers.3.mlp.shared_expert_gate.weight").kind == "copy"
    assert tiny_adapter.renumber_item("model.layers.3.mlp.experts.17.up_proj.weight", 2) == \
        "model.layers.3.mlp.experts.2.up_proj.weight"


REAL_INDEX = Path(__file__).resolve().parents[1] / "models/qwen3.6-35b-a3b/model.safetensors.index.json"


@pytest.mark.skipif(not REAL_INDEX.exists(), reason="real checkpoint index not downloaded")
def test_classify_covers_real_index(tiny_adapter):
    names = json.loads(REAL_INDEX.read_text())["weight_map"]
    kinds = {"expert_fused": 0, "expert_item": 0, "router": 0, "drop": 0, "copy": 0}
    for n in names:
        kinds[tiny_adapter.classify(n).kind] += 1
    assert kinds["expert_fused"] == 80, kinds  # 40 layers x (gate_up_proj + down_proj)
    assert kinds["expert_item"] == 0, kinds
    assert kinds["router"] == 40, kinds
    assert kinds["drop"] >= 15, kinds          # all mtp.* tensors
