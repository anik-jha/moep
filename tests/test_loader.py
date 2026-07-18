"""The streaming loader must be functionally identical to from_pretrained."""

import torch
from transformers import Qwen3_5MoeForCausalLM

from moep.loader import load_streaming


def test_streaming_loader_matches_from_pretrained(tiny_ckpt, sample_ids):
    path, base = tiny_ckpt
    via_hf = Qwen3_5MoeForCausalLM.from_pretrained(path)
    via_stream = load_streaming(path, device="cpu", log=lambda *_: None)
    via_hf.eval(), via_stream.eval()
    with torch.no_grad():
        a = via_hf(input_ids=sample_ids).logits
        b = via_stream(input_ids=sample_ids).logits
    assert torch.equal(a, b)
    assert not any(p.is_meta for p in via_stream.parameters())
