import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from moep.adapters import get_adapter

TINY_KW = dict(
    hidden_size=64,
    num_hidden_layers=2,
    layer_types=["full_attention", "full_attention"],
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=512,
    max_position_embeddings=256,
    tie_word_embeddings=False,
    dtype="float32",
)


def build_tiny(seed: int = 0) -> Qwen3_5MoeForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**TINY_KW))
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    """A saved tiny checkpoint on disk plus its loaded model (fp32, CPU, deterministic)."""
    path = tmp_path_factory.mktemp("tiny") / "base"
    model = build_tiny()
    model.save_pretrained(path)
    return path, model


@pytest.fixture(scope="session")
def tiny_adapter(tiny_ckpt):
    import json

    path, _ = tiny_ckpt
    return get_adapter(json.loads((path / "config.json").read_text()))


@pytest.fixture()
def sample_ids():
    torch.manual_seed(123)
    return torch.randint(0, 512, (2, 48))
