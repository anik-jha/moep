"""Architecture adapters: where experts live, what to slice, how the router behaves.

Every constant here was read from downloaded artifacts or installed modeling source
(see CLAUDE.md anti-hallucination rules), not from memory:
- qwen3_5_moe: models/qwen3.6-35b-a3b/{config.json, model.safetensors.index.json} and
  transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py (v5.13.1). Router does fp32
  softmax over ALL experts, then top-k, then renormalizes over the selected k. Experts are
  fused 3D params [E, out, in]: gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter];
  router weight is mlp.gate.weight [E, hidden]. MTP tensors (mtp.*) are ignored by the HF
  class at load time and dropped from pruned exports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TensorAction:
    kind: str  # "expert_fused" | "expert_item" | "router" | "drop" | "copy"
    layer: int | None = None
    expert: int | None = None


class Qwen3_5MoeAdapter:
    """Two serialization layouts exist for the same architecture:
    - fused (Hub checkpoint): experts.gate_up_proj / experts.down_proj, [E, out, in]
    - legacy per-expert (transformers 5.x save_pretrained): experts.<i>.{gate,up,down}_proj.weight
    Surgery slices dim 0 in the first case, drops+renumbers tensor names in the second."""

    model_types = ("qwen3_5_moe", "qwen3_5_moe_text")
    router_prob = "softmax"
    _expert_fused_re = re.compile(r"^(?:model\.(?:language_model\.)?)layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj)$")
    _expert_item_re = re.compile(r"^(?:model\.(?:language_model\.)?)layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate_proj|up_proj|down_proj)\.weight$")
    _router_re = re.compile(r"^(?:model\.(?:language_model\.)?)layers\.(\d+)\.mlp\.gate\.weight$")
    _drop_res = (re.compile(r"^mtp\."),)

    def __init__(self, config: dict):
        self.config = config
        tc = config.get("text_config", config)
        self.num_experts: int = tc["num_experts"]
        self.top_k: int = tc["num_experts_per_tok"]
        self.num_layers: int = tc["num_hidden_layers"]
        self.hidden_size: int = tc["hidden_size"]
        self.moe_intermediate_size: int = tc["moe_intermediate_size"]

    # ---- loaded-model access (profiling) ----
    def text_layers(self, model):
        inner = model.model
        return getattr(inner, "language_model", inner).layers

    def moe_blocks(self, model):
        """Yield (layer_idx, block) for each sparse-MoE block. block.gate is the router
        (weight [E, hidden]); block.experts holds fused gate_up_proj/down_proj/act_fn."""
        for i, layer in enumerate(self.text_layers(model)):
            if hasattr(layer.mlp, "experts"):
                yield i, layer.mlp

    # ---- checkpoint-tensor access (surgery; no model load) ----
    def classify(self, name: str) -> TensorAction:
        for r in self._drop_res:
            if r.match(name):
                return TensorAction("drop")
        m = self._expert_fused_re.match(name)
        if m:
            return TensorAction("expert_fused", int(m.group(1)))
        m = self._expert_item_re.match(name)
        if m:
            return TensorAction("expert_item", int(m.group(1)), int(m.group(2)))
        m = self._router_re.match(name)
        if m:
            return TensorAction("router", int(m.group(1)))
        return TensorAction("copy")

    @staticmethod
    def renumber_item(name: str, new_id: int) -> str:
        return re.sub(r"\.experts\.\d+\.", f".experts.{new_id}.", name, count=1)

    def pruned_config(self, num_kept: int) -> dict:
        cfg = json.loads(json.dumps(self.config))  # deep copy
        tc = cfg.get("text_config", cfg)
        tc["num_experts"] = num_kept
        if tc.get("mtp_num_hidden_layers"):
            tc["mtp_num_hidden_layers"] = 0  # mtp.* tensors are dropped
        return cfg


class Glm4MoeAdapter:
    """GLM-4.5-Air (glm4_moe). Verified 2026-07-13 from models/glm-4.5-air-fp8/
    {config.json, safetensors index} and transformers/models/glm4_moe/modeling_glm4_moe.py.
    Experts are per-item and FP8: layers.N.mlp.experts.M.{gate,up,down}_proj.{weight,weight_scale}.
    Router is DeepSeek-style: mlp.gate.weight [E,H] plus mlp.gate.e_score_correction_bias [E],
    BOTH sliced on the expert dim. Layer 0 is dense (first_k_dense_replace=1, mlp.{gate,up,
    down}_proj, no experts) and is copied. The trailing MTP layer (index == num_hidden_layers)
    carries its own experts and is dropped. n_group=1/topk_group=1 so per-layer pruning is
    unconstrained. Surgery reuses the expert_item drop+renumber path; weight_scale rides along
    because the item regex matches .weight and .weight_scale identically."""

    model_types = ("glm4_moe",)
    router_prob = "sigmoid"
    _expert_item_re = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate_proj|up_proj|down_proj)\.(?:weight|weight_scale)$")
    _router_re = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(?:weight|e_score_correction_bias)$")

    def __init__(self, config: dict):
        self.config = config
        tc = config.get("text_config", config)
        self.num_experts: int = tc["n_routed_experts"]
        self.top_k: int = tc["num_experts_per_tok"]
        self.num_layers: int = tc["num_hidden_layers"]
        self.hidden_size: int = tc["hidden_size"]
        self.moe_intermediate_size: int = tc["moe_intermediate_size"]
        self.first_dense: int = tc.get("first_k_dense_replace", 0)
        self._mtp_layer = self.num_layers  # trailing extra layer index

    def text_layers(self, model):
        inner = model.model
        return getattr(inner, "language_model", inner).layers

    def moe_blocks(self, model):
        for i, layer in enumerate(self.text_layers(model)):
            if hasattr(layer.mlp, "experts"):
                yield i, layer.mlp

    def classify(self, name: str) -> TensorAction:
        m = re.match(r"^model\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) == self._mtp_layer:
            return TensorAction("drop")
        m = self._expert_item_re.match(name)
        if m:
            return TensorAction("expert_item", int(m.group(1)), int(m.group(2)))
        m = self._router_re.match(name)
        if m:
            return TensorAction("router", int(m.group(1)))
        return TensorAction("copy")

    @staticmethod
    def renumber_item(name: str, new_id: int) -> str:
        return re.sub(r"\.experts\.\d+\.", f".experts.{new_id}.", name, count=1)

    def pruned_config(self, num_kept: int) -> dict:
        cfg = json.loads(json.dumps(self.config))
        tc = cfg.get("text_config", cfg)
        tc["n_routed_experts"] = num_kept
        for mtp_key in ("num_nextn_predict_layers", "num_mtp_layers"):
            if tc.get(mtp_key):
                tc[mtp_key] = 0
        return cfg


class Gemma4MoeAdapter:
    """Gemma-4-26B-A4B (gemma4 / gemma4_text). Verified 2026-07-14 from
    models/gemma-4-26b-a4b-it/{config.json, model.safetensors.index.json} and
    transformers/models/gemma4/modeling_gemma4.py (v5.13.1).
    Experts are fused 3D like qwen3_5_moe but live on the layer, not under .mlp
    (layers.N.experts.{gate_up_proj,down_proj}); the per-layer dense mlp.* is a parallel
    shared path combined additively with the MoE output, and is copied untouched.
    Router = layers.N.router: proj.weight [E, hidden] and per_expert_scale [E] are BOTH
    sliced on the expert dim; router.scale is [hidden] and copied. The router does fp32
    softmax over all experts, top-k, renormalizes over k, then multiplies by
    per_expert_scale[idx] (applied gates need not sum to 1). It also reads the pre-MLP
    residual through its own norm+scale, NOT the experts' normed input, so full-softmax
    probabilities cannot be recomputed from the experts hook input; attach_prob_hook
    captures them from the router module itself."""

    model_types = ("gemma4", "gemma4_text")
    router_prob = "softmax"
    _expert_fused_re = re.compile(r"^(?:model\.(?:language_model\.)?)layers\.(\d+)\.experts\.(?:gate_up_proj|down_proj)$")
    _router_re = re.compile(r"^(?:model\.(?:language_model\.)?)layers\.(\d+)\.router\.(?:proj\.weight|per_expert_scale)$")

    def __init__(self, config: dict):
        self.config = config
        tc = config.get("text_config", config)
        self.num_experts: int = tc["num_experts"]
        self.top_k: int = tc["top_k_experts"]
        self.num_layers: int = tc["num_hidden_layers"]
        self.hidden_size: int = tc["hidden_size"]
        self.moe_intermediate_size: int = tc["moe_intermediate_size"]

    # ---- loaded-model access (profiling) ----
    def text_layers(self, model):
        inner = model.model
        return getattr(inner, "language_model", inner).layers

    def moe_blocks(self, model):
        """Yield (layer_idx, layer): router and fused experts sit on the decoder layer."""
        for i, layer in enumerate(self.text_layers(model)):
            if getattr(layer, "enable_moe_block", False):
                yield i, layer

    def attach_prob_hook(self, block, stats, row):
        """Accumulate full-softmax probability mass from Gemma4TextRouter's first output
        (fp32, normalized over all experts, computed on the router's own normed input)."""
        def hook(module, args, output, stats=stats, row=row):
            stats.tensors["prob_mass"][row] += output[0].detach().float().sum(dim=0)
        return block.router.register_forward_hook(hook)

    # ---- checkpoint-tensor access (surgery; no model load) ----
    def classify(self, name: str) -> TensorAction:
        m = self._expert_fused_re.match(name)
        if m:
            return TensorAction("expert_fused", int(m.group(1)))
        m = self._router_re.match(name)
        if m:
            return TensorAction("router", int(m.group(1)))
        return TensorAction("copy")

    @staticmethod
    def renumber_item(name: str, new_id: int) -> str:
        raise NotImplementedError("gemma4 checkpoints store fused 3D experts only")

    def pruned_config(self, num_kept: int) -> dict:
        cfg = json.loads(json.dumps(self.config))
        tc = cfg.get("text_config", cfg)
        tc["num_experts"] = num_kept
        return cfg


_ADAPTERS = [Qwen3_5MoeAdapter, Glm4MoeAdapter, Gemma4MoeAdapter]


def get_adapter(config: dict):
    mt = config.get("model_type")
    for cls in _ADAPTERS:
        if mt in cls.model_types:
            return cls(config)
    raise ValueError(
        f"No adapter for model_type={mt!r}. Add one to moep/adapters.py after verifying "
        "the architecture from its config.json + safetensors index + installed modeling source."
    )
