"""Memory-deterministic checkpoint loader for unified-memory machines.

The stock from_pretrained path peaked near 2x model size on the DGX Spark's unified
pool (shard page cache + device copies count against the same 119 GB) and the kernel
OOM-killed it at ~64% of the 72 GB checkpoint. This loader:
  1. builds the model with meta parameters (real buffers, so rotary tables etc. exist),
  2. streams each safetensors shard directly to the target device and installs tensors
     with load_state_dict(assign=True),
  3. drops the shard's page cache (posix_fadvise DONTNEED) before moving on.
Peak memory ~= final weights + one shard. The same path scales to disk-streaming
calibration for checkpoints larger than memory (Phase 3).
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file


def _fuse_legacy_experts(legacy: dict[str, torch.Tensor], legacy_re: re.Pattern) -> dict[str, torch.Tensor]:
    by_layer: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for name, t in legacy.items():
        prefix, idx, proj = legacy_re.match(name).groups()
        by_layer.setdefault(prefix, {}).setdefault(int(idx), {})[proj] = t
    fused: dict[str, torch.Tensor] = {}
    for prefix, experts in by_layer.items():
        ids = sorted(experts)
        if ids != list(range(len(ids))):
            raise RuntimeError(f"{prefix}: non-contiguous expert ids {ids[:5]}...")
        fused[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]], dim=0) for i in ids])
        fused[f"{prefix}.down_proj"] = torch.stack([experts[i]["down_proj"] for i in ids])
    return fused


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    return [model_dir / "model.safetensors"]


def _drop_page_cache(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def load_streaming(model_dir: str | Path, device: str = "cuda:0",
                   attn_implementation: str = "sdpa", log=print):
    import transformers
    from transformers import AutoConfig

    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = attn_implementation
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    if arch is None or not hasattr(transformers, arch):
        raise ValueError(f"cannot resolve architecture class from config: {arch!r}")
    cls = getattr(transformers, arch)

    with init_empty_weights(include_buffers=False):
        model = cls(cfg)
    model.eval()

    unexpected: list[str] = []
    legacy: dict[str, torch.Tensor] = {}
    legacy_re = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    for shard in _shards(model_dir):
        sd = load_file(str(shard), device=device)
        for k in [k for k in sd if legacy_re.match(k)]:
            legacy[k] = sd.pop(k)
        result = model.load_state_dict(sd, strict=False, assign=True)
        unexpected += result.unexpected_keys
        del sd
        gc.collect()
        _drop_page_cache(shard)
        log(f"loader: {shard.name} installed")

    if legacy:
        # transformers 5.x save_pretrained writes the legacy per-expert layout even though
        # the module holds fused 3D params; from_pretrained converts on load and so must we.
        # Experts of one layer can span shards, hence the staging dict.
        fused = _fuse_legacy_experts(legacy, legacy_re)
        result = model.load_state_dict(fused, strict=False, assign=True)
        unexpected += result.unexpected_keys
        log(f"loader: fused {len(legacy)} legacy expert tensors -> {len(fused)}")
        del legacy, fused
        gc.collect()

    # Tied-embedding checkpoints (gemma4: lm_head shares embed_tokens) ship no lm_head
    # tensor; re-tie so the shared weight replaces the meta placeholder before the check.
    model.tie_weights()

    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(f"{len(meta_left)} parameters missing from checkpoint, e.g. {meta_left[:5]}")
    for _, b in model.named_buffers():
        if b.is_meta:
            raise RuntimeError("meta buffer survived load; include_buffers assumption broke")
    model.to(device)  # buffers were created on CPU; params are already on device
    if unexpected:
        log(f"loader: {len(unexpected)} unexpected keys ignored (e.g. {sorted(unexpected)[:3]})")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return model


def load_offloaded(model_dir: str | Path, max_gpu_gib: int, offload_dir: str | Path,
                   attn_implementation: str = "sdpa", log=print):
    """For checkpoints larger than the memory budget (e.g. GLM-4.5-Air FP8, 112.6 GB on a
    119 GB box). Uses HF from_pretrained with an accelerate device_map so it:
      - handles compressed-tensors FP8 dequant/module-setup (which the manual meta-init
        streaming path above cannot), and
      - caps resident weights at max_gpu_gib, spilling the overflow to NVMe (offload_dir),
        swapped in per-forward. On this unified-memory box CPU offload is pointless (same
        pool), so we give the map only the GPU budget plus a disk offload folder.
    Profiling reads expert weights inside a forward hook; accelerate's align-devices hook
    moves each block on-GPU before our hook fires, so module weights are materialized then.
    Slower than a resident load (disk swaps) but correct at full checkpoint precision, and
    surgery still operates on the original FP8 files regardless of how we profiled."""
    from transformers import AutoModelForCausalLM

    offload_dir = Path(offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)
    log(f"loader(offload): from_pretrained device_map cap={max_gpu_gib}GiB, offload->{offload_dir}")
    # On this unified-memory box CPU RAM and GPU RAM are the SAME pool, so a CPU offload
    # buffer does not free anything and only competes with the load-time dequant peak. Cap
    # both GPU and CPU tightly so accelerate is forced to spill the remainder to NVMe.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        max_memory={0: f"{max_gpu_gib}GiB", "cpu": "2GiB"},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        offload_buffers=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    n_off = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    log(f"loader(offload): loaded; {n_off} params disk-resident (materialized per-forward)")
    return model
