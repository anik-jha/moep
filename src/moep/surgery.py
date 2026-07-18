"""Checkpoint surgery: remove experts from safetensors shards without loading the model.

Streams shard by shard (peak memory ~= largest single tensor), so the same code path
scales from the 35B proving ground to 744B-class checkpoints. Handles both expert
serialization layouts (see adapters): fused 3D tensors are index_select'd on dim 0;
legacy per-expert tensors are dropped or renamed to their new contiguous index. Router
weights are row-sliced. Drop-listed tensors (mtp.*) are omitted. Everything else is
copied verbatim. After writing, a verification pass re-opens both checkpoints and
compares sampled kept-expert rows byte-for-byte.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AUX_PATTERNS = ("tokenizer*", "*.jinja", "generation_config.json", "preprocessor_config.json",
                "processor_config.json", "special_tokens_map.json", "merges.txt", "vocab.json",
                "video_preprocessor_config.json", "LICENSE*")


def _weight_map(src: Path) -> dict[str, str]:
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
        return {name: "model.safetensors" for name in f.keys()}


def prune_checkpoint(src: str | Path, out: str | Path, keep: dict[int, list[int]],
                     adapter, log=print) -> dict:
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weight_map = _weight_map(src)

    keep_counts = {len(v) for v in keep.values()}
    if len(keep_counts) != 1:
        raise ValueError(f"non-uniform keep counts not supported yet: {sorted(keep_counts)}")
    kept_count = next(iter(keep_counts))
    for layer, ids in keep.items():
        if ids != sorted(set(ids)):
            raise ValueError(f"layer {layer}: keep ids must be sorted and unique")
    keep_idx = {layer: torch.tensor(ids, dtype=torch.long) for layer, ids in keep.items()}
    new_id = {layer: {old: new for new, old in enumerate(ids)} for layer, ids in keep.items()}

    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    new_map: dict[str, str] = {}
    total_size = 0
    n = {"sliced": 0, "renamed": 0, "copied": 0, "dropped": 0}

    for shard in sorted(shards):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in shards[shard]:
                act = adapter.classify(name)
                if act.kind == "drop":
                    n["dropped"] += 1
                    continue
                if act.kind in ("expert_fused", "expert_item", "router") and act.layer not in keep_idx:
                    raise KeyError(f"{name}: no keep set for layer {act.layer}")
                if act.kind == "expert_item":
                    if act.expert not in new_id[act.layer]:
                        n["dropped"] += 1
                        continue
                    tensors[adapter.renumber_item(name, new_id[act.layer][act.expert])] = f.get_tensor(name)
                    n["renamed"] += 1
                elif act.kind in ("expert_fused", "router"):
                    t = f.get_tensor(name)
                    tensors[name] = t.index_select(0, keep_idx[act.layer]).contiguous()
                    n["sliced"] += 1
                else:
                    tensors[name] = f.get_tensor(name)
                    n["copied"] += 1
        if tensors:
            save_file(tensors, str(out / shard), metadata={"format": "pt"})
            for name, t in tensors.items():
                new_map[name] = shard
                total_size += t.numel() * t.element_size()
            log(f"surgery: wrote {shard} ({len(tensors)} tensors)")
        del tensors

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": new_map}, indent=2))
    (out / "config.json").write_text(json.dumps(adapter.pruned_config(kept_count), indent=2))
    (out / "pruning_map.json").write_text(json.dumps(
        {"keep": {str(layer): ids for layer, ids in keep.items()},
         "original_num_experts": adapter.num_experts, "kept_num_experts": kept_count}, indent=2))
    for pat in AUX_PATTERNS:
        for p in src.glob(pat):
            shutil.copy2(p, out / p.name)

    checks = verify_prune(src, out, keep, adapter)
    bad = [c for c in checks if not c["equal"]]
    if bad:
        raise RuntimeError(f"surgery verification failed: {bad[:3]}")

    report = n | {"kept_experts": kept_count, "original_experts": adapter.num_experts,
                  "out_bytes": total_size, "row_checks": len(checks)}
    log(f"surgery: done {report}")
    return report


def verify_prune(src: Path, out: Path, keep: dict[int, list[int]], adapter,
                 samples_per_layer: int = 2) -> list[dict]:
    """Compare sampled kept experts byte-for-byte between source and pruned checkpoints."""
    src_map, out_map = _weight_map(Path(src)), _weight_map(Path(out))
    checks = []

    def row(root: Path, wmap: dict, name: str, i: int) -> torch.Tensor:
        # get_slice reads only the requested row from the mmap, so verifying the 35B+
        # checkpoints costs MBs of IO, not a re-read of every expert slab.
        with safe_open(Path(root) / wmap[name], framework="pt", device="cpu") as f:
            return f.get_slice(name)[i]

    def full(root: Path, wmap: dict, name: str) -> torch.Tensor:
        with safe_open(Path(root) / wmap[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    for name in src_map:
        action = adapter.classify(name)
        if action.kind not in ("expert_fused", "router"):
            continue
        ids = keep[action.layer]
        for pos in sorted({0, len(ids) // 2, len(ids) - 1}):
            checks.append({"tensor": name, "old_row": ids[pos], "new_row": pos,
                           "equal": bool(torch.equal(row(src, src_map, name, ids[pos]),
                                                     row(out, out_map, name, pos)))})
    for layer, ids in keep.items():
        for pos in range(min(samples_per_layer, len(ids))):
            old_names = [s for s in src_map
                         if (a := adapter.classify(s)).kind == "expert_item"
                         and a.layer == layer and a.expert == ids[pos]]
            for old_name in old_names:
                new_name = adapter.renumber_item(old_name, pos)
                checks.append({"tensor": new_name, "old_row": ids[pos], "new_row": pos,
                               "equal": bool(torch.equal(full(src, src_map, old_name),
                                                         full(out, out_map, new_name)))})
    return checks
