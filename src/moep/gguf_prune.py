"""Expert pruning directly on a GGUF file, with no dequantization or re-quantization.

Each kept expert's weights are copied byte for byte, so quantization-aware-trained or imatrix
weights survive untouched. The expert axis is the outermost axis of every per-expert tensor
(numpy axis 0), so pruning is a row selection on the router, the routing bias, and the stacked
expert tensors. Layers may keep different numbers of experts; llama.cpp architectures that read
the per-layer count from the router shape (nemotron_h_moe) load that directly, while ones with a
model-wide count (qwen4exp) need the same count in every layer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# gguf-py from the pinned llama.cpp build that loads these models (types such as NVFP4 and
# archs such as qwen4exp must match the runtime)
GGUF_PY = Path(__file__).resolve().parents[2] / "tools" / "llama.cpp-upstream" / "gguf-py"
if GGUF_PY.is_dir() and str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402

EXPERT_TENSOR = re.compile(
    r"^blk\.(\d+)\.(ffn_gate_inp\.weight|exp_probs_b\.bias|"
    r"ffn_(?:up|down|gate|gate_up)_exps\.(?:weight|scale|input_scale))$")


def _experts_used(reader: gguf.GGUFReader, arch: str) -> dict[int, int] | int:
    field = reader.fields[f"{arch}.expert_used_count"]
    val = field.contents()
    return dict(enumerate(val)) if isinstance(val, list) else int(val)


def experts_used(path: str | Path) -> dict[int, int] | int:
    """Experts routed per token: per layer when the GGUF stores an array, else one int."""
    reader = gguf.GGUFReader(str(path))
    return _experts_used(reader, reader.fields[gguf.Keys.General.ARCHITECTURE].contents())


def _router_counts(reader: gguf.GGUFReader) -> dict[int, int]:
    """Layer index -> number of experts, read from each router's shape."""
    return {int(m.group(1)): t.data.shape[0] for t in reader.tensors
            if (m := EXPERT_TENSOR.match(t.name)) and m.group(2) == "ffn_gate_inp.weight"}


def _plan(reader: gguf.GGUFReader, keep: dict[int, list[int]]) -> list[tuple[gguf.ReaderTensor, np.ndarray | None]]:
    """(tensor, kept expert indices or None for a verbatim copy) for every tensor, in file order."""
    n_expert = _router_counts(reader)
    if missing := set(keep) - set(n_expert):
        raise ValueError(f"keep names layers without a router: {sorted(missing)}")
    idx = {}
    for il, experts in keep.items():
        sel = np.array(sorted(set(experts)), dtype=np.int64)
        if sel.size == 0 or sel[0] < 0 or sel[-1] >= n_expert[il]:
            raise ValueError(f"layer {il}: kept experts must lie in [0, {n_expert[il]})")
        idx[il] = sel
    plan = []
    for t in reader.tensors:
        m = EXPERT_TENSOR.match(t.name)
        il = int(m.group(1)) if m else None
        if m and il in idx:
            if t.data.shape[0] != n_expert[il]:
                raise ValueError(f"{t.name}: axis 0 is {t.data.shape[0]}, router has {n_expert[il]} experts")
            plan.append((t, idx[il]))
            continue
        lm = re.match(r"^blk\.(\d+)\.", t.name)
        if lm and int(lm.group(1)) in idx and "exp" in t.name and t.data.shape[0] == n_expert[int(lm.group(1))] \
                and "shexp" not in t.name:
            raise ValueError(f"{t.name} looks per-expert but is not handled; refusing to copy it unpruned")
        plan.append((t, None))
    return plan


def prune_gguf(src: str | Path, dst: str | Path, keep: dict[int, list[int]], split_max_size: int = 0) -> dict:
    """Write dst with only the kept experts of each layer in keep (other layers copied verbatim).
    The model-wide expert count becomes the largest per-layer count after pruning. Returns a summary."""
    reader = gguf.GGUFReader(str(src))
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    used = _experts_used(reader, arch)
    for il, experts in keep.items():
        need = used[il] if isinstance(used, dict) else used
        if len(set(experts)) < need:
            raise ValueError(f"layer {il}: keeps {len(set(experts))} experts but routes to {need}")
    plan = _plan(reader, keep)
    new_count = max(len(set(keep[il])) if il in keep else n for il, n in _router_counts(reader).items())

    writer = gguf.GGUFWriter(str(dst), arch, split_max_size=split_max_size)
    count_key = f"{arch}.expert_count"
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith(("GGUF.", "split.")):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = new_count if field.name == count_key else field.contents()
        writer.add_key_value(field.name, val, vtype, sub_type=sub)

    def payload(t, sel):
        return t.data if sel is None else np.take(t.data, sel, axis=0)

    total = 0
    for t, sel in plan:
        shape = t.data.shape if sel is None else (sel.size, *t.data.shape[1:])
        nbytes = t.data.nbytes if sel is None else t.data.nbytes // t.data.shape[0] * sel.size
        writer.add_tensor_info(t.name, shape, t.data.dtype, nbytes, t.tensor_type)
        total += nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t, sel in plan:
        writer.write_tensor_data(payload(t, sel), tensor_endianess=reader.endianess)
    writer.close()
    return {"src": str(src), "dst": str(dst), "arch": arch, "expert_count": new_count,
            "kept_per_layer": {il: len(set(v)) for il, v in sorted(keep.items())}, "tensor_bytes": total}


def verify_pruned(src: str | Path, dst: str | Path, keep: dict[int, list[int]]) -> dict:
    """Byte-compare dst against src: kept expert rows identical, every other tensor identical,
    pruned layers carry exactly the kept count. Raises on the first mismatch."""
    a = gguf.GGUFReader(str(src))
    b = {t.name: t for t in gguf.GGUFReader(str(dst)).tensors}
    checked = 0
    for t, sel in _plan(a, keep):
        out = b.pop(t.name)
        want = t.data if sel is None else np.take(t.data, sel, axis=0)
        if out.tensor_type != t.tensor_type or out.data.shape != want.shape or not np.array_equal(out.data, want):
            raise AssertionError(f"{t.name}: pruned output differs from the source selection")
        checked += 1
    if b:
        raise AssertionError(f"unexpected tensors in output: {sorted(b)[:5]}")
    return {"tensors_checked": checked, "identical": True}
