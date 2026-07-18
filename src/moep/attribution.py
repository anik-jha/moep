"""D0 failure attribution: does the base model route base-solves/reap25-fails items
into the experts that reap25 pruned, more than it does on items reap25 still solves?

Item sets come from E1's turn-1 (single-shot) outcomes: FAIL = base_q8 t1 pass and
reap25_q8 t1 fail; CTRL = both pass. For each item, the BASE model is teacher-forced
over its own turn-1 chat context (the evalplus user message plus base's sanitized
solution re-wrapped in a code fence, an identical approximation for both groups), and
per layer the fraction of top-k routing mass landing on pruned experts is accumulated
over the solution tokens only (the generation-time positions). Group difference gets
an unpaired bootstrap CI; items are independent between groups, so the paired design
used for E1's model-vs-model deltas does not apply here.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import torch

DATASETS = ("humaneval", "mbpp")


def _t1_status(run_dir: Path, tag: str, dataset: str) -> dict[str, bool]:
    tr = json.loads((run_dir / "evalplus" / tag / f"{dataset}-repair" /
                     "transitions.json").read_text())
    return {tid: v["t1"] == "pass" for tid, v in tr["per_task"].items()}


def build_item_sets(run_dir: str | Path, base_tag: str = "base_q8",
                    pruned_tag: str = "reap25_q8") -> dict[str, dict[str, list[str]]]:
    """{dataset: {"fail": [task_id...], "ctrl": [task_id...]}} from E1 transitions."""
    run_dir = Path(run_dir)
    sets: dict[str, dict[str, list[str]]] = {}
    for ds in DATASETS:
        base = _t1_status(run_dir, base_tag, ds)
        pruned = _t1_status(run_dir, pruned_tag, ds)
        sets[ds] = {
            "fail": sorted(t for t in base if base[t] and not pruned[t]),
            "ctrl": sorted(t for t in base if base[t] and pruned[t]),
        }
    return sets


def pruned_mask(keep_path: str | Path) -> torch.Tensor:
    """Bool [num_layers, num_experts], True where the expert was PRUNED by the keep set."""
    k = json.loads(Path(keep_path).read_text())
    num_experts = k["info"]["num_experts"]
    layer_ids = k["layer_ids"]
    mask = torch.ones(len(layer_ids), num_experts, dtype=torch.bool)
    for row, lid in enumerate(layer_ids):
        mask[row, k["keep"][str(lid)] if isinstance(k["keep"], dict) else k["keep"][row]] = False
    return mask


class ItemAccumulator:
    """Per-item routing-mass accumulator, fed by forward-pre-hooks on the experts
    modules (same hook point and args as profiler.attach). `start` restricts
    accumulation to positions >= start (the solution tokens)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.start = 0
        self.reset()

    def reset(self):
        self.pruned = torch.zeros(self.num_layers, dtype=torch.float64)
        self.total = torch.zeros(self.num_layers, dtype=torch.float64)

    def fractions(self) -> dict:
        per_layer = (self.pruned / self.total.clamp(min=1e-12)).tolist()
        return {"pooled": float(self.pruned.sum() / max(self.total.sum(), 1e-12)),
                "per_layer": per_layer}


def attach_item_hooks(model, adapter, mask: torch.Tensor) -> tuple[ItemAccumulator, list]:
    blocks = list(adapter.moe_blocks(model))
    assert len(blocks) == mask.shape[0], (len(blocks), mask.shape)
    acc = ItemAccumulator(len(blocks))
    handles = []

    def make(row: int):
        @torch.no_grad()
        def hook(_module, args):
            top_k_index, top_k_weights = args[1], args[2]
            idx = top_k_index[acc.start:].reshape(-1).cpu()
            w = top_k_weights[acc.start:].reshape(-1).double().cpu()
            sel = mask[row][idx]
            acc.pruned[row] += float(w[sel].sum())
            acc.total[row] += float(w.sum())
        return hook

    for row, (_, block) in enumerate(blocks):
        handles.append(block.experts.register_forward_pre_hook(make(row)))
    return acc, handles


@torch.no_grad()
def measure_item(model, acc: ItemAccumulator, input_ids: torch.Tensor, start: int) -> dict:
    """One teacher-forced pass; returns pruned-mass fractions over positions >= start."""
    assert 0 < start < input_ids.shape[-1], (start, input_ids.shape)
    acc.reset()
    acc.start = start
    device = next(model.parameters()).device
    model(input_ids=input_ids.view(1, -1).to(device), use_cache=False)
    return acc.fractions()


def bootstrap_diff(fail: list[float], ctrl: list[float], b: int = 10000,
                   seed: int = 0) -> dict:
    rng = random.Random(seed)
    point = sum(fail) / len(fail) - sum(ctrl) / len(ctrl)
    diffs = sorted(
        sum(rng.choices(fail, k=len(fail))) / len(fail)
        - sum(rng.choices(ctrl, k=len(ctrl))) / len(ctrl)
        for _ in range(b))
    return {"diff": point, "ci95": [diffs[int(0.025 * b)], diffs[int(0.975 * b)]],
            "p_le_0": sum(1 for d in diffs if d <= 0) / b,
            "mean_fail": sum(fail) / len(fail), "mean_ctrl": sum(ctrl) / len(ctrl),
            "n_fail": len(fail), "n_ctrl": len(ctrl)}


def _dump_prompts(dataset: str, out: Path):
    helper = Path(__file__).parents[2] / "scripts" / "d0_prompts_dump.py"
    r = subprocess.run(["uv", "run", "--no-project", "--with", "evalplus", "python",
                        str(helper), "--dataset", dataset, "--out", str(out)],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"d0_prompts_dump rc={r.returncode}: {r.stderr[-2000:]}")


def _samples(run_dir: Path, tag: str, dataset: str) -> dict[str, str]:
    from .codeval import _select_samples_file
    f = _select_samples_file(run_dir / "evalplus" / tag / dataset, dataset)
    return {r["task_id"]: r["solution"]
            for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def _encode(tok, prompt: str, solution: str) -> tuple[torch.Tensor, int]:
    from .repair_eval import turn1_user_message
    messages = [{"role": "user", "content": turn1_user_message(prompt)}]
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if not isinstance(ids, list):  # installed transformers returns a BatchEncoding here
        ids = ids["input_ids"]
    sol = tok(f"```python\n{solution.strip()}\n```", add_special_tokens=False).input_ids
    return torch.tensor(ids + sol, dtype=torch.long), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3.6-35b-a3b")
    ap.add_argument("--keep", default="runs/qwen36/keep_reap_0.25.json")
    ap.add_argument("--e1", default="runs/e1")
    ap.add_argument("--out", default="runs/d0")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_path = out_dir / "attribution.json"
    if out_path.exists() and not a.force:
        print(f"d0: {out_path} exists (use --force)")
        print("D0_DONE fails=0")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = build_item_sets(a.e1)
    for ds in DATASETS:
        print(f"d0[{ds}]: fail={len(sets[ds]['fail'])} ctrl={len(sets[ds]['ctrl'])}")

    prompts: dict[str, dict[str, str]] = {}
    for ds in DATASETS:
        p = out_dir / f"prompts_{ds}.json"
        if not p.exists():
            _dump_prompts(ds, p)
        prompts[ds] = json.loads(p.read_text())

    from transformers import AutoTokenizer

    from .adapters import get_adapter
    from .loader import load_streaming

    tok = AutoTokenizer.from_pretrained(a.model)
    adapter = get_adapter(json.loads((Path(a.model) / "config.json").read_text()))
    mask = pruned_mask(a.keep)
    print(f"d0: pruned mask {tuple(mask.shape)}, pruned/layer={int(mask[0].sum())}")
    model = load_streaming(a.model)
    acc, handles = attach_item_hooks(model, adapter, mask)

    t0 = time.time()
    per_item: dict[str, dict] = {}
    try:
        for ds in DATASETS:
            sols = _samples(Path(a.e1), "base_q8", ds)
            todo = [(tid, grp) for grp in ("fail", "ctrl") for tid in sets[ds][grp]]
            for i, (tid, grp) in enumerate(todo):
                ids, start = _encode(tok, prompts[ds][tid], sols[tid])
                if ids.shape[0] > a.max_seq:
                    print(f"d0[{ds}]: skip {tid} ({ids.shape[0]} tokens)")
                    continue
                fr = measure_item(model, acc, ids, start)
                per_item[tid] = {"dataset": ds, "group": grp, "n_tokens": ids.shape[0] - start,
                                 "pooled": fr["pooled"], "per_layer": fr["per_layer"]}
                if i % 25 == 0:
                    print(f"d0[{ds}]: {i}/{len(todo)} items, {time.time()-t0:.0f}s")
    finally:
        for h in handles:
            h.remove()

    stats = {}
    for scope in (*DATASETS, "pooled"):
        f = [v["pooled"] for v in per_item.values()
             if v["group"] == "fail" and (scope == "pooled" or v["dataset"] == scope)]
        c = [v["pooled"] for v in per_item.values()
             if v["group"] == "ctrl" and (scope == "pooled" or v["dataset"] == scope)]
        stats[scope] = bootstrap_diff(f, c)
        print(f"d0[{scope}]: fail={stats[scope]['mean_fail']:.4f} "
              f"ctrl={stats[scope]['mean_ctrl']:.4f} diff={stats[scope]['diff']:+.4f} "
              f"CI{stats[scope]['ci95']} P(<=0)={stats[scope]['p_le_0']:.4f}")

    out_path.write_text(json.dumps(
        {"model": a.model, "keep": a.keep, "e1": a.e1, "item_sets": sets,
         "stats": stats, "per_item": per_item}, indent=1))
    print(f"d0: written {out_path}")
    print("D0_DONE fails=0")


if __name__ == "__main__":
    main()
