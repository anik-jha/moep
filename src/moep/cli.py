"""moep CLI: one subcommand per pipeline stage, each recording a manifest in the run dir.

  facts    print verified architecture facts from a local checkpoint (no weights needed)
  calib    build the calibration mix (D3)
  profile  stream calibration through the model, accumulate expert stats
  score    rank experts, write keep-set selection
  prune    checkpoint surgery -> pruned safetensors model
  smoke    greedy generations on fixed prompts (eval-gate stage 1)
  ppl      packed perplexity on a held-out set (eval-gate stage 2)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_config(model_dir: str) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())


def _adapter(model_dir: str):
    from .adapters import get_adapter

    return get_adapter(_load_config(model_dir))


def _load_model(model_dir: str, device: str = "cuda:0", max_gpu_gib: int = 0,
                offload_dir: str = ""):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if max_gpu_gib:
        from .loader import load_offloaded

        return load_offloaded(model_dir, max_gpu_gib, offload_dir or "runs/offload"), tok
    from .loader import load_streaming

    return load_streaming(model_dir, device=device), tok


def cmd_facts(a):
    ad = _adapter(a.model)
    facts = {"model_type": ad.config.get("model_type"), "num_experts": ad.num_experts,
             "top_k": ad.top_k, "num_layers": ad.num_layers, "hidden_size": ad.hidden_size,
             "moe_intermediate_size": ad.moe_intermediate_size}
    print(json.dumps(facts, indent=2))


def cmd_calib(a):
    from .data import build_mix
    from .manifest import Run

    run, t0 = Run(a.run), time.time()
    if run.done("calib") and not a.force:
        print("calib: already done (use --force)")
        return
    report = build_mix(a.out, n_total=a.n, holdout=a.holdout, seed=a.seed)
    run.record("calib", {"n": a.n, "seed": a.seed, "holdout": a.holdout},
               {"calib": Path(a.out) / "calib.jsonl", "report": Path(a.out) / "report.json"},
               report, t0)


def cmd_profile(a):
    import torch

    from .data import read_jsonl
    from .manifest import Run
    from .profiler import pack_token_stream, profile, write_summary

    run, t0 = Run(a.run), time.time()
    stage = f"profile_{a.tag}" if a.tag else "profile"
    if run.done(stage) and not a.force:
        print(f"{stage}: already done (use --force)")
        return
    ad = _adapter(a.model)
    model, tok = _load_model(a.model, max_gpu_gib=a.max_gpu_gib, offload_dir=a.offload_dir)
    rows = read_jsonl(a.calib)
    if a.max_samples:
        rows = rows[: a.max_samples]
    from .data import tokenize_rows

    batches = pack_token_stream(tokenize_rows(tok, rows), a.seq_len, tok.eos_token_id)
    if a.max_seqs:
        batches = batches[: a.max_seqs]
    print(f"{stage}: {len(rows)} samples -> {batches.shape[0]} packed seqs of {a.seq_len}")
    stats, layer_ids = profile(model, ad, batches, a.batch_size)
    out = Path(a.run) / f"{stage}.stats.pt"
    summary = write_summary(out, stats, layer_ids,
                            {"model": a.model, "calib": a.calib, "seq_len": a.seq_len,
                             "n_samples": len(rows)})
    run.record(stage, vars(a) | {"func": None}, {"stats": out}, summary, t0)
    print(json.dumps(summary, indent=2))
    del model
    torch.cuda.empty_cache()


def cmd_score(a):
    from .manifest import Run
    from .profiler import load_stats
    from .scoring import save_selection, select

    run, t0 = Run(a.run), time.time()
    stats, meta = load_stats(a.stats or Path(a.run) / "profile.stats.pt")
    ref = None
    if a.reference_stats:
        ref, _ = load_stats(a.reference_stats)
    keeps, info = select(stats, a.criterion, a.keep_ratio, reference_stats=ref, seed=a.seed)
    out = Path(a.run) / f"keep_{a.criterion}_{a.keep_ratio}.json"
    save_selection(out, keeps, meta["layer_ids"], info | {"stats_meta": {k: v for k, v in meta.items() if k != 'layer_ids'}})
    run.record(f"score_{a.criterion}_{a.keep_ratio}", {"criterion": a.criterion, "keep_ratio": a.keep_ratio},
               {"selection": out}, info, t0)
    print(json.dumps(info, indent=2))


def cmd_prune(a):
    from .manifest import Run
    from .scoring import load_selection
    from .surgery import prune_checkpoint

    run, t0 = Run(a.run), time.time()
    keep, info = load_selection(a.selection)
    report = prune_checkpoint(a.model, a.out, keep, _adapter(a.model))
    run.record(f"prune_{Path(a.out).name}", {"selection": a.selection, "out": a.out},
               {"model": Path(a.out) / "model.safetensors.index.json"}, report | {"selection_info": info}, t0)


def cmd_gguf_score(a):
    from .gguf_prune import experts_used
    from .gguf_select import load_stats, select, split_half
    from .manifest import Run
    from .scoring import save_selection

    run, t0 = Run(a.run), time.time()
    stats, used = load_stats(a.stats), experts_used(a.gguf)
    keep, info = select(stats, a.criterion, a.keep_ratio, used)
    info["split_half"] = split_half(stats, a.criterion, a.keep_ratio, used)
    save_selection(a.out, list(keep.values()), list(keep), info)
    sh = info["split_half"]
    run.record(f"gguf_score_{a.criterion}_{a.keep_ratio}", vars(a) | {"func": None}, {"selection": a.out},
               {k: v for k, v in sh.items() if not isinstance(v, dict)}
               | {"experts_kept": info["experts_kept"], "experts_total": info["experts_total"]}, t0)


def cmd_gguf_prune(a):
    from .gguf_prune import prune_gguf, verify_pruned
    from .manifest import Run
    from .scoring import load_selection

    run, t0 = Run(a.run), time.time()
    keep, info = load_selection(a.selection)
    report = prune_gguf(a.src, a.out, keep)
    report["verify"] = verify_pruned(a.src, a.out, keep)
    report["file_bytes"] = Path(a.out).stat().st_size
    run.record(f"gguf_prune_{Path(a.out).stem}", vars(a) | {"func": None}, {"gguf": a.out},
               report | {"selection_info": {k: v for k, v in info.items() if k != "split_half"}}, t0)


def cmd_smoke(a):
    from .evalx import eval_has, smoke, write_eval
    from .manifest import Run

    Run(a.run)
    key = f"smoke_{a.tag or Path(a.model).name}"
    if eval_has(a.run, key) and not a.force:
        prior = json.loads((Path(a.run) / "eval.json").read_text())[key]
        print(f"smoke: {key} already in eval.json (--force to redo)")
        if prior["degenerate"]:
            raise SystemExit(f"HARD FAIL (recorded): {prior['degenerate']} degenerate completions")
        return
    model, tok = _load_model(a.model)
    results = smoke(model, tok, a.prompts, max_new_tokens=a.max_new_tokens)
    n_bad = sum(r["degenerate"] for r in results)
    payload = {"model": a.model, "n": len(results), "degenerate": n_bad, "results": results}
    p = write_eval(a.run, f"smoke_{a.tag or Path(a.model).name}", payload)
    print(f"smoke: {len(results)} prompts, {n_bad} degenerate -> {p}")
    if n_bad:
        raise SystemExit(f"HARD FAIL: {n_bad} degenerate completions")


def cmd_export(a):
    from .export import export_gguf
    from .manifest import Run

    run, t0 = Run(a.run), time.time()
    stage = f"export_{Path(a.model).name}_{a.quant}"
    if run.done(stage) and not a.force:
        print(f"export: {stage} already done (--force to redo)")
        return
    report = export_gguf(a.model, a.out, quant=a.quant, keep_bf16=a.keep_bf16)
    run.record(f"export_{Path(a.model).name}_{a.quant}", {"model": a.model, "quant": a.quant},
               {"gguf": report["gguf"]}, report, t0)


def cmd_codeval(a):
    from .codeval import SERVER_BIN, run_codeval
    from .evalx import eval_has, write_eval
    from .manifest import Run

    Run(a.run)
    key = f"codeval_{a.tag}_{a.dataset}"
    if eval_has(a.run, key) and not a.force:
        print(f"codeval: {key} already in eval.json (--force to redo)")
        return
    result = run_codeval(a.gguf, a.run, a.tag, dataset=a.dataset, port=a.port, force=a.force,
                         server=a.server or SERVER_BIN, reasoning_format=a.reasoning_format)
    p = write_eval(a.run, f"codeval_{a.tag}_{a.dataset}", result)
    print(json.dumps(result | {"written": str(p)}, indent=2))


def cmd_toolcall(a):
    from .codeval import SERVER_BIN
    from .evalx import eval_has, write_eval
    from .manifest import Run
    from .toolcall_eval import run_toolcall

    Run(a.run)
    key = f"toolcall_{a.tag}"
    if eval_has(a.run, key) and not a.force:
        print(f"toolcall: {key} already in eval.json (--force to redo)")
        return
    result = run_toolcall(a.gguf, a.run, a.tag, port=a.port, server=a.server or SERVER_BIN)
    p = write_eval(a.run, key, result)
    print(json.dumps(result | {"written": str(p)}, indent=2))


def cmd_repaireval(a):
    from .evalx import eval_has, write_eval
    from .manifest import Run
    from .repair_eval import run_repair_eval

    Run(a.run)
    key = f"repaireval_{a.tag}_{a.dataset}"
    if eval_has(a.run, key) and not a.force:
        print(f"repaireval: {key} already in eval.json (--force to redo)")
        return
    result = run_repair_eval(a.gguf, a.run, a.tag, dataset=a.dataset, port=a.port,
                             force=a.force)
    p = write_eval(a.run, key, result)
    print(json.dumps(result | {"written": str(p)}, indent=2))


def cmd_heal(a):
    from .heal import heal
    from .manifest import Run

    run, t0 = Run(a.run), time.time()
    stage = f"heal_{Path(a.out).name}"
    if run.done(stage) and not a.force:
        print(f"heal: {stage} already done (--force to redo)")
        return
    report = heal(a.model, a.data, a.out, lr=a.lr, r=a.r, epochs=a.epochs,
                  seq_len=a.seq_len, grad_accum=a.grad_accum)
    run.record(stage, {k: getattr(a, k) for k in ("model", "data", "lr", "r", "epochs")},
               {"model": Path(a.out) / "model.safetensors.index.json"}, report, t0)


def cmd_ppl(a):
    from .evalx import eval_has, perplexity, write_eval
    from .manifest import Run

    Run(a.run)
    key = f"ppl_{a.tag or Path(a.model).name}"
    if eval_has(a.run, key) and not a.force:
        print(f"ppl: {key} already in eval.json (--force to redo)")
        return
    model, tok = _load_model(a.model)
    payload = {"model": a.model, "data": a.data,
               **perplexity(model, tok, a.data, seq_len=a.seq_len, max_seqs=a.max_seqs,
                            plain=a.plain)}
    p = write_eval(a.run, f"ppl_{a.tag or Path(a.model).name}", payload)
    print(json.dumps(payload | {"written": str(p)}, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="moep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("facts"); p.add_argument("--model", required=True); p.set_defaults(func=cmd_facts)

    p = sub.add_parser("calib")
    p.add_argument("--run", required=True); p.add_argument("--out", default="data/calib")
    p.add_argument("--n", type=int, default=2048); p.add_argument("--holdout", type=int, default=256)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_calib)

    p = sub.add_parser("profile")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--calib", required=True); p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-seqs", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-gpu-gib", type=int, default=0,
                   help="cap resident weights and disk-offload the rest (for models > memory budget)")
    p.add_argument("--offload-dir", default="runs/offload")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("score")
    p.add_argument("--run", required=True); p.add_argument("--stats", default="")
    p.add_argument("--criterion", default="reap",
                   choices=["reap", "reap_flow", "freq", "gate_mass", "prob_mass", "man", "msan",
                            "contrastive", "hybrid_share", "hybrid_rank", "random",
                            "guard16", "guard32"])
    p.add_argument("--keep-ratio", type=float, required=True)
    p.add_argument("--reference-stats", default=""); p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("prune")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--selection", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("smoke")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--prompts", default="tests/prompts/smoke.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=160); p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("gguf-score")
    p.add_argument("--run", required=True); p.add_argument("--stats", required=True)
    p.add_argument("--gguf", required=True, help="source GGUF (for experts routed per layer)")
    p.add_argument("--criterion", default="reap", choices=["reap", "gate_mass", "freq"])
    p.add_argument("--keep-ratio", type=float, required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gguf_score)

    p = sub.add_parser("gguf-prune")
    p.add_argument("--run", required=True); p.add_argument("--src", required=True)
    p.add_argument("--selection", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gguf_prune)

    p = sub.add_parser("codeval")
    p.add_argument("--run", required=True); p.add_argument("--gguf", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--dataset", default="humaneval",
                                                           choices=["humaneval", "mbpp"])
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--server", help="llama-server binary (default: codeval.SERVER_BIN)")
    p.add_argument("--reasoning-format", default="none", choices=["none", "deepseek"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_codeval)

    p = sub.add_parser("toolcall")
    p.add_argument("--run", required=True); p.add_argument("--gguf", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--port", type=int, default=8098)
    p.add_argument("--server", help="llama-server binary (default: codeval.SERVER_BIN)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_toolcall)

    p = sub.add_parser("repaireval")
    p.add_argument("--run", required=True); p.add_argument("--gguf", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--dataset", default="humaneval",
                                                           choices=["humaneval", "mbpp"])
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_repaireval)

    p = sub.add_parser("export")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", default="models/gguf"); p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--keep-bf16", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("heal")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--r", type=int, default=64)
    p.add_argument("--epochs", type=float, default=1.0); p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=8); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_heal)

    p = sub.add_parser("ppl")
    p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--data", required=True); p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-seqs", type=int, default=48); p.add_argument("--tag", default="")
    p.add_argument("--plain", action="store_true",
                   help="no chat template; required for template-sensitive families (gemma4)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ppl)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
