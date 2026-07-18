"""Eval ladder helpers: smoke generations and perplexity deltas (stages 1-2 of eval-gate).
Code benchmarks (EvalPlus) and llama-bench throughput run via their own tools; results
land in the same runs/<id>/eval.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .data import read_jsonl


@torch.no_grad()
def smoke(model, tokenizer, prompts_path: str | Path, max_new_tokens: int = 160) -> list[dict]:
    """Greedy completions for the fixed smoke prompts. Deterministic by construction."""
    out = []
    for row in read_jsonl(prompts_path):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                            add_generation_prompt=True, return_tensors="pt",
                                            return_dict=False)
        ids = ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        out.append({"id": row["id"], "prompt": row["prompt"], "completion": text,
                    "degenerate": _degenerate(text)})
    return out


def _degenerate(text: str) -> bool:
    """Cheap gibberish detector: hard repetition loops or near-empty output."""
    if len(text.strip()) < 8:
        return True
    words = text.split()
    if len(words) >= 24:
        tail = words[-24:]
        if len(set(tail)) <= 3:
            return True
    for n in (12, 24, 48):
        if len(text) >= 2 * n and text[-n:] == text[-2 * n:-n]:
            return True
    return False


@torch.no_grad()
def perplexity(model, tokenizer, jsonl_path: str | Path, seq_len: int = 2048,
               max_seqs: int = 48, plain: bool = False) -> dict:
    """Packed-sequence perplexity over a held-out set. Same packing as profiling, so
    base-vs-pruned deltas are apples to apples. plain=True skips the chat template
    (see tokenize_rows_plain: required for template-sensitive families like gemma4)."""
    from .data import tokenize_rows, tokenize_rows_plain
    from .profiler import pack_token_stream

    rows = read_jsonl(jsonl_path)
    token_lists = (tokenize_rows_plain(tokenizer, rows) if plain
                   else tokenize_rows(tokenizer, rows, log=lambda *_: None))
    batches = pack_token_stream(token_lists, seq_len, tokenizer.eos_token_id)[:max_seqs]
    if batches.shape[0] == 0:
        raise ValueError(f"not enough tokens in {jsonl_path} for one {seq_len}-token sequence")

    nll_sum, n_tok = 0.0, 0
    for i in range(batches.shape[0]):
        ids = batches[i : i + 1].to(model.device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        nll_sum += float(-logp.gather(1, tgt.unsqueeze(1)).sum())
        n_tok += tgt.numel()
    return {"nll_per_token": nll_sum / n_tok, "ppl": math.exp(nll_sum / n_tok),
            "tokens": n_tok, "sequences": int(batches.shape[0])}


def write_eval(run_dir: str | Path, section: str, payload: dict):
    """Merge a section into runs/<id>/eval.json (the single source of quotable numbers)."""
    p = Path(run_dir) / "eval.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data[section] = payload
    p.write_text(json.dumps(data, indent=2))
    return p


def eval_has(run_dir: str | Path, section: str) -> bool:
    p = Path(run_dir) / "eval.json"
    return p.exists() and section in json.loads(p.read_text())
