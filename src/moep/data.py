"""Calibration mix builder (decision D3: ~70% code+tool+SWE / 30% general ballast).

Each source is normalized to {"messages": [{role, content}, ...], "source": name}.
Gated or renamed datasets fall back down the chain and the substitution is recorded,
so the manifest always states exactly what the mix contained.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


def _evol(n, seed):
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"messages": [{"role": "user", "content": r["instruction"]},
                          {"role": "assistant", "content": r["output"]}]} for r in ds]


def _xlam(n, seed):
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        tools = r.get("tools", "")
        q = f"You have access to these tools:\n{tools}\n\n{r['query']}"
        out.append({"messages": [{"role": "user", "content": q},
                                 {"role": "assistant", "content": r["answers"]}]})
    return out


def _hermes_fc(n, seed):
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        msgs = [{"role": m["from"].replace("human", "user").replace("gpt", "assistant"),
                 "content": m["value"]} for m in r["conversations"]]
        msgs = [m for m in msgs if m["role"] in ("system", "user", "assistant")]
        if len(msgs) >= 2:
            out.append({"messages": msgs})
    return out


def _swe_smith(n, seed):
    """messages is a JSON string; content is str or a list of {type, text} parts; tool
    results become user turns so any chat template can render the trajectory."""
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        msgs = []
        for m in json.loads(r["messages"]):
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            role = {"tool": "user"}.get(m["role"], m["role"])
            if role in ("system", "user", "assistant") and content:
                msgs.append({"role": role, "content": str(content)})
        if len(msgs) >= 2:
            out.append({"messages": msgs})
    return out


def _tulu(n, seed):
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    out = []
    rng = random.Random(seed)
    for r in ds.shuffle(seed=seed, buffer_size=10_000):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]
                if m["role"] in ("system", "user", "assistant")]
        if len(msgs) >= 2 and rng.random() < 0.5:
            out.append({"messages": msgs})
        if len(out) >= n:
            break
    return out


# (source name, loader, fallbacks)
SOURCES = {
    "evol-codealpaca": (_evol, []),
    "xlam-function-calling": (_xlam, [("hermes-function-calling", _hermes_fc)]),
    "swe-smith-trajectories": (_swe_smith, []),
    "tulu-3-general": (_tulu, []),
}

# D3 mix at n=2048: 35% evol, 17.5% tool-calling, 17.5% SWE agentic, 30% general.
MIX = {"evol-codealpaca": 0.35, "xlam-function-calling": 0.175,
       "swe-smith-trajectories": 0.175, "tulu-3-general": 0.30}


def build_mix(out_dir: str | Path, n_total: int = 2048, holdout: int = 256,
              seed: int = 17, log=print) -> dict:
    """Writes calib.jsonl (profiling), holdout_code.jsonl and holdout_general.jsonl (ppl eval).
    Returns a report of what was actually loaded, including any fallback substitutions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_total": n_total, "seed": seed, "sources": {}}
    rows: list[dict] = []
    extra: dict[str, list[dict]] = {}

    for name, frac in MIX.items():
        want = round(n_total * frac) + (holdout if name in ("evol-codealpaca", "tulu-3-general") else 0)
        loader, fallbacks = SOURCES[name]
        chain = [(name, loader)] + fallbacks
        got, used = [], None
        for src_name, fn in chain:
            try:
                got = fn(want, seed)
                used = src_name
                break
            except Exception as e:  # gated dataset, schema change, network
                log(f"calib: {src_name} failed ({type(e).__name__}: {e}); trying fallback")
        if not got:
            raise RuntimeError(f"no source in chain for {name} could be loaded")
        for r in got:
            r["source"] = used
        report["sources"][name] = {"used": used, "n": len(got), "requested": want}
        if name == "evol-codealpaca":
            extra["holdout_code"] = got[:holdout]
            got = got[holdout:]
        elif name == "tulu-3-general":
            extra["holdout_general"] = got[:holdout]
            got = got[holdout:]
        rows.extend(got)

    random.Random(seed).shuffle(rows)
    _write_jsonl(out_dir / "calib.jsonl", rows)
    _write_jsonl(out_dir / "holdout_code.jsonl", extra.get("holdout_code", []))
    _write_jsonl(out_dir / "holdout_general.jsonl", extra.get("holdout_general", []))
    report["written"] = {"calib": len(rows),
                         "holdout_code": len(extra.get("holdout_code", [])),
                         "holdout_general": len(extra.get("holdout_general", []))}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"calib: {report['written']}")
    return report


def tokenize_rows_plain(tok, rows: list[dict]) -> list[list[int]]:
    """Render chat rows as bare message content, no chat template. Gemma-4's template
    inflates teacher-forced ppl ~30x on identical content (measured 2026-07-15: 416.8
    templated vs 13.2 plain on the same holdout sample; base model generates HumanEval+
    at 0.927, so the model is fine and the templated measurement is the artifact).
    Plain-text ppl is the template-robust protocol for cross-family comparisons."""
    return [tok("\n\n".join(m["content"] for m in r["messages"])).input_ids for r in rows]


def tokenize_rows(tok, rows: list[dict], log=print) -> list[list[int]]:
    """Render chat rows to token ids. Consecutive same-role turns are merged (tool
    results were remapped to user turns) and samples a template rejects are skipped
    and counted rather than killing a long profiling run."""
    out, skipped = [], 0
    for r in rows:
        msgs = []
        for m in r["messages"]:
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1]["content"] += "\n\n" + m["content"]
            else:
                msgs.append(dict(m))
        try:
            # return_dict=False: transformers 5.x otherwise returns a BatchEncoding
            out.append(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))
        except Exception:
            skipped += 1
    if skipped:
        log(f"tokenize: skipped {skipped}/{len(rows)} samples the chat template rejected")
    return out


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
