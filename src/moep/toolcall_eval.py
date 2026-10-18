"""Tool-calling eval on BFCL v3 single-call categories (simple, multiple) through llama-server's
OpenAI-compatible tools interface.

Two metrics per item:
  valid    - the reply carries exactly one tool call naming a provided function, with arguments
             that parse as a JSON object
  correct  - valid, the function matches the ground truth, every required argument is present
             with an allowed value, and every optional argument given has an allowed value

The correctness check is a simplified version of BFCL's AST checker (string normalization as
BFCL does, numeric equality across int/float), so scores are comparable between models run here,
not with the official leaderboard. Function names containing '.' are sent with '_' (OpenAI-style
names), as BFCL's own OpenAI handler does, and mapped back for scoring.

Data: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
"""

from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .codeval import SERVER_BIN, llama_server
from .evalx import _degenerate

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array"}


def _schema(node):
    """BFCL parameter schema -> JSON schema (dict/float/tuple renamed, 'any' left untyped)."""
    if isinstance(node, list):
        return [_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            if v != "any":
                out[k] = TYPE_MAP.get(v, v)
        else:
            out[k] = _schema(v)
    return out


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^]", "", v).lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _allowed(value, options) -> bool:
    return any(_norm(value) == _norm(o) for o in options)


def score_call(call: dict | None, functions: list[dict], truth: list[dict]) -> dict:
    """call = {"name": original function name, "arguments": dict} or None when invalid."""
    if call is None:
        return {"valid": False, "correct": False}
    gt_name, gt_args = next(iter(truth[0].items()))
    fn = next((f for f in functions if f["name"] == call["name"]), None)
    if fn is None:
        return {"valid": False, "correct": False}
    required = set(fn["parameters"].get("required", []))
    props = set(fn["parameters"].get("properties", {}))
    args = call["arguments"]
    ok = call["name"] == gt_name and set(args) <= props
    for p, options in gt_args.items():
        if p in args:
            ok &= _allowed(args[p], options)
        else:
            ok &= p not in required and "" in options
    return {"valid": True, "correct": bool(ok)}


def _request(port: int, item: dict, max_tokens: int) -> dict:
    to_api = {f["name"]: f["name"].replace(".", "_") for f in item["function"]}
    from_api = {v: k for k, v in to_api.items()}
    tools = [{"type": "function", "function": {"name": to_api[f["name"]], "description": f["description"],
                                               "parameters": _schema(f["parameters"])}} for f in item["function"]]
    body = {"messages": item["question"][0], "tools": tools, "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    call = None
    if len(calls) == 1:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
            if isinstance(args, dict):
                call = {"name": from_api.get(calls[0]["function"]["name"], calls[0]["function"]["name"]),
                        "arguments": args}
        except (json.JSONDecodeError, TypeError):
            pass
    content = msg.get("content") or ""
    return {"call": call, "n_calls": len(calls), "content": content[:2000],
            "degenerate": not calls and _degenerate(content)}


def run_toolcall(gguf: str | Path, run_dir: str | Path, tag: str, data_dir: str | Path = "data/bfcl",
                 categories: tuple[str, ...] = ("simple", "multiple"), port: int = 8098,
                 server: str | Path = SERVER_BIN, parallel: int = 4, max_tokens: int = 1024) -> dict:
    out_dir = Path(run_dir) / "toolcall" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"gguf": str(gguf), "tag": tag, "server": str(server)}
    with llama_server(gguf, port, ctx=8192, parallel=parallel, log_file=out_dir / "server.log",
                      extra_args=("--reasoning", "off"), server=server, reasoning_format="deepseek"):
        for cat in categories:
            items = [json.loads(x) for x in Path(data_dir, f"BFCL_v3_{cat}.json").read_text().splitlines() if x]
            truth = {d["id"]: d["ground_truth"] for d in (json.loads(x) for x in
                     Path(data_dir, "possible_answer", f"BFCL_v3_{cat}.json").read_text().splitlines() if x)}
            with ThreadPoolExecutor(parallel) as pool:
                replies = list(pool.map(lambda it: _request(port, it, max_tokens), items))
            rows = [{"id": it["id"], **rep, **score_call(rep["call"], it["function"], truth[it["id"]])}
                    for it, rep in zip(items, replies)]
            (out_dir / f"{cat}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            n = len(rows)
            summary[cat] = {"n": n, "valid": round(sum(r["valid"] for r in rows) / n, 4),
                            "correct": round(sum(r["correct"] for r in rows) / n, 4),
                            "degenerate": sum(r["degenerate"] for r in rows)}
    return summary
