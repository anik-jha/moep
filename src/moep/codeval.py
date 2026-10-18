"""Code benchmarks (eval-gate stage 3): EvalPlus HumanEval+ / MBPP+ against a GGUF
served by llama-server. One backend for base and pruned keeps comparisons fair; base
quant and pruned quant must match (e.g. both Q8_0).

evalplus runs in its own uvx environment (its dependency pins stay isolated from ours);
llama-server applies the GGUF-embedded chat template via /v1/chat/completions.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

SERVER_BIN = Path("tools/llama.cpp/build/bin/llama-server")


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: int = 300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode} during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("llama-server /health never turned 200")


@contextmanager
def llama_server(gguf: str | Path, port: int, ctx: int = 4096, parallel: int = 1,
                 log_file: Path | None = None, extra_args: tuple = (),
                 server: str | Path = SERVER_BIN, reasoning_format: str = "none"):
    """Start llama-server on the GGUF, yield once /health is 200, always tear down.
    The batch-safe way to exercise a GGUF (llama-cli drops into conversation mode
    when the model embeds a chat template and then blocks on a tty)."""
    out = log_file.open("ab") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(
        # --reasoning-format none: leave the assistant message unparsed in `content`.
        # The structured parser 500s on the unterminated output a degenerate checkpoint
        # emits ("does not match the expected peg-native format", observed 2026-07-14 on
        # random50), and evalplus retries a 500 forever, so a collapsed model hangs the
        # eval instead of scoring near zero. Parsing off means a looping model returns its
        # loop, fails the tests, and is recorded. Content is unchanged for models that do
        # not emit thought tags (the Qwen3.6 and Gemma4 runs). A model that emits an empty
        # <think></think> even with --reasoning off (Nemotron 3.5 Bolt, 2026-10-17) needs
        # "deepseek": evalplus.sanitize truncates "<think></think>def f(...)" to its first
        # line, failing every task.
        [str(server), "-m", str(gguf), "--port", str(port), "-ngl", "999",
         "-c", str(ctx * parallel), "--parallel", str(parallel), "--temp", "0",
         "--reasoning-format", reasoning_format, *extra_args],
        stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        # Local chat.cpp patch (env-gated): a generation that fails the chat-format
        # grammar is returned as plain content instead of an HTTP 500. evalplus retries
        # 500s indefinitely, so without this a degenerate checkpoint hangs the eval for
        # its full timeout instead of scoring near zero (random50, 2026-07-14: two 4h
        # timeouts, second one with --reasoning-format none already set).
        env=os.environ | {"LLAMA_CHAT_PARSE_LENIENT": "1"})
    try:
        _wait_health(port, proc)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_file:
            out.close()


def completion(port: int, prompt: str, n_predict: int = 48) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"]


def _preflight_chat(port: int):
    """One chat completion that must yield visible code. Catches configurations where
    a thinking model burns the whole budget inside its reasoning block and the eval
    harness would silently score 164 empty solutions as 0.0 (observed 2026-07-12)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content":
                         "Write a Python function add(a, b) that returns a + b. Code only."}],
                         "max_tokens": 256, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    if "def " not in content and "return" not in content:
        raise RuntimeError(
            f"codeval preflight got no usable code (content={content[:200]!r}); "
            "check --reasoning/thinking settings before burning a full eval run")


def parse_scores(stdout: str) -> dict[str, float]:
    """Parse evalplus.evaluate stdout into {suite name: pass@1}."""
    scores = dict(re.findall(r"(\S+ \(.*?tests\))\s*\npass@1:\s*([0-9.]+)", stdout))
    if not scores:
        scores = dict(re.findall(r"(pass@1):\s*([0-9.]+)", stdout))
    return {k: float(v) for k, v in scores.items()}


def evaluate_samples(dataset: str, samples: Path, log_to: Path) -> dict[str, float]:
    """Run evalplus.evaluate on a samples jsonl, return parsed scores or raise."""
    ev = subprocess.run(
        ["uvx", "--from", "evalplus", "evalplus.evaluate", "--dataset", dataset,
         "--samples", str(samples)],
        capture_output=True, text=True, timeout=3600)
    log_to.write_text(ev.stdout[-20000:] + "\n--stderr--\n" + ev.stderr[-20000:])
    scores = parse_scores(ev.stdout)
    if ev.returncode != 0 or not scores:
        raise RuntimeError(f"evalplus.evaluate rc={ev.returncode}, parsed={scores}; see {log_to}")
    return scores


def _select_samples_file(root: Path, dataset: str) -> Path | None:
    """Pick the sanitized evalplus samples file, never the raw (markdown-fenced) sibling.

    evalplus.codegen writes both the sanitized samples file and a raw, unsanitized
    sibling ending in ".raw.jsonl" into the same directory. Both match "*.jsonl", and
    glob() order is filesystem-dependent, so without excluding the raw file this could
    silently select fenced code -> every task fails -> pass@1 0.0 (observed 2026-07-12
    on base_q8 and reap25_q8).
    """
    def sanitized_only(paths):
        return next((p for p in paths if not p.name.endswith(".raw.jsonl")), None)
    return sanitized_only(root.glob(f"{dataset}/*.jsonl")) or sanitized_only(root.glob("**/*.jsonl"))


def run_codeval(gguf: str | Path, run_dir: str | Path, tag: str, dataset: str = "humaneval",
                port: int = 8090, ctx: int = 16384, parallel: int = 4, force: bool = False,
                log=print, server: str | Path = SERVER_BIN, reasoning_format: str = "none") -> dict:
    run_dir = Path(run_dir)
    root = run_dir / "evalplus" / tag / dataset
    # evalplus.codegen silently RESUMES from any samples already under --root and will
    # re-score stale generations without running the model at all (observed 2026-07-12:
    # a "fixed" rerun scored old thinking-mode garbage in 0s, "resuming from 1" per task).
    # A rerun that is meant to regenerate must purge the workspace first.
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    # Code evals run in non-thinking mode for every model (base and pruned alike):
    # greedy pass@1 with reasoning off is the standard, comparable configuration.
    with llama_server(gguf, port, ctx=ctx, parallel=parallel, log_file=root / "server.log",
                      extra_args=("--reasoning", "off"), server=server,
                      reasoning_format=reasoning_format):
        _preflight_chat(port)
        log(f"codeval[{tag}/{dataset}]: preflight OK, generating...")
        gen = subprocess.run(
            ["uvx", "--from", "evalplus", "evalplus.codegen", "--model", tag,
             "--dataset", dataset, "--backend", "openai",
             "--base_url", f"http://127.0.0.1:{port}/v1",
             "--greedy", "--root", str(root)],
            capture_output=True, text=True, timeout=4 * 3600,
            env=os.environ | {"OPENAI_API_KEY": "local"})
        (root / "codegen.log").write_text(gen.stdout[-20000:] + "\n--stderr--\n" + gen.stderr[-20000:])
        if gen.returncode != 0:
            raise RuntimeError(f"evalplus.codegen rc={gen.returncode}; see {root}/codegen.log")

    samples = _select_samples_file(root, dataset)
    if samples is None:
        raise FileNotFoundError(f"no generated samples under {root}")
    log(f"codeval[{tag}/{dataset}]: evaluating {samples.name}")
    scores = evaluate_samples(dataset, samples, root / "evaluate.log")

    result = {"gguf": str(gguf), "dataset": dataset, "tag": tag, "server": str(server),
              "reasoning_format": reasoning_format, "scores": scores}
    log(f"codeval[{tag}/{dataset}]: {result['scores']}")
    return result
