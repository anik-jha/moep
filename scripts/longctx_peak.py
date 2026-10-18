"""Peak GPU memory during one real long-context request, plus a single needle-retrieval smoke check.

Serves the GGUF fully offloaded with one slot at the given context, builds a user message of about
<target_tokens> tokens from calibration code text with one passphrase sentence placed at 60% depth,
asks for the passphrase, and samples the server's GPU memory (nvidia-smi, every 0.25 s) from before
the request until the reply. The needle check is one prompt, not a long-context benchmark.
Writes longctx_<tag> into <run>/eval.json.

Usage: longctx_peak.py <run_dir> <tag> <model.gguf> <server_bin> <ctx> <target_tokens>
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from moep.codeval import llama_server
from moep.data import read_jsonl
from moep.evalx import write_eval
from measure_mem import proc_gpu_mib

PORT = 8101
NEEDLE = "The deployment passphrase for this repository is violet-anchor-7316."
QUESTION = "\n\nWhat is the deployment passphrase for this repository? Reply with the passphrase only."


def _post(path: str, body: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _n_tokens(text: str) -> int:
    return len(_post("/tokenize", {"content": text})["tokens"])


def build_prompt(target: int) -> str:
    texts = [m["content"] for r in read_jsonl("data/calib/calib_code.jsonl") for m in r["messages"]
             if m["role"] != "system"]
    parts, total = [], 0
    for t in texts:
        n = _n_tokens(t)
        if total + n > target:
            break
        parts.append(t)
        total += n
    parts.insert(int(len(parts) * 0.6), NEEDLE)
    return "\n\n".join(parts) + QUESTION


def main(run_dir: str, tag: str, gguf: str, server: str, ctx: str, target: str) -> int:
    out = Path(run_dir) / "longctx"
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{tag}.server.log"
    log.unlink(missing_ok=True)
    with llama_server(gguf, PORT, ctx=int(ctx), parallel=1, log_file=log, server=server,
                      extra_args=("--reasoning", "off"), reasoning_format="deepseek") as proc:
        prompt = build_prompt(int(target))
        idle = proc_gpu_mib(proc.pid)
        samples, done = [], threading.Event()

        def sample():
            while not done.is_set():
                samples.append(proc_gpu_mib(proc.pid) or 0)
                time.sleep(0.25)

        t = threading.Thread(target=sample)
        t.start()
        t0 = time.time()
        r = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                                           "max_tokens": 32, "temperature": 0})
        wall = time.time() - t0
        done.set()
        t.join()
    answer = r["choices"][0]["message"]["content"] or ""
    result = {"gguf": gguf, "ctx": int(ctx), "prompt_tokens": r["usage"]["prompt_tokens"],
              "idle_gpu_mib": idle, "peak_gpu_mib": max(samples), "peak_gpu_gib": round(max(samples) / 1024, 2),
              "samples": len(samples), "request_wall_s": round(wall, 1),
              "prefill_tps": round(r["timings"]["prompt_per_second"], 1),
              "needle_found": "violet-anchor-7316" in answer, "answer": answer[:200]}
    write_eval(run_dir, f"longctx_{tag}", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
