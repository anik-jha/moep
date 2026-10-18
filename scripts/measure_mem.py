"""Measure GPU memory for a GGUF served fully offloaded by llama-server, at one or more context
lengths. Records llama.cpp's own breakdown (model + context + compute on the GPU, and host
buffers) and nvidia-smi's per-process used memory, which adds the CUDA context. Writes
memory_<tag> into <run>/eval.json.

Usage: measure_mem.py <run_dir> <tag> <model.gguf> <server_bin> <ctx> [<ctx> ...]
"""

import re
import subprocess
import sys
from pathlib import Path

from moep.codeval import llama_server
from moep.evalx import write_eval

BREAKDOWN = re.compile(r"CUDA0 \(.*?\)\s*\|\s*\d+\s*=\s*\d+\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)")
HOST = re.compile(r"- Host\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)")
PORT = 8100


def proc_gpu_mib(pid: int) -> int | None:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return next((int(m) for p, m in (line.split(",") for line in out.strip().splitlines()) if int(p) == pid), None)


def measure(gguf: str, server: str, ctx: int, log: Path) -> dict:
    with llama_server(gguf, PORT, ctx=ctx, parallel=1, log_file=log, extra_args=("-v",), server=server) as proc:
        smi = proc_gpu_mib(proc.pid)
    text = log.read_text(errors="replace")
    gpu, host = BREAKDOWN.findall(text)[-1], HOST.findall(text)[-1]
    return {"ctx": ctx, "gpu_self_mib": int(gpu[0]), "gpu_model_mib": int(gpu[1]),
            "gpu_context_mib": int(gpu[2]), "gpu_compute_mib": int(gpu[3]),
            "host_mib": int(host[0]), "nvidia_smi_process_mib": smi,
            "nvidia_smi_process_gib": round(smi / 1024, 2) if smi else None}


def main(run_dir: str, tag: str, gguf: str, server: str, *ctxs: str) -> int:
    out = Path(run_dir) / "memory"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in map(int, ctxs):
        log = out / f"{tag}_ctx{c}.log"
        log.unlink(missing_ok=True)
        rows.append(measure(gguf, server, c, log))
        print(rows[-1])
    write_eval(run_dir, f"memory_{tag}", {"gguf": gguf, "server": server, "by_ctx": rows})
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
