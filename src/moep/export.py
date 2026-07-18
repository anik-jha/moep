"""Export stage: HF safetensors -> GGUF -> quantize -> load-verify -> bench.

Uses the in-tree llama.cpp converter (tools/llama.cpp), which registers
Qwen3_5MoeForConditionalGeneration for both the text model and the vision mmproj.
The vision tower is copied verbatim by surgery, so one mmproj converted from the BASE
checkpoint works for every pruned variant.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LLAMA = Path("tools/llama.cpp")
BIN = LLAMA / "build/bin"


def _run(cmd: list[str], log_path: Path, env_extra: dict | None = None) -> None:
    import os

    env = os.environ | {"PYTHONPATH": str(LLAMA / "gguf-py")} | (env_extra or {})
    with log_path.open("ab") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n".encode())
        f.flush()
        subprocess.run(list(map(str, cmd)), check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def export_gguf(model_dir: str | Path, out_dir: str | Path, quant: str = "Q4_K_M",
                keep_bf16: bool = False, log=print) -> dict:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = model_dir.name
    log_path = out_dir / "export.log"
    bf16 = out_dir / f"{name}-bf16.gguf"
    quant_path = out_dir / f"{name}-{quant}.gguf"

    if not quant_path.exists():
        if not bf16.exists():
            log(f"export: converting {model_dir} -> {bf16.name}")
            _run(["uv", "run", "python", LLAMA / "convert_hf_to_gguf.py", model_dir,
                  "--outfile", bf16, "--outtype", "bf16"], log_path)
        log(f"export: quantizing -> {quant_path.name}")
        _run([BIN / "llama-quantize", bf16, quant_path, quant], log_path)
        if not keep_bf16:
            bf16.unlink(missing_ok=True)

    log("export: load-verify via llama-server one-shot completion")
    from moep.codeval import completion as server_completion, llama_server

    with llama_server(quant_path, port=8099, log_file=log_path):
        text = server_completion(8099, "def fibonacci(n):", n_predict=48)
    if len(text.strip()) < 10:
        raise RuntimeError(f"load-verify produced no usable completion: {text!r}")

    log("export: llama-bench")
    bench = subprocess.run(
        [BIN / "llama-bench", "-m", quant_path, "-p", "512", "-n", "128", "-o", "json"],
        capture_output=True, text=True, timeout=1800)
    bench_rows = json.loads(bench.stdout) if bench.returncode == 0 else []

    report = {
        "gguf": str(quant_path),
        "quant": quant,
        "size_gb": round(quant_path.stat().st_size / 1e9, 2),
        "smoke_completion": text[:600],
        "bench": [{"test": f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r.get('n_gen', 0)}",
                   "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts")} for r in bench_rows],
    }
    (out_dir / f"{name}-{quant}.export.json").write_text(json.dumps(report, indent=2))
    log(f"export: done {report['size_gb']} GB, bench {report['bench']}")
    return report
