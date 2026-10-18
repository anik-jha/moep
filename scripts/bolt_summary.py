"""A7 stage-3 summary for the pruned Nemotron 3.5 Bolt builds, from raw results only.

Per build: HE+/MBPP+ pass@1 recomputed from evalplus per-task verdicts, paired difference vs
Bolt-24GB (bootstrap CI, McNemar), retention vs Bolt-24GB, degenerate generations, BFCL tool-call
validity/correctness, measured GPU memory, and llama-bench speed. Per tier, the shipped criterion
is the one with the higher pooled HE+ and MBPP+ pass rate (ties go to REAP), as pre-registered in
runs/plan.json before any stage-3 data existed. Writes runs/bolt/summary.json.
"""

import json
from pathlib import Path

from moep.evalx import _degenerate
from moep.pairstats import paired, plus_pass

R = Path("runs/bolt")
BASE = "bolt24_iq2xxs"
TIERS = {"50": ("reap50", "gate_mass50"), "25": ("reap25", "gate_mass25")}


def degenerate_count(tag: str) -> dict:
    out = {}
    for ds in ("humaneval", "mbpp"):
        raw = next((R / "evalplus" / tag / ds).glob("**/*.raw.jsonl"))
        sols = [json.loads(line)["solution"] for line in raw.read_text().splitlines() if line]
        out[ds] = {"n": len(sols), "degenerate": sum(_degenerate(s) for s in sols)}
    return out


def speed(bench: list[dict], gguf: str) -> dict:
    rows = [r for r in bench if Path(r["model_filename"]).name == Path(gguf).name]
    return {("prefill_tps" if r["n_prompt"] else "decode_tps"): round(r["avg_ts"], 1) for r in rows}


def main():
    ev = json.loads((R / "eval.json").read_text())
    bench = json.loads((R / "bench.json").read_text())
    base = {ds: plus_pass(BASE, ds, R) for ds in ("humaneval", "mbpp")}
    bolt_gguf = ev[f"codeval_{BASE}_humaneval"]["gguf"]
    out = {"baseline": {"tag": BASE, "gguf": bolt_gguf,
                        **{ds: round(sum(p.values()) / len(p), 4) for ds, p in base.items()},
                        "toolcall": ev[f"toolcall_{BASE}"], "memory": ev[f"memory_{BASE}"]["by_ctx"],
                        "speed": speed(bench, bolt_gguf)},
           "builds": {}, "shipped": {}}
    for crits in TIERS.values():
        for c in crits:
            tag = f"bolt_{c}"
            row = {"gguf": ev[f"codeval_{tag}_humaneval"]["gguf"]}
            for ds in ("humaneval", "mbpp"):
                p = plus_pass(tag, ds, R)
                rate = sum(p.values()) / len(p)
                row[ds] = {"pass": round(rate, 4), "retention_vs_24gb": round(rate / out["baseline"][ds], 4),
                           "paired_vs_24gb": paired(base[ds], p)}
            pooled = [v for ds in ("humaneval", "mbpp") for v in plus_pass(tag, ds, R).values()]
            row["pooled_pass"] = round(sum(pooled) / len(pooled), 4)
            row["degenerate"] = degenerate_count(tag)
            row["toolcall"] = ev[f"toolcall_{tag}"]
            row["memory"] = ev[f"memory_{tag}"]["by_ctx"]
            row["speed"] = speed(bench, row["gguf"])
            out["builds"][c] = row
    for tier, (reap, gm) in TIERS.items():
        a, b = out["builds"][reap]["pooled_pass"], out["builds"][gm]["pooled_pass"]
        out["shipped"][tier] = reap if a >= b else gm
    (R / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"shipped": out["shipped"],
                      **{c: {k: b[k] for k in ("pooled_pass",)} | {ds: b[ds]["pass"] for ds in ("humaneval", "mbpp")}
                         for c, b in out["builds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
