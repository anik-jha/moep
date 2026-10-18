"""Routing invariants for a native/moe_stats output: in every MoE layer the routed-slot count equals
tokens x experts-per-token, and renormalized gate weights sum to one per token. A failure means
the collector read the wrong tensors for this architecture; stop before profiling at scale.

Usage: check_moe_stats.py <stats.json> <experts_per_token>
"""

import json
import sys


def main(path: str, n_used: str) -> int:
    d = json.load(open(path))
    k = int(n_used)
    bad = []
    for il, layer in d["layers"].items():
        tokens = sum(h["tokens"] for h in layer["halves"])
        slots = sum(sum(h["counts"]) for h in layer["halves"])
        gate = sum(sum(h["gate_sum"]) for h in layer["halves"])
        if slots != tokens * k or abs(gate - tokens) > 1e-3 * tokens:
            bad.append((il, tokens, slots, round(gate, 2)))
    print(f"{len(d['layers'])} MoE layers, {d['n_tokens']} tokens; violations: {bad[:5]}")
    return 1 if bad or not d["layers"] else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
