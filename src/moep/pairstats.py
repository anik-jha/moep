"""Paired statistics on EvalPlus per-task verdicts: HumanEval+/MBPP+ pass maps, paired bootstrap
CIs with a fixed seed, and exact McNemar tests. Shared by the paper statistics and later studies so
every comparison uses the same estimator."""

import json
import random
from math import comb
from pathlib import Path

BOOT_SEED, BOOT_N = 20260714, 10000


def plus_pass(tag: str, dataset: str, root: Path) -> dict[str, bool] | None:
    """task_id -> HumanEval+/MBPP+ pass (base AND plus tests) from evalplus eval_results under
    root/evalplus/<tag>/<dataset>. Excludes the unsanitized .raw_eval_results.json sibling
    (scoring fenced markdown gives a spurious all-fail)."""
    hits = [p for p in (root / "evalplus" / tag / dataset).glob("**/*_eval_results.json")
            if ".raw" not in p.name]
    if not hits:
        return None
    ev = json.loads(hits[0].read_text())["eval"]
    return {k: (v[0]["base_status"] == "pass" and v[0].get("plus_status") == "pass") for k, v in ev.items()}


def _mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(gained, lost) + 1)) / 2 ** n) if n else 1.0


def paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """b vs a: net, gained (b-only), lost (a-only), bootstrap CI, exact McNemar p."""
    t = sorted(set(a) & set(b))
    gained = sum(1 for k in t if b[k] and not a[k])
    lost = sum(1 for k in t if a[k] and not b[k])
    diff = (sum(b[k] for k in t) - sum(a[k] for k in t)) / len(t)
    rng = random.Random(BOOT_SEED)
    ds = sorted((sum(b[k] for k in s) - sum(a[k] for k in s)) / len(t)
                for s in ([rng.choice(t) for _ in t] for _ in range(BOOT_N)))
    return {"n": len(t), "net": gained - lost, "gained": gained, "lost": lost,
            "diff": round(diff, 4), "ci95": [round(ds[int(.025 * BOOT_N)], 4), round(ds[int(.975 * BOOT_N)], 4)],
            "mcnemar_p": round(_mcnemar(gained, lost), 4)}


def pooled_mcnemar(pairs: list[tuple[dict[str, bool], dict[str, bool]]]) -> dict:
    g = l = 0
    for a, b in pairs:
        t = set(a) & set(b)
        g += sum(1 for k in t if b[k] and not a[k])
        l += sum(1 for k in t if a[k] and not b[k])
    return {"gained": g, "lost": l, "mcnemar_p": round(_mcnemar(g, l), 4)}
