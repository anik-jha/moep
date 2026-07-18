"""Self-repair eval (E1): single-shot pass@1 vs one execution-feedback repair turn.

Thesis under test: single-shot evals mispredict how viable a compressed MoE is in an
agentic loop, where the model sees its failure and retries. Kill criterion (plan
strat-e1): base-vs-reap25 repair-delta gap < 3 pass@1 points.

Turn 1 is run_codeval unchanged - the same code path behind every recorded codeval_*
number, so single-shot scores stay comparable with the paper tables. Turn 2 replays
the evalplus chat context (its instruction + task prompt as the user turn, the failed
solution as the assistant turn) and appends a feedback message built from the first
failing test input recorded in eval_results.json, plus the expected output computed
by executing the dataset's canonical solution (trusted code) in the evalplus env.
The merged solution set (turn-1 passes untouched, turn-2 attempts for failures) is
re-scored by evalplus.evaluate; per-item transitions land next to the samples.

Repair artifacts live under evalplus/<tag>/<dataset>-repair/, deliberately outside
the <dataset> tree that _select_samples_file globs for turn-1 samples.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .codeval import (_preflight_chat, _select_samples_file, evaluate_samples,
                      llama_server, run_codeval)

# Verified against the installed evalplus 0.3.1 (uv cache, evalplus/codegen.py:176 and
# evalplus/provider/openai.py:24): the openai chat backend sends exactly one user
# message of INSTRUCTION_PREFIX + "\n```python\n{prompt.strip()}\n```". The repair
# conversation must reconstruct that turn verbatim so the assistant's failed solution
# sits in the context it was actually generated in.
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the "
                      "following problem in a markdown code block:")
# evalplus/provider/base.py:13 default; turn 2 gets the same budget as turn 1.
MAX_NEW_TOKENS = 768


def turn1_user_message(task_prompt: str) -> str:
    return INSTRUCTION_PREFIX + f"\n```python\n{task_prompt.strip()}\n```"


def build_feedback(args_repr: str | None, expected_repr: str | None) -> tuple[str, str]:
    """Return (feedback message, feedback mode) for a failed task.

    Modes degrade with available information: failing call + expected output when the
    canonical solution produced one, failing call alone when it did not, and a generic
    notice when eval_results recorded no failing input (e.g. the solution crashed
    before any test ran).
    """
    fix = ("Identify the bug and rewrite the complete corrected function in a "
           "markdown code block. Keep the same function signature.")
    if args_repr and expected_repr:
        return (f"Your solution failed its unit tests. For example, `{args_repr}` "
                f"should return `{expected_repr}`, but your function does not. "
                f"{fix}", "input_expected")
    if args_repr:
        return (f"Your solution failed its unit tests. For example, it fails on "
                f"`{args_repr}`. {fix}", "input_only")
    return (f"Your solution failed its unit tests. {fix}", "generic")


def build_repair_messages(task_prompt: str, failed_solution: str, feedback: str) -> list[dict]:
    return [
        {"role": "user", "content": turn1_user_message(task_prompt)},
        {"role": "assistant", "content": f"```python\n{failed_solution.strip()}\n```"},
        {"role": "user", "content": feedback},
    ]


def load_eval_results(path: Path) -> dict[str, dict]:
    """{task_id: first-sample record} from an evalplus eval_results.json (greedy => one
    sample per task)."""
    data = json.loads(path.read_text())["eval"]
    return {tid: recs[0] for tid, recs in data.items() if recs}


def failed_task_ids(results: dict[str, dict]) -> list[str]:
    """plus-suite failures; anything that is not an explicit pass counts as failed."""
    return sorted(t for t, r in results.items() if r.get("plus_status") != "pass")


def merge_solutions(turn1_rows: list[dict], failed_ids: set[str],
                    turn2_solutions: dict[str, str]) -> tuple[list[dict], int]:
    """Final agent state per task: turn-1 solution unless it failed and turn 2
    produced non-empty code. Returns (merged rows, n tasks with no usable turn-2)."""
    merged, no_attempt = [], 0
    for row in turn1_rows:
        tid = row["task_id"]
        if tid in failed_ids:
            t2 = (turn2_solutions.get(tid) or "").strip()
            if t2:
                row = {"task_id": tid, "solution": t2}
            else:
                no_attempt += 1
        merged.append(dict(row))
    return merged, no_attempt


def transitions(t1: dict[str, dict], t2: dict[str, dict]) -> dict:
    """Per-task turn-1 -> merged outcome plus summary counts. `regressed` (a turn-1
    pass failing re-evaluation of the identical solution) indicates test flakiness
    and is alarm-worthy, not silently absorbable."""
    per_task, counts = {}, {"repaired": 0, "still_fail": 0, "regressed": 0, "kept_pass": 0}
    for tid, r1 in t1.items():
        s1 = r1.get("plus_status")
        s2 = t2[tid].get("plus_status") if tid in t2 else None
        per_task[tid] = {"t1": s1, "t2": s2}
        if s1 == "pass":
            counts["kept_pass" if s2 == "pass" else "regressed"] += 1
        else:
            counts["repaired" if s2 == "pass" else "still_fail"] += 1
    return {"per_task": per_task, **counts}


def chat_completion(port: int, messages: list[dict], max_tokens: int = MAX_NEW_TOKENS) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": messages, "max_tokens": max_tokens,
                         "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def _dump_repair_context(dataset: str, eval_results: Path, out: Path):
    """Run the context extractor inside the evalplus env (dataset access + canonical
    execution live there; our venv deliberately does not carry evalplus pins)."""
    helper = Path(__file__).parents[2] / "scripts" / "e1_dataset_dump.py"
    r = subprocess.run(
        ["uv", "run", "--no-project", "--with", "evalplus", "python", str(helper),
         "--dataset", dataset, "--eval-results", str(eval_results), "--out", str(out)],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"e1_dataset_dump rc={r.returncode}: {r.stderr[-2000:]}")


def _sanitize(raw: Path) -> Path:
    """evalplus.sanitize writes <name>-sanitized.jsonl next to the input (verified
    evalplus 0.3.1 sanitize.py:193)."""
    r = subprocess.run(["uvx", "--from", "evalplus", "evalplus.sanitize",
                        "--samples", str(raw)],
                       capture_output=True, text=True, timeout=600)
    out = raw.parent / raw.name.replace(".jsonl", "-sanitized.jsonl")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"evalplus.sanitize rc={r.returncode}: {r.stderr[-2000:]}")
    return out


def run_repair_eval(gguf: str | Path, run_dir: str | Path, tag: str,
                    dataset: str = "humaneval", port: int = 8090, ctx: int = 16384,
                    parallel: int = 4, force: bool = False, log=print) -> dict:
    run_dir = Path(run_dir)
    rdir = run_dir / "evalplus" / tag / f"{dataset}-repair"
    if force and rdir.exists():
        import shutil
        shutil.rmtree(rdir)

    single = run_codeval(gguf, run_dir, tag, dataset=dataset, port=port, force=force, log=log)

    root = run_dir / "evalplus" / tag / dataset
    samples = _select_samples_file(root, dataset)
    results_path = samples.parent / (samples.name.replace(".jsonl", "_eval_results.json"))
    if not results_path.exists():
        raise FileNotFoundError(f"turn-1 eval_results missing: {results_path}")
    t1 = load_eval_results(results_path)
    failed = failed_task_ids(t1)
    turn1_rows = [json.loads(l) for l in samples.read_text().splitlines() if l.strip()]
    log(f"repaireval[{tag}/{dataset}]: {len(t1)} tasks, {len(failed)} plus-suite failures")

    result = {"gguf": str(gguf), "dataset": dataset, "tag": tag,
              "single": single["scores"], "n_tasks": len(t1), "n_fail_t1": len(failed)}
    if not failed:
        result |= {"repaired": single["scores"], "n_repaired": 0, "n_no_attempt": 0,
                   "n_regressed": 0, "repair_rate": None, "feedback_modes": {}}
        return result

    rdir.mkdir(parents=True, exist_ok=True)
    ctx_path = rdir / "context.json"
    _dump_repair_context(dataset, results_path, ctx_path)
    context = json.loads(ctx_path.read_text())

    jobs, modes = [], {}
    for tid in failed:
        c = context.get(tid)
        if c is None:
            raise RuntimeError(f"no repair context for failed task {tid} (helper bug)")
        feedback, mode = build_feedback(c.get("args_repr"), c.get("expected_repr"))
        modes[mode] = modes.get(mode, 0) + 1
        jobs.append({"task_id": tid,
                     "messages": build_repair_messages(c["prompt"], t1[tid]["solution"],
                                                       feedback),
                     "feedback_mode": mode})
    (rdir / "repair_prompts.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs))

    log(f"repaireval[{tag}/{dataset}]: generating {len(jobs)} repair turns "
        f"(modes {modes})")
    turn2: dict[str, str] = {}
    with llama_server(gguf, port, ctx=ctx, parallel=parallel,
                      log_file=rdir / "server.log", extra_args=("--reasoning", "off")):
        _preflight_chat(port)

        def _one(job):
            for attempt in (1, 2):
                try:
                    return job["task_id"], chat_completion(port, job["messages"])
                except Exception as e:
                    if attempt == 2:
                        log(f"repaireval[{tag}/{dataset}]: {job['task_id']} "
                            f"turn-2 failed twice: {e}")
                        return job["task_id"], ""

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for tid, content in pool.map(_one, jobs):
                turn2[tid] = content

    (rdir / "turn2.gen.jsonl").write_text(
        "".join(json.dumps({"task_id": t, "solution": s}) + "\n"
                for t, s in turn2.items()))
    sanitized = _sanitize(rdir / "turn2.gen.jsonl")
    t2_solutions = {r["task_id"]: r["solution"]
                    for r in (json.loads(l) for l in sanitized.read_text().splitlines()
                              if l.strip())}

    merged_rows, no_attempt = merge_solutions(turn1_rows, set(failed), t2_solutions)
    merged = rdir / "merged.jsonl"
    merged.write_text("".join(json.dumps(r) + "\n" for r in merged_rows))
    # evalplus.evaluate loads an existing eval_results file instead of re-scoring;
    # a stale one from an interrupted run must never be scored as this run's result.
    merged_results = merged.parent / (merged.name.replace(".jsonl", "_eval_results.json"))
    merged_results.unlink(missing_ok=True)
    repaired_scores = evaluate_samples(dataset, merged, rdir / "evaluate.log")
    tr = transitions(t1, load_eval_results(merged_results))
    (rdir / "transitions.json").write_text(json.dumps(tr, indent=2))

    result |= {"repaired": repaired_scores, "n_repaired": tr["repaired"],
               "n_no_attempt": no_attempt, "n_regressed": tr["regressed"],
               "repair_rate": round(tr["repaired"] / len(failed), 4),
               "feedback_modes": modes}
    log(f"repaireval[{tag}/{dataset}]: single {single['scores']} -> repaired "
        f"{repaired_scores} (repaired {tr['repaired']}/{len(failed)}, "
        f"regressed {tr['regressed']})")
    return result
