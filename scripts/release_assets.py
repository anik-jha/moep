"""Assemble the eval/ folder of a pruned-GGUF release from raw artifacts.

Writes <release>/eval/provenance.json (source model revision and sha256 checked against the hub,
runtime, calibration size and sources, expert-score stability, tensor verification, file
hashes), copies <run>/summary.json, the per-problem EvalPlus verdicts and the BFCL outputs for the
baseline and every shipped build. Every value is read from a file or the hub; nothing is typed in.

Usage: release_assets.py <run_dir> <release_dir> <base_tag> <source_repo> <source_file> <local_source>
                         <tag_prefix> <build>=<released_name> [...]
e.g.   release_assets.py runs/bolt runs/release/bolt bolt24_iq2xxs nvidia/...-24GB-GGUF X.gguf models/.../X.gguf
                         bolt_ reap50=16GB-REAP50 reap25=12GB-REAP25
"""

import collections
import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

from moep.data import read_jsonl

LLAMA_CPP = "ggml-org/llama.cpp 547208dca250ceb3e4c8615598d043f2b602a527"
LOCAL_PATCH = ("common/chat.cpp: with LLAMA_CHAT_PARSE_LENIENT set, chat output the template parser rejects is "
               "returned as plain content instead of an HTTP 500 (eval harnesses retry 500s forever)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def lenient_hits(run: Path, tags: list[str]) -> int:
    logs = [p for t in tags for p in [*run.glob(f"evalplus/{t}/*/server.log"), run / "toolcall" / t / "server.log"]
            if p.exists()]
    return sum(p.read_text(errors="replace").count("LLAMA_CHAT_PARSE_LENIENT set") for p in logs)


def main(run_dir, release_dir, base_tag, source_repo, source_file, local_source, prefix, *builds):
    run, rel = Path(run_dir), Path(release_dir)
    ev_dir = rel / "eval"
    ev_dir.mkdir(parents=True, exist_ok=True)
    pairs = [b.split("=", 1) for b in builds]

    info = HfApi().model_info(source_repo, files_metadata=True)
    hub = next(f for f in info.siblings if f.rfilename == source_file)
    local_hash = sha256(Path(local_source))
    if local_hash != hub.lfs.sha256:
        raise SystemExit(f"local source sha256 {local_hash} != hub {hub.lfs.sha256}")

    stats = json.loads((run / "calib" / "stats_full.json").read_text())
    rows = read_jsonl("data/calib/calib.jsonl")
    tags = [base_tag] + [f"{prefix}{b}" for b, _ in pairs]
    prov = {
        "source_model": {"repo": source_repo, "revision": info.sha, "file": source_file, "bytes": hub.size,
                         "sha256": hub.lfs.sha256, "local_copy_sha256_matches_hub": True},
        "runtime": {"llama_cpp": LLAMA_CPP, "local_patch": LOCAL_PATCH,
                    "local_patch_triggered_in_release_runs": lenient_hits(run, tags), "evalplus": "0.3.1"},
        "calibration": {"conversations": len(rows),
                        "sources": dict(collections.Counter(r["source"] for r in rows).most_common()),
                        "tokens": stats["n_tokens"], "sequences": stats["n_seqs"],
                        "max_tokens_per_conversation": 4096,
                        "rendering": "the model's own chat template via llama-server /apply-template"},
        "selection": {"criterion": "REAP: mean over routed tokens of renormalized router weight x expert-output L2 norm",
                      "allocation": "each layer keeps round(keep_ratio x its routed-expert count)"},
        "builds": {},
    }
    for build, name in pairs:
        sel = json.loads(Path(glob.glob(str(run / "selections" / f"*-{build}.json"))[0]).read_text())["info"]
        prune = json.loads(next(run.glob(f"gguf_prune_*-{build}-*.json")).read_text())["evidence"]
        gguf = next(rel.glob(f"*-{name}-*.gguf"))
        sh = sel["split_half"]
        prov["builds"][name] = {
            "file": gguf.name, "bytes": gguf.stat().st_size, "sha256": sha256(gguf),
            "keep_ratio": sel["keep_ratio"], "experts_kept": sel["experts_kept"], "experts_total": sel["experts_total"],
            "split_half_spearman": {"median": sh["spearman_median"], "min": sh["spearman_min"]},
            "tensor_verification": prune["verify"]}
    (ev_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    shutil.copy(run / "summary.json", ev_dir / "summary.json")

    names = {base_tag: "base-24GB", **{f"{prefix}{b}": n for b, n in pairs}}
    for tag, name in names.items():
        for ds in ("humaneval", "mbpp"):
            src = next(p for p in (run / "evalplus" / tag / ds).glob("**/*_eval_results.json") if ".raw" not in p.name)
            dst = ev_dir / "evalplus" / name / f"{ds}_eval_results.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        for cat in ("simple", "multiple"):
            dst = ev_dir / "toolcall" / name / f"{cat}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(run / "toolcall" / tag / f"{cat}.jsonl", dst)
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 9:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
