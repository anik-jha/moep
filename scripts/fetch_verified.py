"""Download files from a Hugging Face repo and verify them byte-for-byte by size against the hub.

Writes the marker file only after every requested file exists locally at its hub size, so a
watcher or a plan.json done_when probe never fires on a partial download. Idempotent:
re-running skips files that already match.

Usage: fetch_verified.py <repo_id> <local_dir> <marker> <pattern> [<pattern> ...]
"""

import fnmatch
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main(repo_id: str, local_dir: str, marker: str, *patterns: str) -> int:
    api = HfApi()
    out = Path(local_dir)
    Path(marker).unlink(missing_ok=True)
    remote = {f.rfilename: f.size for f in api.model_info(repo_id, files_metadata=True).siblings}
    wanted = sorted(n for n in remote if any(fnmatch.fnmatch(n, p) for p in patterns))
    if not wanted:
        print(f"no files in {repo_id} match {patterns}")
        return 1
    total = sum(remote[n] for n in wanted)
    print(f"{repo_id}: {len(wanted)} files, {total / 2**30:.2f} GiB -> {out}")
    for name in wanted:
        dest = out / name
        if dest.exists() and dest.stat().st_size == remote[name]:
            print(f"=== already complete: {name}")
            continue
        print(f"=== downloading {name} ({remote[name] / 2**30:.2f} GiB)")
        hf_hub_download(repo_id, name, local_dir=out)
    bad = [n for n in wanted if not (out / n).exists() or (out / n).stat().st_size != remote[n]]
    for n in wanted:
        print(f"{'MISMATCH' if n in bad else 'OK'} {n}: hub={remote[n]} local={(out / n).stat().st_size if (out / n).exists() else None}")
    if bad:
        return 1
    Path(marker).parent.mkdir(parents=True, exist_ok=True)
    Path(marker).write_text(f"{repo_id} {len(wanted)} files verified\n")
    print("FETCH_VERIFIED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
