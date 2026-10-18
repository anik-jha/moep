"""Render the calibration mix with a GGUF's own chat template, for native/moe_stats.

Same row handling as moep.data.tokenize_rows: consecutive same-role turns are merged and rows the
template rejects are skipped and counted. Rendering goes through llama-server's /apply-template,
so the text matches what the model sees at inference.

Usage: render_calib.py <model.gguf> <calib.jsonl> <out.txt> [server_bin]
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from moep.codeval import SERVER_BIN, llama_server
from moep.data import read_jsonl

DOC_SEP = "<|moe_stats_doc|>"
PORT = 8096


def merged(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append(dict(m))
    return out


def apply_template(messages: list[dict]) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/apply-template",
        data=json.dumps({"messages": messages}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["prompt"]


def main(gguf: str, calib: str, out: str, server: str = str(SERVER_BIN)) -> int:
    rows = read_jsonl(calib)
    docs, skipped = [], 0
    with llama_server(gguf, PORT, ctx=4096, server=server, log_file=Path(out).with_suffix(".server.log")):
        for r in rows:
            try:
                docs.append(apply_template(merged(r["messages"])))
            except urllib.error.HTTPError:
                skipped += 1
    Path(out).write_text(f"\n{DOC_SEP}\n".join(docs))
    print(f"rendered {len(docs)}/{len(rows)} rows ({skipped} rejected by the template) -> {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
