"""Run directories and stage manifests. Every stage records params, outputs, and evidence;
a stage whose manifest says ok and whose outputs exist is skipped on re-run (idempotency)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Run:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, stage: str) -> Path:
        return self.path / f"{stage}.json"

    def done(self, stage: str) -> bool:
        p = self.manifest_path(stage)
        if not p.exists():
            return False
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            return False
        return m.get("status") == "ok" and all(Path(o).exists() for o in m.get("outputs", {}).values())

    def load(self, stage: str) -> dict:
        return json.loads(self.manifest_path(stage).read_text())

    def record(self, stage: str, params: dict, outputs: dict, evidence: dict, t0: float) -> dict:
        m = {
            "stage": stage,
            "status": "ok",
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_s": round(time.time() - t0, 1),
            "params": params,
            "outputs": {k: str(v) for k, v in outputs.items()},
            "evidence": evidence,
        }
        self.manifest_path(stage).write_text(json.dumps(m, indent=2))
        return m
