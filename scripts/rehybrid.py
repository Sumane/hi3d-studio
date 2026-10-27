#!/usr/bin/env python3
"""Re-run placement + hybrid merge + colour on an existing job, without the GPU.

  .venv/bin/python scripts/rehybrid.py outputs/<job-id> [--no-hybrid] [--no-colour] [--tau 0.6]

Rewrites refined_aligned / refined_hybrid / refined_coloured in that job folder
and updates job.json (restart or refresh Hi3D Studio to see them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh  # noqa: E402

from app import hybrid as hybrid_mod  # noqa: E402
from app.colour import export_glb, place_and_colour  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--tau", type=float, default=0.6, help="coverage threshold, %% of model extent (default 0.6)")
    a = ap.parse_args()

    job = a.job.resolve()
    meta_path = job / "job.json"
    meta = json.loads(meta_path.read_text())
    refined = job / "refined.glb"
    if not refined.exists():
        print("no refined.glb in", job)
        return 1
    fit = meta.get("params", {}).get("fit_target")
    fit = Path(fit) if fit and Path(fit).exists() else None

    default = hybrid_mod.hybrid_merge.__defaults__
    hybrid_mod.hybrid_merge.__defaults__ = (a.tau / 100,) + default[1:]
    print("loading", refined.name, "...", flush=True)
    mesh = trimesh.load(refined, force="mesh", process=False)
    out, info = place_and_colour(mesh, job / "coarse.glb", fit, colour=not a.no_colour, hybrid=not a.no_hybrid)
    for name, m in out.items():
        export_glb(m, job / f"{name}.glb")
        meta.setdefault("outputs", {})[name] = f"{name}.glb"
        print("wrote", f"{name}.glb", f"({len(m.faces):,} faces)")
    meta.setdefault("stats", {})["alignment"] = info
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
