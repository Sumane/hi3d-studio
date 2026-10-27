#!/usr/bin/env python3
"""Post-process a finished job on the CPU: hybrid merge, decimation, TRELLIS PBR texture.

  .venv/bin/python scripts/postprocess.py outputs/<job-id> [options]

  --no-hybrid          use pure Hi3D geometry instead of the hybrid merge
  --decimate N         also write decimated_<N>.glb (geometry only, output frame)
  --texture            bake textured.glb from the job's TRELLIS PBR dump
  --tex-faces N        face budget for the textured mesh (0 = full; default 300000)
  --atlas PX           texture size (default 2048)
  --tau PCT            hybrid coverage threshold, % of extent (default 0.6)

Updates job.json so Hi3D Studio shows the new files. Emits "@@" JSON progress
lines when run by the server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh  # noqa: E402

from app import hybrid as hybrid_mod  # noqa: E402
from app.colour import export_glb, place_and_colour  # noqa: E402


def emit(kind: str, **kw) -> None:
    print("@@" + json.dumps({"type": kind, **kw}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--decimate", type=int, default=0)
    ap.add_argument("--texture", action="store_true")
    ap.add_argument("--tex-faces", type=int, default=300_000)
    ap.add_argument("--atlas", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=0.6)
    a = ap.parse_args()

    t0 = time.time()
    job = a.job.resolve()
    meta_path = job / "job.json"
    meta = json.loads(meta_path.read_text())
    if not (job / "refined.glb").exists():
        emit("error", message="this job has no Hi3D output (refined.glb)")
        return 1
    fit = meta.get("params", {}).get("fit_target")
    fit = Path(fit) if fit and Path(fit).exists() else None
    outputs = meta.setdefault("outputs", {})
    stats = meta.setdefault("stats", {})

    defaults = hybrid_mod.hybrid_merge.__defaults__
    hybrid_mod.hybrid_merge.__defaults__ = (a.tau / 100,) + defaults[1:]
    emit("stage", message="Placing" + ("" if a.no_hybrid else " + hybrid merge") + ("" if a.no_colour else " + colours"))
    mesh = trimesh.load(job / "refined.glb", force="mesh", process=False)
    out, info, final_c, to_place = place_and_colour(mesh, job / "coarse.glb", fit, colour=not a.no_colour,
                                                    hybrid=not a.no_hybrid, return_internal=True)
    for name, m in out.items():
        export_glb(m, job / f"{name}.glb")
        outputs[name] = f"{name}.glb"
        emit("log", message=f"wrote {name}.glb ({len(m.faces):,} faces)")
    stats["alignment"] = info

    if a.decimate:
        from app.texture import decimate
        emit("stage", message=f"Decimating to {a.decimate:,} faces")
        low = decimate(final_c, a.decimate)
        low.apply_transform(to_place)
        name = f"decimated_{a.decimate // 1000}k"
        export_glb(low, job / f"{name}.glb")
        outputs[name] = f"{name}.glb"
        emit("log", message=f"wrote {name}.glb ({len(low.faces):,} faces)")

    if a.texture:
        dump = job / "trellis_post.bin"
        if not dump.is_file():
            emit("log", message="No TRELLIS PBR dump in this job (it predates texture support) — re-run the job "
                                "to bake a texture")
        else:
            from app.texture import bake_job_mesh
            emit("stage", message="Baking TRELLIS PBR texture")
            tinfo = bake_job_mesh(final_c, to_place, dump, a.tex_faces, a.atlas,
                                  log=lambda m: emit("log", message=m), out_path=job / "textured.glb")
            outputs["textured"] = "textured.glb"
            stats["texture"] = tinfo
            emit("log", message=f"wrote textured.glb: {tinfo}")

    meta_path.write_text(json.dumps(meta, indent=1))
    emit("result", outputs=outputs, stats=stats, seconds=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
