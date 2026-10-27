#!/usr/bin/env python3
"""Re-run Hi3D on a job using the stock Sparc3DPipeline, exactly as Hi3D's own
demo/refine_mesh.py does (no Hi3D Studio subclass, no fusion, no VRAM tricks).

  .venv/bin/python scripts/official_refine.py outputs/<job-id> [--res 1024] [--seed 0]

Writes outputs/<job-id>/official_<res>.glb. If this is clean and the studio's
refined.glb is not, the problem is in the studio's worker; if this is just as
layered, it is Hi3D (or the environment) itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

cfg = config.load()
os.environ.update(config.gpu_env(cfg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--res", type=int, default=1024, choices=(512, 1024, 1536))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-raw", action="store_true",
                    help="also save the FlexiCubes mesh before Hi3D's clean-up/simplify step (raw_<res>.ply)")
    a = ap.parse_args()

    import trimesh
    from sparc3d import Sparc3DPipeline
    from app.refine_worker import load_image

    job = a.job.resolve()
    meta = json.loads((job / "job.json").read_text())
    images = sorted((job / "inputs").glob("*"))
    front = int(meta.get("params", {}).get("front", 0))
    img_path = images[front] if front < len(images) else images[0]
    image = load_image(str(img_path), "auto")   # same cut-out as the studio
    print("image:", img_path.name)

    import torch

    class Fit16GB(Sparc3DPipeline):
        # The only change from stock: move the transformers to system RAM while the
        # decoder runs (it never uses them) and back before sampling. Moving weights
        # does not change any numbers; Hi3D's own sequential offload still OOMs here
        # because it loads transformer_1024 straight onto the GPU.
        def _decode_latents(self, latents, structure_resolution):
            for t in (self.transformer_512, self._transformer_1024):
                if t is not None:
                    t.to("cpu")
            torch.cuda.empty_cache()
            return super()._decode_latents(latents, structure_resolution)

        def _postprocess_mesh(self, mesh, *, target_face_count):
            if a.keep_raw:
                raw = job / f"raw_{a.res}.ply"
                mesh.export(raw)
                print(f"wrote {raw}  ({len(mesh.faces):,} faces, before simplification)")
            return super()._postprocess_mesh(mesh, target_face_count=target_face_count)

        def _sample_latents(self, transformer, *args, **kwargs):
            transformer.to(self._device)
            return super()._sample_latents(transformer, *args, **kwargs)

    pipe = Fit16GB.from_pretrained(cfg["hi3d_model"], local_files_only=False)
    pipe.to("cuda")
    coarse = trimesh.load(job / "coarse.glb", force="mesh", process=False)
    t0 = time.time()
    result = pipe(image, coarse, resolution=a.res, num_inference_steps=25, seed=a.seed)
    out = job / f"official_{a.res}.glb"
    result.mesh.export(out)
    print(f"wrote {out}  ({len(result.mesh.faces):,} faces, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
