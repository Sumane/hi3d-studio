"""One-shot Hi3D refinement process: python -m app.refine_worker '<json spec>'.

Runs in its own process so all VRAM is returned when it exits. Emits machine-
readable lines prefixed with "@@" for the job server; everything else is log.

Multi-view fusion (experimental): Hi3D was trained on a single conditioning
image. ``multidiffusion`` averages the conditional flow prediction over every
view at each solver evaluation, then applies guidance once against the null
condition — the training-free scheme TRELLIS uses for multi-image input.
``stochastic`` conditions each evaluation on one view in turn. Neither mode is
validated by the Hi3D authors.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def emit(kind: str, **payload) -> None:
    print("@@" + json.dumps({"type": kind, **payload}), flush=True)


def build_pipeline_class():
    import torch
    from sparc3d import Sparc3DPipeline
    import sparc3d.pipelines.sparc3d_pipeline as sp
    from sparc3d.schedulers.schedule import apply_effective_guidance

    class StudioPipeline(Sparc3DPipeline):
        fusion: str = "front"
        view_weights: torch.Tensor | None = None
        _eval_counter: int = 0

        # Keep only one 1.8B transformer resident: park the 512 model on the CPU
        # before the 1024 model is materialised for passes 2 and 3.
        @property
        def transformer_1024(self):
            if (self._transformer_1024 is None and not self._sequential_cpu_offload
                    and self._device.type == "cuda"):
                emit("log", message="Moving transformer_512 to CPU before loading transformer_1024")
                self.transformer_512.to("cpu")
                torch.cuda.empty_cache()
            return Sparc3DPipeline.transformer_1024.fget(self)

        def _guided_velocity(self, transformer, sample, timestep, condition, negative_condition):
            views = condition.shape[0]
            negative = negative_condition[:1]
            if views == 1:
                return Sparc3DPipeline._guided_velocity(transformer, sample, timestep, condition, negative)
            if self.fusion == "stochastic":
                v = self._eval_counter % views
                self._eval_counter += 1
                return Sparc3DPipeline._guided_velocity(transformer, sample, timestep, condition[v:v + 1], negative)
            model_t = (1000.0 * timestep).reshape(1)
            w = self.view_weights
            positive = None
            for v in range(views):
                pred = transformer(sample, model_t, condition[v:v + 1]).feats * float(w[v])
                positive = pred if positive is None else positive + pred
            if float(timestep) < 0.7:
                return positive
            neg = transformer(sample, model_t, negative).feats
            return apply_effective_guidance(positive, neg, timestep)

        @torch.inference_mode()
        def refine(self, images, coarse_mesh, *, resolution, steps, seed, fusion, front_weight):
            from sparc3d.sparcubes.postprocess import validate_mesh
            from sparc3d.utils.image import prepare_image

            generator = self._resolve_generator(None, seed)
            self._validate_steps(steps)
            self._validate_target_resolution(resolution)
            self.fusion = fusion if len(images) > 1 else "front"
            used = images if self.fusion != "front" else images[:1]

            emit("stage", stage="encode", message=f"Encoding {len(used)} view(s) with DINOv2 + CLIP")
            if self._resident():
                self.conditioner.to(self._device)  # may have been parked by a previous attempt
            conds = [self.conditioner(prepare_image(im)) for im in used]
            condition = torch.cat(conds, 0).to(self._device)
            weights = torch.ones(len(used))
            weights[0] = front_weight
            self.view_weights = weights / weights.sum()
            self._eval_counter = 0
            self.pass_budgets_used = []
            if self._resident():
                self.conditioner.to("cpu")  # only needed once; frees ~2 GB
                torch.cuda.empty_cache()

            emit("stage", stage="voxelize", message="Voxelizing coarse mesh at 64³")
            mesh = validate_mesh(coarse_mesh)
            coords = self._adaptive_voxelize(mesh, 64, sp.PASS_BUDGETS[0])
            emit("log", message=f"{len(coords):,} occupied voxels in pass 1")
            emit("stage", stage="sample", message="Sampling sparse latents")
            return self._run_refinement(condition, coords, structure_resolution=64, resolution=resolution,
                                        num_inference_steps=steps, generator=generator)

        def _resident(self) -> bool:
            return not self._sequential_cpu_offload and self._device.type == "cuda"

        def _sample_latents(self, transformer, *args, **kwargs):
            # The transformer may have been parked on the CPU for the previous decode.
            if self._resident():
                transformer.to(self._device)
            t0 = time.time()
            out = super()._sample_latents(transformer, *args, **kwargs)
            emit("log", message=f"sampling took {time.time() - t0:.0f}s, "
                                f"peak VRAM so far {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                 if torch.cuda.is_available() else f"sampling took {time.time() - t0:.0f}s")
            return out

        def _decode_latents(self, latents, structure_resolution):
            # Decoding never touches the transformers, and at 192³ the decoder's
            # activations alone approach 9-10 GiB. Park both 1.8B models (~3.7 GiB each)
            # on the CPU first; _sample_latents moves the next one back.
            if self._resident():
                for t in (self.transformer_512, self._transformer_1024):
                    if t is not None:
                        t.to("cpu")
                torch.cuda.empty_cache()
            emit("stage", stage="decode", message=f"Decoding pass at {structure_resolution}³ structure "
                                                  f"({latents.feats.shape[0]:,} latent voxels)")
            return super()._decode_latents(latents, structure_resolution)

        def _extract_mesh(self, decoded):
            emit("stage", stage="extract", message="Extracting mesh (FlexiCubes)")
            return Sparc3DPipeline._extract_mesh(decoded)

        def _run_refinement(self, condition, voxel_coords, *, structure_resolution, resolution,
                            num_inference_steps, generator):
            """Same ladder as Sparc3DPipeline, but an out-of-memory in a later pass
            re-voxelizes the previous pass's mesh with a smaller budget and retries
            only that pass (the base version would lose all earlier work)."""
            import gc
            S, O = sp.STRUCTURE_RESOLUTIONS, sp.OUTPUT_RESOLUTIONS
            start, target = S.index(structure_resolution), O.index(resolution)
            ladder = list(range(start, target + 1))
            coords, mesh, prev_mesh = voxel_coords, None, None
            for call_i, li in enumerate(ladder):
                structure = S[li]
                budget = sp.PASS_BUDGETS[li]
                while True:
                    state = generator.get_state()
                    latents = decoded = None
                    oom = False
                    try:
                        transformer = self.transformer_512 if structure == 64 else self.transformer_1024
                        latents = self._sample_latents(transformer, condition, coords, structure,
                                                       num_inference_steps, generator,
                                                       pass_index=call_i, num_passes=len(ladder))
                        decoded = self._decode_latents(latents, structure)
                        if li < target and self._decoded_feature_count(decoded) > sp.DECODE_FEATURE_CEILING:
                            raise RuntimeError("decoded feature ceiling exceeded before another pass")
                        mesh = self._extract_mesh(decoded)
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    if not oom:
                        break
                    # Out of the except block so the traceback no longer pins the tensors.
                    latents = decoded = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    if prev_mesh is None or budget < 0.4 * sp.PASS_BUDGETS[li]:
                        raise torch.cuda.OutOfMemoryError(
                            f"pass {call_i + 1} ({structure}³) does not fit even at {budget:,} voxels")
                    budget = int(budget * 0.85)
                    emit("log", message=f"Out of VRAM in pass {call_i + 1} — retrying only this pass "
                                        f"with a {budget:,}-voxel budget")
                    coords = self._adaptive_voxelize(prev_mesh, structure, budget)
                    generator.set_state(state)
                self.pass_budgets_used.append(budget)
                del latents, decoded
                if li < target:
                    prev_mesh = mesh
                    coords = self._adaptive_voxelize(mesh, S[li + 1], sp.PASS_BUDGETS[li + 1])
            mesh = self._postprocess_mesh(mesh, target_face_count=sp.TARGET_FACE_COUNTS[target])
            from sparc3d.pipelines.output import Sparc3DPipelineOutput
            return Sparc3DPipelineOutput(mesh=mesh, resolution=resolution)

        def _postprocess_mesh(self, mesh, *, target_face_count):
            emit("stage", stage="postprocess", message=f"Cleaning and simplifying to ~{target_face_count:,} faces")
            if os.environ.get("HI3D_STOCK_SIMPLIFY") == "1":
                return super()._postprocess_mesh(mesh, target_face_count=target_face_count)
            # Hi3D's pyfqmr step tears dense meshes into tens of thousands of pieces;
            # app/simplify.py does the same job with a topology-preserving decimator.
            from app.simplify import clean_simplify, mesh_health
            grid = {512: 512, 1024: 640, 1536: 768}.get(getattr(self, "out_res", 1024), 640)
            out = clean_simplify(mesh, target_face_count, grid=grid, log=lambda m: emit("log", message=m))
            h = mesh_health(out)
            emit("log", message=f"Mesh: {h['faces']:,} faces, {h['pieces']} piece(s) "
                                f"(largest {h['largest_piece_pct']}%), {h['nonmanifold_edges']} non-manifold edges")
            return out

    return StudioPipeline


def load_image(path: str, remove_bg: str):
    from PIL import Image
    image = Image.open(path)
    image.load()
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        alpha = image.convert("RGBA").getchannel("A")
        has_alpha = alpha.getextrema()[0] < 250
    if remove_bg == "always" or (remove_bg == "auto" and not has_alpha):
        from sparc3d.utils.image import remove_background
        emit("log", message=f"Removing background from {Path(path).name} (rembg)")
        image = remove_background(image)
    return image


def main() -> int:
    spec = json.loads(sys.argv[1])
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        import torch
        import trimesh

        if spec.get("torch_conv") or os.environ.get("HI3D_TORCH_CONV") == "1":
            # Route every decoder convolution through Hi3D's own pure-PyTorch path
            # (normally only used past spconv's int32 limit at 1536) instead of the
            # cu128 spconv kernels. Slower decode, reference numerics.
            from sparc3d.ops import sparse_backend
            sparse_backend._SPCONV_INT32_BYTE_LIMIT = 0
            emit("log", message="Safe decode: PyTorch sparse convolutions (spconv kernels bypassed)")

        emit("stage", stage="load", message="Loading Hi3D weights")
        StudioPipeline = build_pipeline_class()

        def on_progress(ev):
            emit("progress", pass_index=ev.pass_index, num_passes=ev.num_passes, step=ev.step,
                 steps=ev.num_inference_steps, structure=ev.structure_resolution)

        pipe = StudioPipeline.from_pretrained(spec["model"], local_files_only=spec.get("offline", False),
                                              progress_callback=on_progress)
        if spec.get("offload"):
            pipe.enable_sequential_cpu_offload("cuda")
        else:
            pipe.to("cuda")

        paths = spec["images"]
        front = int(spec.get("front", 0))
        ordered = [paths[front]] + [p for i, p in enumerate(paths) if i != front]
        images = [load_image(p, spec.get("remove_bg", "auto")) for p in ordered]
        coarse = trimesh.load(spec["coarse"], force="mesh", process=False)

        import sparc3d.pipelines.sparc3d_pipeline as sp
        # The 192³ decode is the memory peak (~13.5 GiB at 55k voxels on a 16 GB card;
        # 77k did not fit). Start pass 3 at a budget that fits the card; the per-pass
        # retry shrinks it further only if needed.
        budgets = list(sp.PASS_BUDGETS)
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
        if spec.get("pass3_budget"):
            budgets[2] = int(spec["pass3_budget"])
        elif total_gib and total_gib < 20 and not spec.get("offload"):
            budgets[2] = min(budgets[2], int(65_000 * total_gib / 16))
        sp.PASS_BUDGETS = tuple(budgets)
        if int(spec["resolution"]) == 1536:
            emit("log", message=f"Pass budgets {sp.PASS_BUDGETS} for {total_gib:.0f} GiB")
        pipe.out_res = int(spec["resolution"])
        result = pipe.refine(images, coarse, resolution=int(spec["resolution"]), steps=int(spec["steps"]),
                             seed=int(spec["seed"]), fusion=spec.get("fusion", "front"),
                             front_weight=float(spec.get("front_weight", 2.0)))
        refined_path = out_dir / "refined.glb"
        from app.colour import export_glb
        export_glb(result.mesh, refined_path)
        outputs = {"refined": refined_path.name}
        stats = {"faces": int(len(result.mesh.faces)), "vertices": int(len(result.mesh.vertices)),
                 "pass_budgets": list(pipe.pass_budgets_used)}
        if torch.cuda.is_available():
            stats["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fit = spec.get("fit_target")
        emit("stage", stage="place", message=("Fitting refined mesh onto proxy part" if fit else
                                              "Placing refined mesh back in the coarse frame")
             + (" + transferring colours" if spec.get("colour", True) else ""))
        try:
            from app.colour import place_and_colour
            if spec.get("hybrid", True):
                emit("stage", stage="hybrid", message="Restoring thin parts (cloth, feathers) from TRELLIS where Hi3D lost them")
            meshes, info, final_c, to_place = place_and_colour(
                result.mesh, Path(spec["coarse"]), Path(fit) if fit else None,
                colour=spec.get("colour", True), hybrid=spec.get("hybrid", True), return_internal=True)
            if "hybrid" in info:
                h = info["hybrid"]
                emit("log", message=f"Hybrid: restored {h['patched_area_pct']}% of surface from TRELLIS, "
                                    f"dropped {h['fragments_dropped_faces']:,} fragment faces, "
                                    f"carved {h['ghost_faces_carved']:,} ghost faces")
            for name, mesh in meshes.items():
                export_glb(mesh, out_dir / f"{name}.glb")
                outputs[name] = f"{name}.glb"
            stats["alignment"] = info
            if fit and (info.get("proxy", {}).get("fit_mean_dist_pct") or 0) > 3:
                emit("log", message=f"Warning: proxy fit error {info['proxy']['fit_mean_dist_pct']}% of extent — "
                                    "check the part's placement in Blender")
        except Exception as exc:  # placement is post-processing; keep the raw result
            final_c = None
            emit("log", message=f"Placement/colour skipped: {exc}")

        dump = Path(spec.get("dump") or "")
        if spec.get("texture", True) and final_c is not None:
            if not dump.is_file():
                emit("log", message="No TRELLIS PBR dump for this job (engine didn't write one) — texture bake skipped")
            else:
                try:
                    from app.texture import bake_job_mesh
                    emit("stage", stage="texture", message="Baking TRELLIS PBR texture onto the final mesh")
                    tinfo = bake_job_mesh(final_c, to_place, dump, int(spec.get("tex_faces", 300_000)),
                                          int(spec.get("atlas", 2048)), log=lambda m: emit("log", message=m),
                                          out_path=out_dir / "textured.glb")
                    outputs["textured"] = "textured.glb"
                    stats["texture"] = tinfo
                except Exception as exc:  # noqa: BLE001
                    emit("log", message=f"Texture bake failed: {type(exc).__name__}: {exc}")

        stats["seconds"] = round(time.time() - t0, 1)
        emit("result", outputs=outputs, stats=stats)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        hint = ""
        if "out of memory" in str(exc).lower():
            hint = " — try a lower resolution or enable CPU offload"
        emit("error", message=f"{type(exc).__name__}: {exc}{hint}", traceback=tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
