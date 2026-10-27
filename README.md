# Hi3D Studio

A local desktop app for **image → high-detail 3D** on Linux. It chains two open models behind one UI:

```
reference image(s) ─► TRELLIS.2 coarse mesh ─► Hi3D refinement ─► solidify + clean ─► TRELLIS PBR texture bake ─► GLB
                      (pixal3d.cpp engine,     (Sparc3D sparse flow,   (one watertight     (UV unwrap + base colour
                       single or multi-view)    512³ / 1024³ / 1536³)   surface)            + metal/roughness maps)
```

Built and tested on an **RTX 5070 Ti (Blackwell, sm_120, 16 GB)**; the stack is also built for sm_86 (RTX 30-series).

> **This repository contains only Hi3D Studio's own code.** Hi3D, TRELLIS.2 and the model weights are third-party and are downloaded or supplied separately under their own licences. See [Licences](#licences).

## Features

- **Coarse stage (TRELLIS.2):** single image, multi-view (multidiffusion or stochastic, 2–8 views), or your own mesh.
- **Refine stage (Hi3D v1.0):** 512³ (1 pass), 1024³ (2 passes) or 1536³ (3 passes), with optional multi-view grounding (experimental).
- **Refine later:** generate TRELLIS only, inspect it, then run Hi3D on the results worth refining (or re-run at another detail level).
- **Clean output:** Hi3D's raw output is a thin, self-pinching shell that renders as rings and tears; the studio solidifies it into one watertight surface and snaps it back onto Hi3D's detail.
- **Hybrid thin parts:** optionally restores cloth, feathers and fins from TRELLIS where Hi3D loses them.
- **Texture:** TRELLIS.2's own UV unwrap and PBR bake applied to the *refined* mesh (`textured.glb`).
- **Part-by-part workflow:** fit any result onto a region of a whole-body proxy exported from Blender.
- **16 GB VRAM friendly:** 1536³ runs on 16 GB (transformers parked during decode, per-pass out-of-memory retry).
- **Viewer:** quad / orbit / split-compare, clay / normals / colour / wireframe.
- **Local HTTP API** for automation (`docs/API.md`, `scripts/hi3d_client.py`).

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (tested on Ubuntu-family) |
| GPU | NVIDIA, 16 GB VRAM recommended (12 GB works at 512³/1024³). Driver ≥ 570 |
| CUDA toolkit | ≥ 12.8 (to build the TRELLIS engine; prebuilt fallback otherwise) |
| Tools | `git cmake ninja gcc g++ curl tar`; if gcc is newer than nvcc accepts, also `gcc-14` |
| System libs | `libopengl0` (Ubuntu) / `libglvnd-opengl` (Fedora), used by MeshLab |
| Disk | ~40 GB free on the install drive |
| Browser | Chromium or Chrome (the launcher opens an app window) |

## Install

1. **Clone this repository**
   ```bash
   git clone https://github.com/<you>/hi3d-studio.git
   cd hi3d-studio
   ```
2. **Get the Hi3D v1.0 source** from its official release and unpack it anywhere. The installer copies it into `src/hi3d` (git-ignored).
3. **Have the TRELLIS.2 GGUF models** for trellis.cpp in one folder: `dinov3, ss_flow, ss_dec, shape_flow_512, shape_flow_1024, shape_dec, tex_flow_512, tex_flow_1024, tex_dec` (`.gguf`).
4. **Run the installer**, pointing it at both:
   ```bash
   HI3D_SRC=/path/to/hi3d-v1.0-main \
   TRELLIS_MODELS=/path/to/trellis-cpp-ggufs \
   ./install.sh
   ```

The installer:
- picks the Blackwell GPU (override with `HI3D_GPU=<nvidia-smi index>`),
- creates a Python 3.10 env with torch 2.8.0+cu128, spconv 2.3.8 (sm_120 build), flash-attn 2.8.3 and the app's dependencies,
- installs Hi3D without its pre-Blackwell version pins,
- clones and builds [`raven38/pixal3d.cpp`](https://github.com/raven38/pixal3d.cpp) (`trellis-server`, `trellis-cli`, `post-replay`) for sm_86 + sm_120,
- downloads the Hi3D weights (~7.7 GB) and the DINOv2 / CLIP encoders,
- runs a real kernel from every GPU library and prints PASS/FAIL,
- adds a **Hi3D Studio** launcher to the app menu.

Re-running is safe; finished steps are skipped. Other flags: `--skip-weights`, `--only-verify`.
All caches and temp files stay inside the install folder (`.cache/`), not on the system drive.

## Use

Launch **Hi3D Studio** from the app menu, or run `./hi3d-studio.sh`. It opens at `http://127.0.0.1:7860`.

1. **Drop 1–8 reference images.** ★ marks the front view. For multi-view: same object, same lighting, ~45–90° apart.
2. **1 · Coarse shape:** single image, multi-view, or upload your own mesh.
3. **2 · Hi3D refine:** choose the detail level. Turn the switch **off** to stop after TRELLIS; select any result in History later and click **Refine … with Hi3D**.
4. **3 · Placement (optional):** a proxy-part mesh to land the result on.
5. **4 · Post-process (optional):** decimate or re-bake the texture of a finished result (CPU only).

### Output files (`outputs/<job-id>/`)

| File | What it is |
|---|---|
| `coarse.glb` | Textured TRELLIS.2 mesh |
| `refined.glb` | Hi3D geometry, solidified and cleaned, unit-normalised |
| `refined_aligned.glb` | The same, placed in the coarse mesh's (or proxy's) frame |
| `refined_hybrid.glb` | Hi3D body + TRELLIS thin parts (if Hybrid is on) |
| `refined_coloured.glb` | Final mesh with TRELLIS colours as vertex colours |
| `textured.glb` | Final mesh with UVs and baked PBR maps |
| `trellis_post.bin` | TRELLIS's PBR voxel field (~200 MB), kept for re-baking |

Details of every parameter and endpoint: [`docs/API.md`](docs/API.md).

## How the Hi3D output is cleaned

Hi3D's raw FlexiCubes mesh (~9M faces at 1024³) is a shell about one voxel thick that keeps pinching shut across the surface. Rendered directly it shows topographic rings and speckle, and Hi3D's own pyfqmr simplification tears it into tens of thousands of pieces (measured: 1 piece / 98 non-manifold edges before, 58,411 pieces / 77,171 non-manifold edges after). This happens with the stock pipeline too.

`app/simplify.py` replaces that step:
1. rasterise the shell, close the pinch holes, flood-fill the outside to get the solid,
2. extract one watertight surface (marching cubes),
3. snap it back onto Hi3D's raw surface so the detail is Hi3D's,
4. decimate with MeshLab's topology-preserving quadric collapse.

Result on the same mesh: 1 piece, 0 non-manifold edges, watertight. Trade-off: gaps narrower than ~6 Hi3D voxels are closed. Set `HI3D_STOCK_SIMPLIFY=1` to use Hi3D's original step.

## Diagnostics

| Script | Purpose |
|---|---|
| `./install.sh --only-verify` | Re-run the GPU stack check |
| `scripts/check_spconv.py` | Compare spconv and flash-attn against PyTorch reference maths on your GPU |
| `scripts/official_refine.py <job> [--keep-raw]` | Re-run Hi3D on a job with the stock pipeline (A/B against the studio); `--keep-raw` saves the pre-simplification mesh |
| `scripts/postprocess.py <job>` | Re-place, hybrid, decimate and/or re-bake a finished job on the CPU |
| `scripts/rehybrid.py <job>` | Re-run only placement + hybrid + colours |

Logs: `logs/server.log` and the **Log** panel under the viewer.

## Repository layout

```
install.sh              installer (env, engine build, weights, launcher)
hi3d-studio.sh          launcher: starts the server and opens the app window
app/server.py           FastAPI job queue + file serving + UI
app/coarse.py           trellis-server adapter (single + multi-image)
app/refine_worker.py    Hi3D worker: VRAM management, multi-view fusion, per-pass OOM retry
app/simplify.py         solidify + topology-safe simplification of Hi3D output
app/colour.py           placement (ICP) and TRELLIS colour transfer; GLB export
app/hybrid.py           restore thin parts from TRELLIS
app/texture.py          TRELLIS UV + PBR bake onto the final mesh (post-replay)
app/static/             UI (three.js is vendored by the installer)
scripts/                client, post-processing and diagnostics
docs/API.md             HTTP API and output reference
```

Generated at install/run time and git-ignored: `.venv/ .cache/ src/ engines/ models/ hf-cache/ outputs/ logs/ config.json`.

## Known limitations

- **Faces at low detail:** at 512³ Hi3D can reshape eyes and mouths; use 1024³+ or keep TRELLIS's head.
- **Multi-view Hi3D grounding** is an addition of this app, not validated by the Hi3D authors.
- **Multi-image TRELLIS.2** is marked experimental upstream (pixal3d.cpp).
- **1536³** was only validated upstream on a 96 GB GPU; here it relies on reduced pass-3 budgets.

## Licences

**Hi3D Studio code** (this repository): MIT, see [`LICENSE`](LICENSE).

Third-party components are **not** included in this repository and keep their own terms. See [`NOTICE.md`](NOTICE.md). In short:
- **Hi3D v1.0** (source and weights): Hi3D Open License Agreement v1.0. Commercial use requires visibly showing **"Powered by Hi3D 1.0"**, and additional terms apply above 200k monthly active users or $1M monthly revenue. Read the full agreement shipped with Hi3D.
- **TRELLIS.2 / trellis.cpp / pixal3d.cpp:** MIT.
- **DINOv3** (TRELLIS.2 conditioning): Meta's DINOv3 licence.
- **DINOv2, CLIP, spconv, flash-attn, three.js, MeshLab (pymeshlab)** and others: their respective licences.
