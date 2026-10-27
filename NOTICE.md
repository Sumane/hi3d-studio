# Third-party notices

Hi3D Studio is glue code and a user interface. None of the components below are
included in this repository; `install.sh` downloads or builds them, or you supply
them. Each is used under its own licence, which you accept when you install it.

| Component | Used for | Obtained by | Licence |
|---|---|---|---|
| **Hi3D v1.0** (SPARC3D) source | Refinement stage | You supply it (`HI3D_SRC=...`) | Hi3D Open License Agreement v1.0 |
| **Hi3D v1.0 weights** | Refinement stage | Downloaded from Hugging Face (`hi3d-official/Hi3Dv1.0`) | Hi3D Open License Agreement v1.0 |
| TRELLIS (Microsoft), PTv3 / Pointcept, O-CNN, NVIDIA FlexiCubes | Inside Hi3D | Part of the Hi3D source | MIT / MIT / MIT / Apache-2.0 (see Hi3D's `THIRD_PARTY_LICENSES.md`) |
| **TRELLIS.2** GGUF models | Coarse stage | You supply them (`TRELLIS_MODELS=...`) | MIT (TRELLIS.2); DINOv3 under Meta's DINOv3 licence |
| **pixal3d.cpp** (raven38, fork of trellis.cpp) | Coarse engine, texture bake (`post-replay`) | Cloned and built by `install.sh` | MIT |
| DINOv2-large, CLIP ViT-L/14 | Hi3D image conditioning | Downloaded from Hugging Face | Apache-2.0 / MIT |
| PyTorch, torchvision | Runtime | pip | BSD-3-Clause |
| spconv / cumm (fafraob cu128 wheels) | Sparse convolutions | pip | Apache-2.0 |
| flash-attn | Attention kernels | pip / source build | BSD-3-Clause |
| pymeshlab (MeshLab) | Topology-safe decimation | pip | GPL-3.0 |
| scikit-image, SciPy, NumPy, trimesh, pyfqmr | Mesh processing | pip | BSD / BSD / BSD / MIT / MIT |
| rembg, onnxruntime | Background removal | pip | MIT |
| FastAPI, Uvicorn | Local server | pip | MIT / BSD-3-Clause |
| three.js | 3D viewer | Vendored by `install.sh` | MIT |

## Hi3D attribution requirement

The Hi3D Open License Agreement v1.0 permits commercial use but requires products
built with it to show **"Powered by Hi3D 1.0"**, and sets further terms above
200,000 monthly active users or USD 1M monthly revenue. It also includes an
Acceptable Use Policy. Read the full agreement shipped with the Hi3D source
before using outputs commercially.

## pymeshlab (GPL-3.0)

Hi3D Studio imports pymeshlab at run time but does not include or redistribute
it. If you redistribute a bundle that *includes* pymeshlab (for example a packaged
environment), the GPL-3.0 terms apply to that bundle. Set `HI3D_STOCK_SIMPLIFY=1`
to run without it (Hi3D's original simplification is then used).
