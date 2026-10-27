# Hi3D Studio HTTP API

Local only (`127.0.0.1:7860`, port in `config.json`). This is the same API the UI uses, so an agent can drive the whole studio without a browser. Jobs run one at a time, in the order they were submitted.

`scripts/hi3d_client.py` wraps all of this. Start the server first with `./hi3d-studio.sh`, or run it headless:
`.venv/bin/python -m uvicorn app.server:app --port 7860`.

## Submit a job

`POST /api/jobs`, `multipart/form-data`:

| Part | Required | Meaning |
|---|---|---|
| `params` | yes | JSON settings (below) |
| `images` | yes, 1–8 | reference views (repeat the part). `params.front` picks the front view |
| `coarse_mesh` | no | your own coarse mesh (needs `coarse_mode: "upload"`) |
| `fit_mesh` | no | proxy-part mesh in proxy world space; results are registered onto it |

`params` (all optional except where noted):

```jsonc
{
  "coarse_mode": "single",        // single | multidiffusion | stochastic | upload
  "coarse_res": 1024,             // 512 | 1024 | 1536 (TRELLIS.2)
  "bg_removal": "auto",           // auto | birefnet | threshold (TRELLIS)
  "refine": true,                 // run Hi3D after TRELLIS
  "hi3d_res": 1024,               // 512 | 1024 | 1536
  "steps": 25,                    // Hi3D steps per pass
  "fusion": "front",              // Hi3D view grounding: front | multidiffusion | stochastic
  "front_weight": 2.0,            // multidiffusion front-view bias
  "remove_bg": "auto",            // Hi3D cut-out: auto (if no alpha) | always | never
  "colour": true,                 // carry TRELLIS texture onto refined mesh
  "hybrid": true,                 // restore thin parts Hi3D loses (cloth, feathers) from TRELLIS
  "texture": true,                // TRELLIS UV + PBR bake onto the final mesh -> textured.glb
  "tex_faces": 300000,            // face budget for textured.glb (0 = full mesh)
  "atlas": 2048,                  // texture size
  "offload": false,               // Accelerate CPU offload
  "seed": 0,
  "front": 0,                     // index of the front view among `images`
  "gpu": 1                        // nvidia-smi index; omit to use config.json
}
```

Response: `{"id": "20261026-201502-ab12"}`

## Poll

`GET /api/jobs/{id}` returns:
- `status`: `queued`, `running`, `done`, `error` or `cancelled`
- `progress` (0–1), `stage`, `message`
- `outputs` (name → file), `stats`, `error`, `log`

`GET /api/jobs` lists all jobs (without logs). `POST /api/jobs/{id}/cancel` cancels one, and `DELETE /api/jobs/{id}` deletes a finished one.

## Outputs

`GET /api/jobs/{id}/files/{file}`. Files are also on disk at `outputs/{id}/`.

| File | Frame | Content |
|---|---|---|
| `coarse.glb` | TRELLIS (unit cube) | textured TRELLIS.2 mesh |
| `coarse_aligned.glb` | proxy (only with `fit_mesh`) | the same textured mesh, registered onto the proxy part |
| `refined.glb` | Hi3D (centred, unit radius) | raw Hi3D geometry |
| `refined_aligned.glb` | proxy if `fit_mesh`, else coarse | Hi3D geometry, placed |
| `refined_hybrid.glb` | same as `refined_aligned` (only with `hybrid`) | Hi3D main body + TRELLIS geometry where Hi3D lost coverage; Hi3D's loose fragments dropped |
| `refined_coloured.glb` | same as `refined_aligned` | the final mesh (hybrid if on) plus linear vertex colours |
| `textured.glb` | same as `refined_aligned` (only with `texture`) | final mesh decimated to `tex_faces`, with TRELLIS-generated UVs and PBR maps (base colour + metallic/roughness) |
| `decimated_<N>k.glb` | same as `refined_aligned` | from post-processing: geometry-only decimation |
| `trellis_post.bin` | n/a | TRELLIS's sparse PBR voxel field (~200 MB), kept so textures can be re-baked |

## Refine later (coarse first, Hi3D on demand)

Submit with `"refine": false` to stop after TRELLIS. To run Hi3D on that result afterwards (or re-run it with different settings), call `POST /api/jobs/{id}/refine` with a JSON body of Hi3D settings, for example `{"hi3d_res": 1024, "hybrid": true, "texture": true}`. The accepted keys are `hi3d_res`, `steps`, `fusion`, `front_weight`, `remove_bg`, `seed`, `colour`, `hybrid`, `texture`, `tex_faces`, `atlas`, `offload`, `torch_conv` and `gpu`. The job is queued again on the same id and reuses its `coarse.glb`, reference images and `trellis_post.bin`, so the TRELLIS stage is skipped. Any earlier Hi3D files in the job are replaced.

## Post-process a finished job

`POST /api/jobs/{id}/post` with a JSON body `{"decimate": 60000, "texture": true, "tex_faces": 300000, "atlas": 4096, "hybrid": true}`. It runs on the CPU, so it can overlap a GPU job, and it adds files to the job. The same thing is available from the command line: `.venv/bin/python scripts/postprocess.py outputs/<id> --decimate 60000 --texture --atlas 4096`.

`stats.alignment` reports fit quality: `fit_mean_dist_pct` is the mean ICP distance as a percentage of the target's extent. Under ~1% is a clean lock; above ~3% is logged as a warning and needs checking in Blender. It also reports the starting rotation and the scale that were used.

## Part-by-part workflow

1. Generate the whole-body **proxy** (any resolution). Import `refined_aligned.glb` or `coarse.glb` into Blender; this defines the world frame.
2. For each part, select the matching region of the proxy and export it **as selection only, in world space, without applying transforms**. Crop the reference image(s) to that part.
3. Submit the crop(s) as `images` and the region as `fit_mesh`.
4. Import `refined_aligned.glb` (or `refined_coloured.glb`) for each part. They land on the proxy, ready to join.

Registration is a similarity transform (rotation, uniform scale, translation). It tries four starting rotations about +Y, then ICP. A part whose shape differs a lot from its proxy region, such as a reworked silhouette, can still lock imperfectly; check `fit_mean_dist_pct`.
