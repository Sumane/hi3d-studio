"""Hi3D Studio local server: job queue, file serving and the UI.

Light by design — no torch import here. Heavy work runs in child processes
(trellis-server, then the Hi3D worker) one job at a time, so each stage gets the
whole GPU and returns its memory when it finishes.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .coarse import TrellisServer

app = FastAPI(title="Hi3D Studio")
STATIC = Path(__file__).parent / "static"
config.OUTPUTS.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, "Job"] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, jid: str, params: dict, images: list[str], coarse_upload: str | None):
        self.id = jid
        self.dir = config.OUTPUTS / jid
        self.params = params
        self.images = images
        self.coarse_upload = coarse_upload
        self.status = "queued"
        self.stage = "queued"
        self.message = "Waiting for GPU"
        self.progress = 0.0
        self.log: list[str] = []
        self.outputs: dict[str, str] = {}
        self.stats: dict = {}
        self.error: str | None = None
        self.created = time.time()
        self.cancel = False
        self.proc: subprocess.Popen | None = None
        self.server: TrellisServer | None = None
        self.skip_coarse = False   # set when Hi3D is run later on an existing coarse result

    def add_log(self, line: str) -> None:
        self.log.append(time.strftime("%H:%M:%S ") + line)
        del self.log[:-600]

    def public(self, full: bool = True) -> dict:
        d = {"id": self.id, "status": self.status, "stage": self.stage, "message": self.message,
             "progress": round(self.progress, 4), "outputs": self.outputs, "stats": self.stats,
             "error": self.error, "created": self.created, "params": self.params,
             "images": [Path(p).name for p in self.images]}
        if full:
            d["log"] = self.log[-300:]
        return d

    def save(self) -> None:
        (self.dir / "job.json").write_text(json.dumps(self.public(full=False) | {"log": self.log[-300:]}, indent=1))


# ------------------------------------------------------------------ pipeline
def run_job(job: Job) -> None:
    p = job.params
    refine = bool(p.get("refine", True))
    coarse = job.dir / "coarse.glb"
    reuse = job.skip_coarse and coarse.is_file()
    coarse_share = 0.0 if reuse else (0.35 if refine else 1.0)
    cfg = config.load()
    if p.get("gpu") is not None:
        cfg["gpu_index"] = int(p["gpu"])
    job.add_log(f"Running on GPU {cfg['gpu_index']}")

    def check():
        if job.cancel:
            raise Cancelled()

    # ---- coarse
    fit = p.get("fit_target")
    if reuse:
        job.add_log("Refining the existing TRELLIS result (coarse stage skipped)")
    elif job.coarse_upload:
        shutil.copy(job.coarse_upload, coarse)
        job.add_log(f"Using uploaded coarse mesh {Path(job.coarse_upload).name}")
    else:
        job.stage, job.message = "coarse", "Starting TRELLIS.2 engine"
        mode = p.get("coarse_mode", "single")
        res = int(p.get("coarse_res", 1024))

        def cprog(frac, line):
            job.progress = coarse_share * frac
            job.message = line[-120:]

        with TrellisServer(res, job.add_log, cprog, gpu=cfg["gpu_index"],
                           dump_path=job.dir / "trellis_post.bin") as srv:
            job.server = srv
            check()
            ordered = [job.images[p.get("front", 0)]] + [x for i, x in enumerate(job.images) if i != p.get("front", 0)]
            if mode == "single" or len(ordered) == 1:
                job.message = f"TRELLIS.2 single-image @ {res}"
                srv.generate(Path(ordered[0]), coarse, int(p.get("seed", 0)), p.get("bg_removal", "auto"), str(p.get("faces", "auto")))
            else:
                job.message = f"TRELLIS.2 multi-image ({mode}, {len(ordered)} views) @ {res}"
                srv.generate_mv([Path(x) for x in ordered], coarse, int(p.get("seed", 0)), mode,
                                p.get("bg_removal", "auto"), str(p.get("faces", "auto")))
        job.server = None
    job.outputs["coarse"] = coarse.name
    if fit and Path(fit).exists() and not reuse:
        # Textured coarse mesh placed in the proxy's frame (keeps TRELLIS materials).
        try:
            from .colour import place_scene
            info = place_scene(coarse, Path(fit), job.dir / "coarse_aligned.glb")
            job.outputs["coarse_aligned"] = "coarse_aligned.glb"
            job.stats["coarse_fit"] = info
            job.add_log(f"Coarse mesh fitted to proxy part: {info}")
        except Exception as exc:  # noqa: BLE001
            job.add_log(f"Coarse proxy fit skipped: {exc}")
    job.progress = coarse_share
    check()
    if not refine:
        return

    # ---- refine
    job.stage, job.message = "refine", "Starting Hi3D"
    spec = {
        "model": cfg["hi3d_model"], "images": job.images, "front": p.get("front", 0),
        "coarse": str(coarse), "out_dir": str(job.dir), "resolution": int(p.get("hi3d_res", 1024)),
        "steps": int(p.get("steps", 25)), "seed": int(p.get("seed", 0)), "fusion": p.get("fusion", "front"),
        "front_weight": float(p.get("front_weight", 2.0)), "offload": bool(p.get("offload", False)),
        "torch_conv": bool(p.get("torch_conv", False)),
        "remove_bg": p.get("remove_bg", "auto"), "colour": bool(p.get("colour", True)),
        "fit_target": fit if fit and Path(fit).exists() else None,
        "hybrid": bool(p.get("hybrid", True)),
        "texture": bool(p.get("texture", True)),
        "tex_faces": int(p.get("tex_faces", 300_000)),
        "atlas": int(p.get("atlas", 2048)),
        "dump": str(job.dir / "trellis_post.bin"),
    }
    env = {**os.environ, **config.gpu_env(cfg)}
    cmd = [sys.executable, "-m", cfg.get("refine_worker_module", "app.refine_worker"), json.dumps(spec)]
    job.proc = subprocess.Popen(cmd, cwd=config.ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    result = None
    for raw in job.proc.stdout:
        line = raw.rstrip()
        if not line.startswith("@@"):
            if line:
                job.add_log(line)
            continue
        try:
            ev = json.loads(line[2:])
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "stage":
            job.message = ev["message"]
            job.add_log("▸ " + ev["message"])
        elif t == "log":
            job.add_log(ev["message"])
        elif t == "progress":
            done = ev["pass_index"] + ev["step"] / ev["steps"]
            job.progress = coarse_share + (1 - coarse_share) * 0.97 * done / ev["num_passes"]
            job.message = (f"Hi3D pass {ev['pass_index'] + 1}/{ev['num_passes']} ({ev['structure']}³) — "
                           f"step {ev['step']}/{ev['steps']}")
        elif t == "result":
            result = ev
        elif t == "error":
            job.add_log(ev.get("traceback", ""))
            raise RuntimeError(ev["message"])
    code = job.proc.wait()
    job.proc = None
    check()
    if code != 0 or result is None:
        raise RuntimeError(f"Hi3D worker exited with code {code}")
    job.outputs.update(result["outputs"])
    job.stats.update(result["stats"])


def worker_loop() -> None:
    while True:
        jid = QUEUE.get()
        job = JOBS.get(jid)
        if job is None or job.cancel:
            continue
        job.status = "running"
        try:
            run_job(job)
            job.status, job.stage, job.message, job.progress = "done", "done", "Finished", 1.0
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelled"
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.message = "error", str(exc), str(exc)[:300]
            job.add_log("ERROR: " + str(exc))
        finally:
            job.save()


threading.Thread(target=worker_loop, daemon=True).start()


def _load_history() -> None:
    for jf in sorted(config.OUTPUTS.glob("*/job.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        j = Job(d["id"], d.get("params", {}), [str(jf.parent / "inputs" / n) for n in d.get("images", [])], None)
        j.status = d.get("status", "done") if d.get("status") not in ("running", "queued") else "error"
        j.stage, j.message, j.progress = d.get("stage", ""), d.get("message", ""), d.get("progress", 1)
        j.outputs, j.stats, j.error, j.created, j.log = d.get("outputs", {}), d.get("stats", {}), d.get("error"), d.get("created", 0), d.get("log", [])
        JOBS[j.id] = j


_load_history()


# ------------------------------------------------------------------ API
@app.get("/api/ping")
def ping():
    return {"ok": True}


@app.get("/api/status")
def status():
    cfg = config.load()
    gpus = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        for row in out.strip().splitlines():
            i, name, used, total, util = [x.strip() for x in row.split(",")]
            gpus.append({"index": int(i), "name": name.replace("NVIDIA GeForce ", ""), "used_mb": int(used),
                         "total_mb": int(total), "free_mb": int(total) - int(used), "util": int(util)})
    except Exception:
        pass
    hi3d_ok = (Path(cfg["hi3d_model"]) / "transformer_1024" / "model.safetensors").exists()
    return {"gpu_index": cfg["gpu_index"], "gpus": gpus,
            "trellis_ready": bool(cfg.get("trellis_server")) and Path(cfg.get("trellis_server", "/nonexistent")).exists(),
            "hi3d_ready": hi3d_ok, "busy": any(j.status == "running" for j in JOBS.values()),
            "queued": sum(j.status == "queued" for j in JOBS.values())}


@app.post("/api/jobs")
async def create_job(params: str = Form(...), images: list[UploadFile] = File(default=[]),
                     coarse_mesh: UploadFile | None = File(default=None),
                     fit_mesh: UploadFile | None = File(default=None)):
    """Queue a job. ``params`` is JSON (see docs/API.md); ``fit_mesh`` is an optional
    proxy-part mesh the result is registered onto (proxy world space)."""
    p = json.loads(params)
    if not images:
        raise HTTPException(400, "At least one reference image is required (Hi3D conditions on it).")
    if len(images) > 8:
        raise HTTPException(400, "At most 8 views.")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = config.OUTPUTS / jid / "inputs"
    d.mkdir(parents=True)
    paths = []
    for i, up in enumerate(images):
        ext = Path(up.filename or "img.png").suffix.lower() or ".png"
        dest = d / f"view{i}{ext}"
        dest.write_bytes(await up.read())
        paths.append(str(dest))
    cpath = None
    if coarse_mesh is not None and coarse_mesh.filename:
        cpath = d / ("coarse_upload" + (Path(coarse_mesh.filename).suffix or ".glb"))
        cpath.write_bytes(await coarse_mesh.read())
        cpath = str(cpath)
    p.pop("fit_target", None)  # only via upload, never a client-supplied path
    if fit_mesh is not None and fit_mesh.filename:
        fpath = d / ("fit_target" + (Path(fit_mesh.filename).suffix or ".glb"))
        fpath.write_bytes(await fit_mesh.read())
        p["fit_target"] = str(fpath)
    p["front"] = max(0, min(int(p.get("front", 0)), len(paths) - 1))
    job = Job(jid, p, paths, cpath)
    JOBS[jid] = job
    job.save()
    QUEUE.put(jid)
    return {"id": jid}


@app.get("/api/jobs")
def list_jobs():
    return sorted((j.public(full=False) for j in JOBS.values()), key=lambda d: -d["created"])


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    return JOBS[jid].public()


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404)
    job.cancel = True
    if job.proc and job.proc.poll() is None:
        job.proc.kill()
    if job.server:
        job.server.kill()
    if job.status == "queued":
        job.status, job.message = "cancelled", "Cancelled"
        job.save()
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = JOBS.get(jid)
    if not job or job.status in ("running", "queued"):
        raise HTTPException(400, "Cannot delete a running job")
    shutil.rmtree(job.dir, ignore_errors=True)
    JOBS.pop(jid, None)
    return {"ok": True}


@app.get("/api/jobs/{jid}/files/{name:path}")
def job_file(jid: str, name: str):
    base = (config.OUTPUTS / jid).resolve()
    target = (base / name).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


REFINE_KEYS = ("hi3d_res", "steps", "fusion", "front_weight", "remove_bg", "seed", "colour", "hybrid",
               "texture", "tex_faces", "atlas", "offload", "torch_conv", "gpu")
HI3D_OUTPUTS = ("refined", "refined_aligned", "refined_hybrid", "refined_coloured", "textured")


@app.post("/api/jobs/{jid}/refine")
def refine_job(jid: str, body: dict):
    """Run (or re-run) the Hi3D stage on a finished job's existing TRELLIS mesh.

    Lets you generate coarse-only, look at the TRELLIS result, and only spend the
    Hi3D time on the ones worth refining. ``body`` takes the Hi3D settings from
    the create-job params (hi3d_res, steps, fusion, hybrid, texture, ...)."""
    job = JOBS.get(jid)
    if not job or job.status in ("running", "queued"):
        raise HTTPException(400, "Job is missing or still running")
    if not (job.dir / "coarse.glb").is_file():
        raise HTTPException(400, "This job has no TRELLIS mesh (coarse.glb) to refine")
    if not job.images:
        raise HTTPException(400, "This job has no reference image; Hi3D conditions on it")
    job.params.update({k: body[k] for k in REFINE_KEYS if k in body})
    job.params["refine"] = True
    for k in HI3D_OUTPUTS:
        job.outputs.pop(k, None)
    job.skip_coarse = True
    job.cancel = False
    job.error = None
    job.status, job.stage, job.message, job.progress = "queued", "queued", "Waiting for GPU", 0.0
    job.add_log(f"Hi3D refine requested ({job.params.get('hi3d_res', 1024)}³)")
    job.save()
    QUEUE.put(jid)
    return {"id": jid}


@app.post("/api/jobs/{jid}/post")
def post_process(jid: str, body: dict):
    """CPU post-processing of a finished job: hybrid, decimate, texture (scripts/postprocess.py)."""
    job = JOBS.get(jid)
    if not job or job.status in ("running", "queued"):
        raise HTTPException(400, "Job is missing or still running")
    args = [sys.executable, str(config.ROOT / "scripts" / "postprocess.py"), str(job.dir)]
    if not body.get("hybrid", True):
        args.append("--no-hybrid")
    if int(body.get("decimate", 0)) > 0:
        args += ["--decimate", str(int(body["decimate"]))]
    if body.get("texture"):
        args += ["--texture", "--tex-faces", str(int(body.get("tex_faces", 300_000))),
                 "--atlas", str(int(body.get("atlas", 2048)))]
    prev_status = job.status

    def run():
        job.status, job.stage, job.message, job.progress = "running", "post", "Post-processing", 0.5
        job.add_log("$ " + " ".join(args[1:]))
        try:
            proc = subprocess.Popen(args, cwd=config.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env={**os.environ, **config.gpu_env(config.load())})
            job.proc = proc
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line.startswith("@@"):
                    if line:
                        job.add_log(line)
                    continue
                ev = json.loads(line[2:])
                if ev["type"] in ("stage", "log"):
                    job.add_log(("▸ " if ev["type"] == "stage" else "") + ev["message"])
                    if ev["type"] == "stage":
                        job.message = ev["message"]
                elif ev["type"] == "result":
                    job.outputs.update(ev["outputs"])
                    job.stats.update(ev["stats"])
                elif ev["type"] == "error":
                    job.add_log("ERROR: " + ev["message"])
            proc.wait()
            job.status = prev_status
            job.message = "Post-processing done" if proc.returncode == 0 else "Post-processing failed — see log"
        finally:
            job.proc = None
            job.progress = 1.0
            job.save()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown():
    for j in JOBS.values():
        if j.proc and j.proc.poll() is None:
            j.proc.kill()
        if j.server:
            j.server.kill()
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
