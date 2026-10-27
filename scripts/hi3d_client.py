#!/usr/bin/env python3
"""Minimal Hi3D Studio client for scripts and agents (stdlib only).

Examples:
  # whole-body proxy from front + side views
  hi3d_client.py front.png side.png --coarse-mode multidiffusion --hi3d-res 512 --out proxy/

  # one part, fitted onto its region of the proxy
  hi3d_client.py head_crop.png --fit proxy_head.glb --hi3d-res 1536 --out parts/head/

Prints the job id, streams progress to stderr, downloads every output to --out,
and prints a JSON summary (outputs + stats) on stdout. Exit code 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def _multipart(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, p in files:
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{p.name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + p.read_bytes() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path, help="reference views (first = front unless --front)")
    ap.add_argument("--server", default="http://127.0.0.1:7860")
    ap.add_argument("--out", type=Path, required=True, help="directory to download outputs into")
    ap.add_argument("--front", type=int, default=0)
    ap.add_argument("--coarse-mode", default=None, choices=["single", "multidiffusion", "stochastic", "upload"])
    ap.add_argument("--coarse-mesh", type=Path, help="own coarse mesh (implies --coarse-mode upload)")
    ap.add_argument("--coarse-res", type=int, default=1024, choices=[512, 1024, 1536])
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--hi3d-res", type=int, default=1024, choices=[512, 1024, 1536])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--fusion", default="front", choices=["front", "multidiffusion", "stochastic"])
    ap.add_argument("--front-weight", type=float, default=2.0)
    ap.add_argument("--fit", type=Path, help="proxy-part mesh (world space) to register the result onto")
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    a = ap.parse_args()

    mode = a.coarse_mode or ("upload" if a.coarse_mesh else ("multidiffusion" if len(a.images) > 1 else "single"))
    params = {"coarse_mode": mode, "coarse_res": a.coarse_res, "refine": not a.no_refine, "hi3d_res": a.hi3d_res,
              "steps": a.steps, "fusion": a.fusion, "front_weight": a.front_weight, "colour": not a.no_colour,
              "offload": a.offload, "seed": a.seed, "front": a.front}
    if a.gpu is not None:
        params["gpu"] = a.gpu
    files = [("images", p) for p in a.images]
    if a.coarse_mesh:
        files.append(("coarse_mesh", a.coarse_mesh))
    if a.fit:
        files.append(("fit_mesh", a.fit))
    body, ctype = _multipart({"params": json.dumps(params)}, files)
    req = urllib.request.Request(f"{a.server}/api/jobs", data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=120) as r:
        jid = json.loads(r.read())["id"]
    print(f"job {jid}", file=sys.stderr)

    last = ""
    while True:
        j = _get(f"{a.server}/api/jobs/{jid}")
        line = f"[{j['progress'] * 100:5.1f}%] {j['message']}"
        if line != last:
            print(line, file=sys.stderr)
            last = line
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(2)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, f in j["outputs"].items():
        urllib.request.urlretrieve(f"{a.server}/api/jobs/{jid}/files/{f}", a.out / f)
    summary = {"id": jid, "status": j["status"], "error": j["error"],
               "outputs": {k: str(a.out / f) for k, f in j["outputs"].items()}, "stats": j["stats"]}
    print(json.dumps(summary, indent=2))
    return 0 if j["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
