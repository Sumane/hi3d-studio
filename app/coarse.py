"""Coarse stage: TRELLIS.2 through the pixal3d.cpp fork of trellis.cpp.

The native server is started for one job on a private loopback port and stopped
afterwards, so its VRAM is free before Hi3D loads (same pattern as AISmith 3D).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import requests

from . import config

Log = Callable[[str], None]
Progress = Callable[[float, str], None]

_STEP_RE = re.compile(r"\[(\d+)/(\d+)\]")


def _as_png(path: Path) -> bytes:
    """The multi-image endpoint only accepts PNG; keep alpha if the source has it."""
    from io import BytesIO
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


class TrellisServer:
    """One-shot trellis-server process bound to 127.0.0.1."""

    def __init__(self, resolution: int, log: Log, progress: Progress | None = None, gpu: int | None = None,
                 dump_path: Path | None = None):
        self.cfg = config.load()
        self.dump_path = dump_path
        if gpu is not None:
            self.cfg["gpu_index"] = gpu
        self.resolution = resolution
        self.log = log
        self.progress = progress
        self.proc: subprocess.Popen[str] | None = None
        self.port = 0
        self.lines: list[str] = []

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "TrellisServer":
        binary = self.cfg.get("trellis_server")
        models = self.cfg.get("trellis_models")
        if not binary or not Path(binary).exists():
            raise RuntimeError("trellis-server is not installed — run install.sh")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        base = [binary, "--models", models, "--res", str(self.resolution), "--require-gpu", "--port", str(self.port)]
        for cmd in (base + ["--host", "127.0.0.1"], base):
            self._spawn(cmd)
            if self._wait_ready():
                return self
            self.stop()
            if not any("host" in l.lower() for l in self.lines[-20:]):
                break
        raise RuntimeError("trellis-server failed to start:\n" + "\n".join(self.lines[-30:]))

    def __exit__(self, *exc) -> None:
        self.stop()

    def _spawn(self, cmd: list[str]) -> None:
        self.lines = []
        env = {**os.environ, **config.gpu_env(self.cfg)}
        if self.dump_path:
            # Makes the engine also write its sparse PBR voxel field (see texture.py).
            env["TRELLIS_DUMP_POST"] = str(self.dump_path)
        self.log("$ " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        threading.Thread(target=self._relay, daemon=True).start()

    def _relay(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            self.lines.append(line)
            self.log(line)
            m = _STEP_RE.search(line)
            if m and self.progress:
                a, b = int(m.group(1)), int(m.group(2))
                if b:
                    self.progress(min(a / b, 1.0), line)

    def _wait_ready(self, timeout: float = 120) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                return False
            try:
                if requests.get(self.url("/health"), timeout=1).ok:
                    return True
            except requests.RequestException:
                time.sleep(0.3)
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def capabilities(self) -> dict:
        try:
            r = requests.get(self.url("/capabilities"), timeout=5)
            return r.json() if r.ok else {}
        except (requests.RequestException, ValueError):
            return {}

    # -- generation ------------------------------------------------------
    def _common(self, seed: int, bg_removal: str, faces: str) -> dict:
        data = {"seed": str(seed), "resolution": str(self.resolution)}
        if faces and faces != "auto":
            data["faces"] = faces
        if bg_removal in ("threshold", "birefnet"):
            data["bg_removal"] = bg_removal
        return data

    def generate(self, image: Path, out: Path, seed: int, bg_removal: str = "auto", faces: str = "auto") -> None:
        with open(image, "rb") as fh:
            r = requests.post(self.url("/generate"), files={"image": (image.name, fh, "image/png")},
                              data=self._common(seed, bg_removal, faces), timeout=7200)
        self._save(r, out)

    def generate_mv(self, images: list[Path], out: Path, seed: int, mode: str,
                    bg_removal: str = "auto", faces: str = "auto") -> None:
        """POST /generate-trellis2-mv (pixal3d.cpp #66 contract).

        Fields: num_images (2..8), fusion, image0..imageN-1 (PNG, contiguous, unique),
        plus optional seed/resolution/bg_removal/uv/band/webp. Anything else -> 400,
        so no ``faces`` here (the engine's default simplification applies).
        """
        if not 2 <= len(images) <= 8:
            raise RuntimeError("Multi-view needs 2-8 images")
        data = {"num_images": str(len(images)), "fusion": mode, "seed": str(seed),
                "resolution": str(self.resolution)}
        if bg_removal in ("auto", "threshold", "birefnet"):
            data["bg_removal"] = bg_removal
        files = [(f"image{i}", (f"view{i}.png", _as_png(p), "image/png")) for i, p in enumerate(images)]
        r = requests.post(self.url("/generate-trellis2-mv"), files=files, data=data, timeout=7200)
        if r.status_code == 404:
            raise RuntimeError("This engine build has no /generate-trellis2-mv endpoint — re-run install.sh.")
        self._save(r, out)

    @staticmethod
    def _save(r: requests.Response, out: Path) -> None:
        if not r.ok:
            raise RuntimeError(f"trellis-server HTTP {r.status_code}: {r.text[:600]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        if out.stat().st_size < 64:
            raise RuntimeError("trellis-server returned an empty GLB")
