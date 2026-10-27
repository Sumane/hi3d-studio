"""TRELLIS.2 UV + PBR texture for the refined (Hi3D / hybrid) mesh.

TRELLIS.2 does not paint a 2D image: it generates a sparse 3D PBR field (base
colour, metallic, roughness, alpha per voxel at res 1024), then unwraps its mesh
(tuned xatlas) and bakes that field into an atlas. trellis-server writes the
field when TRELLIS_DUMP_POST is set (coarse.py). Here we hand TRELLIS's own
post-processing tool - ``post-replay`` from the same engine build - a dump that
contains *our* mesh plus TRELLIS's field, so TRELLIS generates the UVs and the
texture for the refined geometry directly.

One adjustment: the field only exists in a thin shell (~1-3 voxels) around
TRELLIS's surface, and Hi3D's sharper surface leaves that shell in places. Before
the bake, every voxel the new surface passes through that TRELLIS left empty is
filled by inverse-distance interpolation from the nearest TRELLIS voxels, so the
engine's trilinear lookup never misses (no smeared seam-fill on detail).

Dump layout (pixal3d.cpp): i32 V,F,Mv,res; f32 verts[V*3]; i32 faces[F*3];
i32 coords[Mv*3]; f32 pbr6[Mv*6] (rgb, metallic, roughness, alpha). Voxel c is
centred at (c + 0.5)/res - 0.5 in TRELLIS's internal frame; the GLB writer maps
internal (x, y, z) -> glTF (x, z, -y).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import config


def load_dump(path: Path) -> dict:
    with open(path, "rb") as f:
        V, F, Mv, res = (int(x) for x in np.frombuffer(f.read(16), dtype=np.int32))
        f.seek(16 + V * 12 + F * 12)  # we only need the field, not TRELLIS's raw mesh
        coords = np.frombuffer(f.read(Mv * 12), dtype=np.int32).reshape(Mv, 3)
        pbr = np.frombuffer(f.read(Mv * 24), dtype=np.float32).reshape(Mv, 6)
    if len(pbr) != Mv:
        raise ValueError(f"{path.name}: truncated PBR dump")
    return {"coords": coords, "pbr": pbr, "res": res}


def glb_to_internal(p: np.ndarray) -> np.ndarray:
    """Inverse of the GLB writer's (x, y, z) -> (x, z, -y)."""
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


def _keys(c: np.ndarray, res: int) -> np.ndarray:
    c = c.astype(np.int64)
    return (c[:, 0] * res + c[:, 1]) * res + c[:, 2]


def extend_field(mesh_int: trimesh.Trimesh, dump: dict, log=print) -> tuple[np.ndarray, np.ndarray, int]:
    """Add interpolated voxels wherever ``mesh_int``'s surface leaves TRELLIS's shell."""
    res = dump["res"]
    n = int(np.clip(mesh_int.area * res * res, 1_000_000, 16_000_000))
    pts, _ = trimesh.sample.sample_surface(mesh_int, n)
    offs = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int32)
    need = np.zeros(0, np.int64)
    for s in range(0, len(pts), 1_000_000):   # chunked: 8 trilinear corners per sample
        base = np.floor((pts[s:s + 1_000_000] + 0.5) * res - 0.5).astype(np.int32)
        corners = (base[:, None, :] + offs[None]).reshape(-1, 3)
        corners = corners[(corners >= 0).all(1) & (corners < res).all(1)]
        need = np.union1d(need, _keys(corners, res))
    del pts
    have = np.sort(_keys(dump["coords"], res))
    pos = np.clip(np.searchsorted(have, need), 0, len(have) - 1)
    missing = need[have[pos] != need]
    if not len(missing):
        return dump["coords"], dump["pbr"], 0
    mc = np.stack([missing // (res * res), (missing // res) % res, missing % res], 1).astype(np.int32)
    log(f"Extending TRELLIS PBR field by {len(mc):,} voxels to cover the refined surface")
    dist, idx = cKDTree(dump["coords"].astype(np.float32)).query(mc.astype(np.float32), k=6, workers=-1)
    w = 1.0 / np.maximum(dist, 0.5) ** 2
    w /= w.sum(1, keepdims=True)
    vals = np.einsum("ij,ijk->ik", w, dump["pbr"][idx]).astype(np.float32)
    return np.concatenate([dump["coords"], mc]), np.concatenate([dump["pbr"], vals]), int(len(mc))


def write_dump(path: Path, verts_int: np.ndarray, faces: np.ndarray, coords: np.ndarray, pbr: np.ndarray,
               res: int) -> None:
    with open(path, "wb") as f:
        np.array([len(verts_int), len(faces), len(coords), res], np.int32).tofile(f)
        np.ascontiguousarray(verts_int, np.float32).tofile(f)
        np.ascontiguousarray(faces, np.int32).tofile(f)
        np.ascontiguousarray(coords, np.int32).tofile(f)
        np.ascontiguousarray(pbr, np.float32).tofile(f)


def set_root_transform(src: Path, dst: Path, matrix: np.ndarray) -> None:
    """Copy a GLB, parenting its scene under one node carrying ``matrix`` (column-major)."""
    import json
    import struct
    data = Path(src).read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + jlen].decode("utf-8"))
    rest = data[20 + jlen:]
    scene = gltf["scenes"][gltf.get("scene", 0)]
    gltf.setdefault("nodes", []).append({"name": "placement", "children": list(scene.get("nodes", [])),
                                         "matrix": [float(x) for x in np.asarray(matrix).T.reshape(-1)]})
    scene["nodes"] = [len(gltf["nodes"]) - 1]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + len(rest)
    Path(dst).write_bytes(struct.pack("<III", magic, version, total) + struct.pack("<II", len(js), jtype) + js + rest)


def post_replay_bin() -> Path:
    server = Path(config.load().get("trellis_server", ""))
    cand = server.parent / "post-replay"
    if not cand.exists():
        raise FileNotFoundError(
            "post-replay is not built. Run:  cmake --build engines/pixal3d.cpp/build --target post-replay")
    return cand


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32))
    s.simplify_mesh(target_count=int(target_faces), aggressiveness=7, preserve_border=True, verbose=False)
    v, f, _ = s.getMesh()
    return trimesh.Trimesh(v, f, process=True)


def bake_job_mesh(mesh_in_coarse: trimesh.Trimesh, to_place: np.ndarray, dump_path: Path,
                  faces: int = 300_000, atlas: int = 2048, log=print, out_path: Path | None = None):
    """Decimate (optional) -> TRELLIS UV + PBR bake via post-replay -> move to the output frame.

    ``mesh_in_coarse`` must be in TRELLIS's GLB frame (i.e. registered onto coarse.glb).
    Writes ``out_path`` and returns info.
    """
    dump = load_dump(Path(dump_path))
    low = decimate(mesh_in_coarse, faces)
    if len(low.faces) != len(mesh_in_coarse.faces):
        log(f"Decimated {len(mesh_in_coarse.faces):,} -> {len(low.faces):,} faces for UV/texture")
    v_int = glb_to_internal(np.asarray(low.vertices))
    mesh_int = trimesh.Trimesh(v_int, low.faces, process=False)
    coords, pbr, added = extend_field(mesh_int, dump, log=log)

    tmp_root = config.ROOT / ".cache" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td = Path(td)
        write_dump(td / "bake.bin", v_int, np.asarray(low.faces), coords, pbr, dump["res"])
        cmd = [str(post_replay_bin()), str(td / "bake.bin"), str(td / "baked.glb"),
               "--no-remesh", "--decim", "0", "--atlas", str(int(atlas))]
        log(f"TRELLIS UV unwrap + PBR bake (post-replay, atlas {atlas})")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip() and ("[" in line or "wrote" in line or "fail" in line.lower()):
                log("  " + line.strip())
        if proc.returncode != 0 or not (td / "baked.glb").exists():
            raise RuntimeError(f"post-replay failed (exit {proc.returncode}): {proc.stderr[-400:]}")
        bake_s = re.search(r"\[bake ([\d.]+)s\]", proc.stdout)
        if np.allclose(to_place, np.eye(4), atol=1e-9):
            shutil.copy(td / "baked.glb", out_path)   # byte-identical engine output
        else:
            # Proxy placement: wrap the scene in a transform node instead of re-encoding,
            # so the engine's WebP/PNG textures pass through untouched.
            set_root_transform(td / "baked.glb", out_path, to_place)
    return {"faces": int(len(low.faces)), "atlas": int(atlas), "voxels_trellis": int(len(dump["coords"])),
            "voxels_extended": added, "bake_seconds": float(bake_s.group(1)) if bake_s else None}
