"""Placement and colour for Hi3D's output.

Hi3D recentres and rescales its mesh to unit radius, which is useless for
reassembling separately generated parts. This module puts the refined mesh back
where it belongs:

* **coarse frame** (default): registered onto the TRELLIS coarse mesh.
* **proxy frame** (``fit_path``): registered onto a region cut from a whole-body
  proxy, so a part generated on its own lands in the proxy's world space.

Registration is a similarity transform (rotation, uniform scale, translation):
bounding-box initialisation, then ICP, with a few starting rotations about the
up axis so a part whose generator faced it differently still locks on.

Colours come from a dense sampling of the coarse *texture* (not its vertices), so
detail finer than the coarse tessellation survives, written as linear COLOR_0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SAMPLES = 3_000_000
FIT_SAMPLES = 150_000


def _geometries(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    out = []
    for node in loaded.graph.nodes_geometry:
        transform, name = loaded.graph[node]
        g = loaded.geometry[name]
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            g = g.copy()
            g.apply_transform(transform)
            out.append(g)
    return out


def _texture_sampler(g: trimesh.Trimesh):
    """Return f(face_idx, barycentric) -> RGBA uint8 (sRGB) for one geometry."""
    vis = g.visual
    if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
        mat = vis.material
        image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        factor = np.asarray(getattr(mat, "baseColorFactor", None) if getattr(mat, "baseColorFactor", None) is not None
                            else [255, 255, 255, 255], dtype=np.float64)
        if factor.max() <= 1.0:
            factor = factor * 255.0
        uv = np.asarray(vis.uv, dtype=np.float64)
        if image is not None:
            def sample(fi, bary):
                tri_uv = uv[g.faces[fi]]
                p = np.einsum("ij,ijk->ik", bary, tri_uv)
                col = trimesh.visual.color.uv_to_color(p, image).astype(np.float64)
                return np.clip(col * factor / 255.0, 0, 255).astype(np.uint8)
            return sample
        flat = np.clip(factor, 0, 255).astype(np.uint8)
        return lambda fi, bary: np.repeat(flat[None], len(fi), 0)
    vc = np.asarray(vis.vertex_colors if hasattr(vis, "vertex_colors") else
                    np.full((len(g.vertices), 4), 180), dtype=np.float64)

    def sample(fi, bary):
        return np.einsum("ij,ijk->ik", bary, vc[g.faces[fi]]).astype(np.uint8)
    return sample


def sample_surface(path: Path, n: int, colours: bool = False, normals: bool = False):
    """Area-weighted surface samples (and optionally texture colours / face normals) of a mesh file.

    Returns (points, colours|None) or, with ``normals``, (points, colours|None, normals).
    """
    geoms = _geometries(path)
    if not geoms:
        raise ValueError(f"{path.name} has no triangle geometry")
    areas = np.array([g.area for g in geoms])
    counts = np.maximum((areas / areas.sum() * n).astype(int), 500)
    pts, cols, nrm = [], [], []
    for g, k in zip(geoms, counts):
        p, fi = trimesh.sample.sample_surface(g, int(k))
        pts.append(p)
        if normals:
            nrm.append(g.face_normals[fi])
        if colours:
            bary = trimesh.triangles.points_to_barycentric(g.triangles[fi], p)
            c = _texture_sampler(g)(fi, bary)
            if c.shape[1] == 3:
                c = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], 1)
            cols.append(c)
    out = (np.concatenate(pts), (np.concatenate(cols) if colours else None))
    return out + (np.concatenate(nrm),) if normals else out


def _bbox_similarity(src: np.ndarray, dst: np.ndarray, rot: np.ndarray) -> np.ndarray:
    s = src @ rot[:3, :3].T
    s_lo, s_hi = s.min(0), s.max(0)
    d_lo, d_hi = dst.min(0), dst.max(0)
    scale = np.linalg.norm(d_hi - d_lo) / max(np.linalg.norm(s_hi - s_lo), 1e-9)
    m = np.eye(4)
    m[:3, :3] = scale * rot[:3, :3]
    m[:3, 3] = (d_lo + d_hi) / 2 - scale * (s_lo + s_hi) / 2
    return m


def fit_similarity(vertices: np.ndarray, ref_pts: np.ndarray, try_rotations: bool = True) -> tuple[np.ndarray, dict]:
    """Similarity transform taking ``vertices`` onto the surface sampled by ``ref_pts``."""
    rng = np.random.default_rng(0)
    v = np.asarray(vertices, dtype=np.float64)
    a = v[rng.choice(len(v), min(len(v), 40_000), replace=False)]
    b = ref_pts[rng.choice(len(ref_pts), min(len(ref_pts), 120_000), replace=False)]
    angles = (0, 90, 180, 270) if try_rotations else (0,)
    results = []
    for deg in angles:
        rot = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])
        init = _bbox_similarity(a, b, rot)
        try:
            m, _, cost = trimesh.registration.icp(a, b, initial=init, threshold=1e-7, max_iterations=60,
                                                  reflection=False, scale=True)
        except Exception:
            m, cost = init, np.inf
        results.append((m, cost, deg))
    # Parts generated from the same front view share the proxy's facing, and a
    # near-symmetric part (torso, leg) fits equally well flipped — which would
    # mirror its texture. Keep the unrotated fit unless another is clearly better.
    matrix, cost, deg = min(results, key=lambda r: r[1])
    if deg != 0 and np.isfinite(results[0][1]) and cost > 0.8 * results[0][1]:
        matrix, cost, deg = results[0]
    if matrix is None:
        matrix = _bbox_similarity(a, b, np.eye(4))
    extent = float(np.linalg.norm(ref_pts.max(0) - ref_pts.min(0)))
    return matrix, {"fit_mean_dist_pct": round(float(cost) / extent * 100, 3) if np.isfinite(cost) else None,
                    "start_rotation_deg": deg, "scale": round(float(np.cbrt(abs(np.linalg.det(matrix[:3, :3])))), 5)}


def _linear_colours(pts: np.ndarray, cols: np.ndarray, query: np.ndarray,
                    pnormals: np.ndarray | None = None, qnormals: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Inverse-distance colour lookup that only trusts samples facing the same way.

    TRELLIS meshes are double-walled (an inner wall ~0.3% inside the outer skin,
    textured dark) and can contain interior cavity surfaces. A plain nearest-sample
    lookup pulls those dark interior colours wherever the refined surface sits
    slightly inside TRELLIS's, so samples whose normal opposes the query normal
    are excluded.
    """
    k = 16 if pnormals is not None else 4
    dist, idx = cKDTree(pts).query(query, k=k, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-9) ** 2
    if pnormals is not None and qnormals is not None:
        agree = np.einsum("ijk,ik->ij", pnormals[idx], qnormals) > 0.2
        has = agree.any(1)
        w = np.where(has[:, None], w * agree, w)   # fall back to plain IDW where nothing agrees
    w /= w.sum(1, keepdims=True)
    vcol = np.einsum("ij,ijk->ik", w, cols[idx].astype(np.float64)) / 255.0
    # glTF COLOR_0 is linear; texture samples are sRGB.
    rgb = vcol[:, :3]
    vcol[:, :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.clip(np.round(vcol * 255), 0, 255).astype(np.uint8), float(np.median(dist[:, 0]))


def place_and_colour(refined: trimesh.Trimesh, coarse_path: Path, fit_path: Path | None = None,
                     colour: bool = True, hybrid: bool = False, return_internal: bool = False):
    """Place the refined mesh (coarse or proxy frame), optionally hybrid-merge it with
    the coarse mesh's thin parts, and optionally vertex-colour the final mesh.

    Returns {"refined_aligned", ["refined_hybrid"], ["refined_coloured"]} and stats.
    ``refined_coloured`` is built from the hybrid mesh when ``hybrid`` is on.
    """
    coarse_pts, coarse_cols, coarse_nrm = sample_surface(coarse_path, SAMPLES if colour else FIT_SAMPLES,
                                                         colours=colour, normals=True)
    m_coarse, info_c = fit_similarity(refined.vertices, coarse_pts, try_rotations=False)
    info = {"coarse": info_c}
    if fit_path is not None:
        target_pts, _ = sample_surface(fit_path, FIT_SAMPLES)
        m_place, info_t = fit_similarity(refined.vertices, target_pts, try_rotations=True)
        info["proxy"] = info_t
    else:
        m_place = m_coarse

    aligned = refined.copy()
    aligned.apply_transform(m_place)
    out = {"refined_aligned": aligned}

    final_in_coarse = refined.copy()
    final_in_coarse.apply_transform(m_coarse)
    final_placed = aligned
    if hybrid:
        from .hybrid import hybrid_merge
        coarse_mesh = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False)
                                                for g in _geometries(coarse_path)])
        merged, _, info["hybrid"] = hybrid_merge(final_in_coarse, coarse_mesh)
        final_in_coarse = merged
        final_placed = merged.copy()
        final_placed.apply_transform(m_place @ np.linalg.inv(m_coarse))
        out["refined_hybrid"] = final_placed

    if colour:
        vcol, med = _linear_colours(coarse_pts, coarse_cols, np.asarray(final_in_coarse.vertices),
                                    coarse_nrm, np.asarray(final_in_coarse.vertex_normals))
        coloured = final_placed.copy()
        coloured.visual = trimesh.visual.ColorVisuals(coloured, vertex_colors=vcol)
        out["refined_coloured"] = coloured
        extent = float(np.linalg.norm(coarse_pts.max(0) - coarse_pts.min(0)))
        info["median_colour_dist_pct"] = round(med / extent * 100, 3)
    if return_internal:
        # final mesh in the coarse (TRELLIS GLB) frame + the transform to the output frame
        return out, info, final_in_coarse, m_place @ np.linalg.inv(m_coarse)
    return out, info


def export_glb(mesh: trimesh.Trimesh, path: Path, roughness: float = 0.8) -> None:
    """Export, then give material-less primitives an explicit dielectric PBR material.

    trimesh writes vertex-coloured / plain meshes with no material, and the glTF
    default material is *fully metallic* (metallicFactor 1). Engines (three.js,
    Godot, Blender) then render it near-black wherever no light reflects straight
    back - dark patches that look like missing geometry.
    """
    import json
    import struct
    mesh.export(path)
    data = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    jlen, jtype = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + jlen].decode("utf-8"))
    rest = data[20 + jlen:]
    mats = gltf.setdefault("materials", [])
    idx = None
    for m in gltf.get("meshes", []):
        for prim in m.get("primitives", []):
            if "material" not in prim:
                if idx is None:
                    mats.append({"name": "hi3d_default", "doubleSided": False,
                                 "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                          "metallicFactor": 0.0, "roughnessFactor": roughness}})
                    idx = len(mats) - 1
                prim["material"] = idx
    if idx is None:
        return
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    Path(path).write_bytes(struct.pack("<III", magic, version, 12 + 8 + len(js) + len(rest))
                           + struct.pack("<II", len(js), jtype) + js + rest)


def place_scene(mesh_path: Path, fit_path: Path, out_path: Path) -> dict:
    """Align a (textured) mesh file onto ``fit_path`` and save it, keeping its materials."""
    scene = trimesh.load(mesh_path, process=False)
    pts, _ = sample_surface(mesh_path, FIT_SAMPLES)
    target, _ = sample_surface(fit_path, FIT_SAMPLES)
    m, info = fit_similarity(pts, target, try_rotations=True)
    scene.apply_transform(m)
    scene.export(out_path)
    return info


# Backwards-compatible name used by earlier worker versions.
def transfer_colours(coarse_path: Path, refined: trimesh.Trimesh):
    meshes, info = place_and_colour(refined, coarse_path, None, colour=True)
    return meshes["refined_coloured"], info
