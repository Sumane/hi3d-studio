"""Replacement for Hi3D's final clean-up + simplification step.

What Hi3D actually outputs (measured on a 1024 face job, and identical with the
stock pipeline): the raw FlexiCubes mesh is a *thin shell about one voxel thick*
- an outer and an inner sheet - and the shell keeps pinching shut and reopening
across the surface. Each pinch is a tiny closed "lens". Rendered, that is the
topographic rings and speckle; Hi3D's pyfqmr simplification then tears it into
~58,000 pieces with ~77,000 non-manifold edges.

Fix (solidify):
  1. rasterise the raw surface onto a grid, close the pinch holes (dilate),
     flood-fill the outside; everything not outside is the solid,
  2. erode back and extract one watertight surface (marching cubes),
  3. snap its vertices onto Hi3D's raw surface so the detail is Hi3D's, not the grid's,
  4. light Taubin smoothing, MeshLab decimation to the face target (topology kept),
  5. centre + unit-radius normalise exactly like Hi3D.

Side effect: gaps narrower than ~6 Hi3D voxels (e.g. between an earring and the
ear) are closed, the same as any solidify/remesh.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import trimesh

COMPONENT_FACE_RATIO = 0.005


def mesh_health(mesh: trimesh.Trimesh) -> dict:
    """Piece count and non-manifold edge count: the two numbers that show tearing."""
    m = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    comps = trimesh.graph.connected_components(m.face_adjacency, min_len=1)
    _, counts = np.unique(m.edges_sorted, axis=0, return_counts=True)
    largest = max((len(c) for c in comps), default=0)
    return {"faces": int(len(m.faces)), "pieces": int(len(comps)),
            "largest_piece_pct": round(100 * largest / max(len(m.faces), 1), 1),
            "nonmanifold_edges": int((counts > 2).sum()), "open_edges": int((counts == 1).sum())}


def _decimate_meshlab(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, np.float64),
                               face_matrix=np.asarray(mesh.faces, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target), qualitythr=0.3, preserveboundary=True, preservenormal=True,
        preservetopology=True, optimalplacement=True, planarquadric=True, autoclean=True)
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)


def solidify(raw: trimesh.Trimesh, grid: int = 640, log=print) -> trimesh.Trimesh:
    """One watertight outer surface from Hi3D's thin pinching shell, snapped back onto it."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure

    t0 = time.time()
    v = np.asarray(raw.vertices, np.float64)
    F = np.asarray(raw.faces)
    ext = float(np.ptp(v, 0).max())
    vox = ext / 1024
    close = max(2, round(3 * grid / 512))          # ~6 Hi3D voxels whatever the grid
    pad = close + 3
    h = ext / (grid - 2 * pad)
    lo = v.min(0) - pad * h

    occ = np.zeros((grid, grid, grid), bool)
    for p in (v, (v[F[:, 0]] + v[F[:, 1]] + v[F[:, 2]]) / 3,
              (v[F[:, 0]] + v[F[:, 1]]) / 2, (v[F[:, 1]] + v[F[:, 2]]) / 2, (v[F[:, 2]] + v[F[:, 0]]) / 2):
        c = np.clip(((p - lo) / h).astype(np.int32), 0, grid - 1)
        occ[c[:, 0], c[:, 1], c[:, 2]] = True
        del c
    occ = ndimage.binary_dilation(occ, iterations=close)
    lab, _ = ndimage.label(~occ)
    del occ
    solid = lab != lab[0, 0, 0]
    del lab
    gc.collect()
    solid = ndimage.binary_erosion(solid, iterations=close)
    field = ndimage.gaussian_filter(solid.astype(np.float32), 0.8)
    del solid
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    del field
    verts = verts * h + lo
    shell = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    log(f"Solidified Hi3D shell on a {grid}³ grid: {len(shell.faces):,} faces ({time.time() - t0:.0f}s)")

    # Snap onto Hi3D's own surface (nearest sheet = the outer one).
    pts = np.concatenate([v, (v[F[:, 0]] + v[F[:, 1]] + v[F[:, 2]]) / 3]).astype(np.float32)
    d, i = cKDTree(pts).query(np.asarray(shell.vertices), k=8, workers=-1)
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    w /= w.sum(1, keepdims=True)
    target = np.einsum("ij,ijk->ik", w, pts[i].astype(np.float64))
    far = d[:, 0] > 3 * vox
    target[far] = shell.vertices[far]
    out = trimesh.Trimesh(target, shell.faces, process=False)
    trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=2)
    log(f"Snapped {100 * (1 - far.mean()):.1f}% of vertices onto Hi3D's surface ({time.time() - t0:.0f}s)")
    return out


def clean_simplify(mesh, target_face_count: int, grid: int = 640, log=print) -> trimesh.Trimesh:
    from sparc3d.sparcubes.postprocess import validate_mesh

    m = validate_mesh(mesh)
    m = solidify(m, grid=grid, log=log)
    if len(m.faces) > target_face_count:
        try:
            m = _decimate_meshlab(m, target_face_count)
        except ImportError:
            log("pymeshlab not installed - keeping the solidified mesh at full resolution")
    # Hi3D's debris rule: drop components under 0.5% of faces.
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)
    largest = max(comps, key=len)
    keep = [c for c in comps if c is largest or len(c) / len(m.faces) >= COMPONENT_FACE_RATIO]
    m = trimesh.Trimesh(m.vertices, m.faces[np.sort(np.concatenate(keep))], process=False)
    m.remove_unreferenced_vertices()
    v = np.asarray(m.vertices, np.float64)
    v = v - v.mean(axis=0, keepdims=True)
    v = v / float(np.linalg.norm(v, axis=1).max())
    return trimesh.Trimesh(v, np.asarray(m.faces, np.int64), process=False)
