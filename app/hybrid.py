"""Hybrid merge: Hi3D detail where it holds, TRELLIS geometry where it doesn't.

Hi3D rebuilds surfaces from a sparse SDF. Thin sheets (cloth, feathers, fins,
straps) are often thinner than a voxel, so they come back perforated or as loose
fragments. TRELLIS's coarse mesh still has them intact. This module:

1. keeps Hi3D's main body (connected components holding >= MAIN_RATIO of faces)
   and drops the loose fragments;
2. marks coarse faces that are farther than ``tau`` from that body - where Hi3D
   lost coverage - removes speckle, and grows the patch a few rings so it
   overlaps the body instead of leaving a gap;
3. carves away refined faces that sit on the patch but aren't explained by any
   other coarse surface (the ghosted remains of the same sheet);
4. returns body + patch as one mesh, plus a per-face mask of which faces came
   from TRELLIS.

Both inputs must already be in the same frame (refined registered onto coarse).
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MAIN_RATIO = 0.05


def _main_body(refined: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    labels = trimesh.graph.connected_component_labels(refined.face_adjacency, len(refined.faces))
    counts = np.bincount(labels)
    keep = np.isin(labels, np.nonzero(counts >= MAIN_RATIO * len(refined.faces))[0])
    body = refined.submesh([np.nonzero(keep)[0]], append=True)
    return body, int((~keep).sum())


def _face_graph(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    adj = mesh.face_adjacency
    n = len(mesh.faces)
    g = sparse.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                          shape=(n, n))
    return g.tocsr()


def hybrid_merge(refined: trimesh.Trimesh, coarse: trimesh.Trimesh, tau: float = 0.006,
                 min_patch_faces: int = 40, grow_rings: int = 2) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    ext = float(np.linalg.norm(coarse.extents))
    body, dropped = _main_body(refined)

    coarse = trimesh.Trimesh(coarse.vertices, coarse.faces, process=True)  # welded for adjacency
    body_pts, _ = trimesh.sample.sample_surface(body, min(3_000_000, len(body.faces) * 2))
    body_tree = cKDTree(body_pts)
    # A face is "lost" if its centroid AND all three corners are off the body.
    d_cent, _ = body_tree.query(coarse.triangles_center, workers=-1)
    d_vert, _ = body_tree.query(coarse.vertices, workers=-1)
    d_face = np.minimum(d_cent, d_vert[coarse.faces].min(1)) / ext
    lost = d_face > tau

    graph = _face_graph(coarse)
    # Remove speckle: tiny islands of "lost" faces are noise, not missing parts.
    sub = graph[lost][:, lost]
    n_isl, lab = connected_components(sub, directed=False)
    isl_size = np.bincount(lab, minlength=n_isl)
    lost_idx = np.nonzero(lost)[0]
    patch = np.zeros(len(coarse.faces), bool)
    patch[lost_idx[isl_size[lab] >= min_patch_faces]] = True
    core = patch.copy()
    for _ in range(grow_rings):  # overlap the body so the seam has no gap
        patch |= (graph @ patch.astype(np.float32)) > 0

    patch_mesh = coarse.submesh([np.nonzero(patch)[0]], append=True) if patch.any() else None
    carved = 0
    if patch_mesh is not None:
        # Refined faces lying on the patch but far from all *other* coarse surface are
        # the ghost of the same thin sheet - remove them so the patch replaces it.
        keep_coarse = coarse.submesh([np.nonzero(~core)[0]], append=True)
        pp, _ = trimesh.sample.sample_surface(patch_mesh, max(50_000, int(patch.sum()) * 4))
        kp, _ = trimesh.sample.sample_surface(keep_coarse, 1_000_000)
        bc = body.triangles_center
        d_patch, _ = cKDTree(pp).query(bc, workers=-1)
        d_keep, _ = cKDTree(kp).query(bc, workers=-1)
        ghost = (d_patch / ext < tau * 1.5) & (d_keep / ext > tau)
        carved = int(ghost.sum())
        if carved:
            body = body.submesh([np.nonzero(~ghost)[0]], append=True)
        merged = trimesh.util.concatenate([body, patch_mesh])
        from_coarse = np.r_[np.zeros(len(body.faces), bool), np.ones(len(patch_mesh.faces), bool)]
    else:
        merged, from_coarse = body, np.zeros(len(body.faces), bool)

    info = {"fragments_dropped_faces": dropped, "patched_coarse_faces": int(patch.sum()),
            "patched_area_pct": round(float(coarse.area_faces[patch].sum() / coarse.area * 100), 2),
            "ghost_faces_carved": carved, "tau_pct": tau * 100}
    return merged, from_coarse, info
