"""S4: depth-residual scoring of a pose against the observed depth image.

Two complementary terms:
- model -> image: visible model points (z-buffered, back-faces culled if the mesh is watertight) are
  projected into the depth image; inlier if observed depth is within tau of the model depth.
  Penalises occlusion, free-space violations and missing depth.
- scene -> model: of the scene points within NEAR_MM of the model surface, the fraction within tau.
  Catches in-plane slides that a pure depth comparison cannot see (object edge strips stay unexplained).
Score = (inliers / visible) * explained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import project
from ..onboarding import Model

NEAR_MM = 10.0
CELL = 3  # z-buffer cell size in pixels (model points are sparse relative to pixels)


@dataclass
class Score:
    score: float
    inlier_ratio: float     # inliers / points with valid depth that are not occluded
    explained: float        # scene -> model term (1.0 if no scene points were given)
    residual_mm: float      # mean |dz| of inliers
    n_visible: int


def score_pose(T: np.ndarray, depth: np.ndarray, K: np.ndarray, model: Model, tau_mm: float,
               scene_pts: np.ndarray | None = None) -> Score:
    p = model.score_points @ T[:3, :3].T + T[:3, 3]
    if model.watertight:
        n = model.score_normals @ T[:3, :3].T
        front = np.einsum("ij,ij->i", n, p) < 0
        p = p[front]
    h, w = depth.shape
    u, v = project(p, K)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (p[:, 2] > 1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    p, ui, vi = p[inside], ui[inside], vi[inside]
    if len(p) == 0:
        return Score(0.0, 0.0, 0.0, 99.0, 0)

    cell = (vi // CELL) * ((w + CELL - 1) // CELL) + ui // CELL
    zmin = np.full(cell.max() + 1, np.inf)
    np.minimum.at(zmin, cell, p[:, 2])
    visible = p[:, 2] <= zmin[cell] + 2.0 * tau_mm
    p, ui, vi = p[visible], ui[visible], vi[visible]
    n_vis = len(p)
    if n_vis == 0:
        return Score(0.0, 0.0, 0.0, 99.0, 0)

    d = depth[vi, ui]
    valid = d > 0
    dz = d - p[:, 2]
    inl = valid & (np.abs(dz) < tau_mm)
    considered = valid & (dz > -tau_mm)          # drop points occluded by nearer scene
    n_inl = int(inl.sum())
    inlier_ratio = n_inl / max(int(considered.sum()), 1)
    residual = float(np.abs(dz[inl]).mean()) if n_inl else 99.0
    explained = 1.0
    if scene_pts is not None and len(scene_pts) > 0:
        d_obj, _ = model.tree.query((scene_pts - T[:3, 3]) @ T[:3, :3])
        near = d_obj < NEAR_MM
        if near.sum() >= 10:
            explained = float((d_obj[near] < tau_mm).mean())
    return Score(n_inl / n_vis * explained, inlier_ratio, explained, residual, n_vis)
