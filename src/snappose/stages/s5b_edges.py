"""S5b: joint depth + image-edge refinement.

Depth-only point-to-plane ICP cannot constrain the pose along flat surfaces (washers, plates, brackets lying on
a table). Sharp CAD edges projected into the image and pulled onto image edges (distance transform) do.
Both residual groups are stacked into one Gauss-Newton system over a camera-frame twist; each group is
normalised to unit total weight so `edges.weight` is a relative influence, independent of point counts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from ..onboarding import Model


@dataclass
class EdgeMap:
    dt: np.ndarray        # distance to nearest image edge, px
    gx: np.ndarray
    gy: np.ndarray


def prepare(img: np.ndarray, size_hw: tuple[int, int], pct: float = 93.0) -> EdgeMap:
    """Image (gray or RGB, any bit depth) -> distance transform of Canny edges at the depth resolution."""
    g = img
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) if g.shape[2] == 3 else g[..., 0]
    if g.dtype != np.uint8:
        g = g.astype(np.float32)
        lo, hi = np.percentile(g, [0.5, 99.5])
        g = np.clip((g - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h, w = size_hw
    if g.shape != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0.8)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    hi = float(np.percentile(mag, pct))
    edges = cv2.Canny(g, 0.4 * hi, hi, L2gradient=True)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return EdgeMap(dt, cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=3) / 8.0)


def _sample(a: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return map_coordinates(a, [v, u], order=1, mode="nearest")


def _edge_rows(R, t, model: Model, em: EdgeMap, depth: np.ndarray, K: np.ndarray, cap_px: float):
    if model.edge_points is None or len(model.edge_points) == 0:
        return None
    p = model.edge_points @ R.T + t
    if model.watertight:
        vis = (np.einsum("ij,ij->i", model.edge_n1 @ R.T, p) < 0) | (np.einsum("ij,ij->i", model.edge_n2 @ R.T, p) < 0)
        p = p[vis]
    z = p[:, 2]
    u = K[0, 0] * p[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * p[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    h, w = em.dt.shape
    ok = (z > 1) & (u > 2) & (u < w - 3) & (v > 2) & (v < h - 3)
    p, z, u, v = p[ok], z[ok], u[ok], v[ok]
    if len(p) == 0:
        return None
    # drop edge points hidden behind nearer scene surface (invalid depth is accepted: shiny edges lose depth)
    d = depth[np.round(v).astype(int), np.round(u).astype(int)]
    ok = (d == 0) | (d > z - 6.0)
    p, z, u, v = p[ok], z[ok], u[ok], v[ok]
    r = _sample(em.dt, u, v)
    ok = r < cap_px
    p, z, u, v, r = p[ok], z[ok], u[ok], v[ok], r[ok]
    if len(p) < 10:
        return None
    gx, gy = _sample(em.gx, u, v), _sample(em.gy, u, v)
    a = np.stack([gx * K[0, 0] / z, gy * K[1, 1] / z, -(gx * K[0, 0] * p[:, 0] + gy * K[1, 1] * p[:, 1]) / z ** 2], axis=1)
    A = np.hstack([np.cross(p, a), a])
    mm = z / K[0, 0]                                  # px -> mm at the point's depth
    wgt = (1.0 - (r / cap_px) ** 2) ** 2              # Tukey
    return A * mm[:, None], -r * mm, wgt


def edge_inlier_ratio(T: np.ndarray, model: Model, em: EdgeMap, depth: np.ndarray, K: np.ndarray,
                      thr_px: float = 2.0) -> float:
    """Fraction of visible CAD edge points within thr_px of an image edge (-1 if undefined)."""
    R, t = T[:3, :3], T[:3, 3]
    p = model.edge_points @ R.T + t
    if model.watertight:
        vis = (np.einsum("ij,ij->i", model.edge_n1 @ R.T, p) < 0) | (np.einsum("ij,ij->i", model.edge_n2 @ R.T, p) < 0)
        p = p[vis]
    z = p[:, 2]
    u = K[0, 0] * p[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * p[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    h, w = em.dt.shape
    ok = (z > 1) & (u > 2) & (u < w - 3) & (v > 2) & (v < h - 3)
    z, u, v = z[ok], u[ok], v[ok]
    if len(z) < 10:
        return -1.0
    d = depth[np.round(v).astype(int), np.round(u).astype(int)]
    ok = (d == 0) | (d > z - 6.0)
    u, v = u[ok], v[ok]
    if len(u) < 10:
        return -1.0
    return float(np.mean(_sample(em.dt, u, v) < thr_px))


def refine(T: np.ndarray, scene_pts: np.ndarray, model: Model, em: EdgeMap, depth: np.ndarray, K: np.ndarray,
           iters: int, tau0: float, tau1: float, weight: float = 1.0, max_px: float = 12.0, min_px: float = 3.0, window=None,
           deadline: float | None = None, tol_mm: float = 0.02, min_pts: int = 12) -> tuple[np.ndarray, int]:
    R, t = T[:3, :3].copy(), T[:3, 3].copy()
    done = 0
    for i in range(iters):
        if deadline is not None and i >= 2 and time.perf_counter() > deadline:
            break
        f = i / max(iters - 1, 1)
        tau = tau0 * (tau1 / tau0) ** f
        cap = max_px * (min_px / max_px) ** f
        rows, tgt, wts = [], [], []

        p_obj = (scene_pts - t) @ R
        d, idx = model.tree.query(p_obj, distance_upper_bound=tau)
        m = np.isfinite(d)
        if m.sum() >= min_pts:
            s_c = scene_pts[m]
            q_c = model.points[idx[m]] @ R.T + t
            n_c = model.normals[idx[m]] @ R.T
            keep = np.ones(len(s_c), bool)
            if model.watertight:
                keep = np.einsum("ij,ij->i", n_c, s_c) < 0.02 * np.linalg.norm(s_c, axis=1)
            if keep.sum() >= min_pts:
                s_c, q_c, n_c, dd = s_c[keep], q_c[keep], n_c[keep], d[m][keep]
                e = np.einsum("ij,ij->i", n_c, s_c - q_c)
                w_d = (1.0 - (dd / tau) ** 2) ** 2
                rows.append(np.hstack([np.cross(q_c, n_c), n_c]))
                tgt.append(e)
                wts.append(w_d / max(w_d.sum(), 1e-9))
        if weight > 0:
            er = _edge_rows(R, t, model, em, depth, K, cap)
            if er is not None:
                A, b, w_e = er
                rows.append(A)
                tgt.append(b)
                wts.append(weight * w_e / max(w_e.sum(), 1e-9))
        if not rows:
            break
        A, b, w = np.vstack(rows), np.concatenate(tgt), np.concatenate(wts)
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        Dr = Rotation.from_rotvec(x[:3]).as_matrix()
        R, t = Dr @ R, Dr @ t + x[3:]
        if window is not None:
            R, t = window.clamp(R, t)
        done += 1
        if np.linalg.norm(x[3:]) < tol_mm and np.linalg.norm(x[:3]) < 1e-4:
            break
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out, done
