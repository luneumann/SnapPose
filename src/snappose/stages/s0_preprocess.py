"""S0: depth filtering, optional downscale, ROI around the prior, point extraction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config
from ..geometry import backproject, voxel_downsample
from ..onboarding import Model


@dataclass
class Scene:
    depth: np.ndarray          # filtered, scaled depth (mm, 0 = invalid)
    K: np.ndarray              # intrinsics matching `depth`
    roi_points: np.ndarray     # Nx3 camera points inside the ROI sphere (not yet voxelised)
    roi_center: np.ndarray
    roi_radius: float


def filter_depth(depth: np.ndarray, cfg: Config) -> np.ndarray:
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[(d < cfg.s0.depth_min_mm) | (d > cfg.s0.depth_max_mm)] = 0
    valid = d > 0
    dmax = cv2.dilate(np.where(valid, d, 0).astype(np.float32), np.ones((3, 3), np.uint8))
    dmin = cv2.erode(np.where(valid, d, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))
    flying = valid & ((dmax - dmin) > cfg.s0.flying_pixel_mm) & (dmin < 1e8)
    d[flying] = 0
    return d


def scale_inputs(depth: np.ndarray, K: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if scale >= 0.999:
        return depth, K
    h, w = depth.shape
    ws, hs = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    sx, sy = ws / w, hs / h
    d = cv2.resize(depth, (ws, hs), interpolation=cv2.INTER_NEAREST)
    Ks = K.astype(float).copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return d, Ks


def run(depth_mm: np.ndarray, K: np.ndarray, prior: np.ndarray, model: Model, cfg: Config) -> Scene:
    d = filter_depth(depth_mm, cfg)
    d, Ks = scale_inputs(d, np.asarray(K, float), cfg.s0.input_scale)

    center = prior[:3, :3] @ model.center + prior[:3, 3]
    tol_t = float(np.linalg.norm(cfg.prior.tolerance_t_mm))
    tol_r = np.radians(max(cfg.prior.tolerance_r_deg))
    radius = model.radius * (1.0 + min(tol_r, 1.0)) + tol_t + 5.0

    z = max(center[2], 1.0)
    u0 = Ks[0, 0] * center[0] / z + Ks[0, 2]
    v0 = Ks[1, 1] * center[1] / z + Ks[1, 2]
    half_u = Ks[0, 0] * radius / z + 2
    half_v = Ks[1, 1] * radius / z + 2
    h, w = d.shape
    c0, c1 = int(max(u0 - half_u, 0)), int(min(u0 + half_u + 1, w))
    r0, r1 = int(max(v0 - half_v, 0)), int(min(v0 + half_v + 1, h))
    if c1 <= c0 or r1 <= r0:
        pts = np.zeros((0, 3))
    else:
        crop = d[r0:r1, c0:c1]
        pts = backproject(crop, Ks, offset=(r0, c0))
        pts = pts[np.linalg.norm(pts - center, axis=1) < radius]
    return Scene(d, Ks, pts, center, radius)


def voxelize(scene: Scene, voxel_mm: float) -> np.ndarray:
    return voxel_downsample(scene.roi_points, voxel_mm)
