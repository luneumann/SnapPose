"""S5: trimmed point-to-plane ICP, scene -> model (model normals are known, scene normals are not)."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..onboarding import Model


def icp(T: np.ndarray, scene_pts: np.ndarray, model: Model, iters: int, tau0: float, tau1: float,
        deadline: float | None = None, min_iters: int = 3, tol_mm: float = 0.02,
        min_pts: int = 12) -> tuple[np.ndarray, int]:
    """Refine T_cam_obj. Correspondence radius tau shrinks geometrically tau0 -> tau1.

    Stops early on convergence or when `deadline` (perf_counter) passes after `min_iters` iterations.
    Returns (T, iterations_done).
    """
    R, t = T[:3, :3].copy(), T[:3, 3].copy()
    done = 0
    for i in range(iters):
        if deadline is not None and i >= min_iters and time.perf_counter() > deadline:
            break
        tau = tau0 * (tau1 / tau0) ** (i / max(iters - 1, 1))
        p = (scene_pts - t) @ R                      # scene in object frame
        d, idx = model.tree.query(p, distance_upper_bound=tau)
        m = np.isfinite(d)
        if m.sum() < min_pts:
            break
        s, q, n, dd = p[m], model.points[idx[m]], model.normals[idx[m]], d[m]
        if model.watertight:
            # a camera only sees front faces: drop matches onto back faces (e.g. the far side of a plate)
            facing = np.einsum("ij,ij->i", n @ R.T, scene_pts[m]) < 0.02 * np.linalg.norm(scene_pts[m], axis=1)
            if facing.sum() < min_pts:
                break
            s, q, n, dd = s[facing], q[facing], n[facing], dd[facing]
        sw = (1.0 - (dd / tau) ** 2)                 # sqrt of Tukey weight
        A = np.hstack([np.cross(s, n), n]) * sw[:, None]
        b = -np.einsum("ij,ij->i", s - q, n) * sw
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        Dr = Rotation.from_rotvec(x[:3]).as_matrix()
        Rn = R @ Dr.T                                # T <- T * D^-1
        t = t - Rn @ x[3:]
        R = Rn
        done += 1
        if np.linalg.norm(x[3:]) < tol_mm and np.linalg.norm(x[:3]) < 1e-4:
            break
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out, done
