"""Prior tolerance window: keeps refined poses inside the region the caller said the object can be in."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import Config

MARGIN = 1.25   # refinement may use a little more than the nominal window


class Window:
    def __init__(self, prior: np.ndarray, cfg: Config, enabled: bool = True):
        self.enabled = enabled
        self.R0, self.t0 = prior[:3, :3], prior[:3, 3]
        self.tol_t = np.asarray(cfg.prior.tolerance_t_mm, float) * MARGIN
        self.tol_r = np.radians(np.asarray(cfg.prior.tolerance_r_deg, float)) * MARGIN
        self.frame = cfg.prior.frame

    def clamp(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return R, t
        dt = t - self.t0
        if self.frame == "object":
            dt = self.R0.T @ dt
        dt = np.clip(dt, -self.tol_t, self.tol_t)
        t = self.t0 + (self.R0 @ dt if self.frame == "object" else dt)
        if self.frame == "camera":
            rel = R @ self.R0.T
        else:
            rel = self.R0.T @ R
        e = Rotation.from_matrix(rel).as_euler("xyz")
        ec = np.clip(e, -self.tol_r, self.tol_r)
        if np.any(ec != e):
            rc = Rotation.from_euler("xyz", ec).as_matrix()
            R = rc @ self.R0 if self.frame == "camera" else self.R0 @ rc
        return R, t

    def inside(self, T: np.ndarray, slack: float = 1.0) -> bool:
        R, t = self.clamp(T[:3, :3], T[:3, 3])
        return bool(np.allclose(t, T[:3, 3], atol=1e-6) and np.allclose(R, T[:3, :3], atol=1e-6))
