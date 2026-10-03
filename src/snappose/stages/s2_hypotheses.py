"""S2 (prior mode): prior pose + Sobol perturbations inside the tolerance window."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc
from scipy.spatial.transform import Rotation

from ..config import Config


def run(prior: np.ndarray, cfg: Config) -> list[np.ndarray]:
    hyps = [prior.copy()]
    n = cfg.s2.n_perturbations
    if n <= 0:
        return hyps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning for non-power-of-2 n
        u = qmc.Sobol(d=6, scramble=True, seed=cfg.s2.seed).random(n)
    u = 2.0 * u - 1.0                     # [-1, 1]^6
    tol_t = np.asarray(cfg.prior.tolerance_t_mm, float)
    tol_r = np.radians(np.asarray(cfg.prior.tolerance_r_deg, float))
    R0, t0 = prior[:3, :3], prior[:3, 3]
    for row in u:
        dt, dr = row[:3] * tol_t, row[3:] * tol_r
        Rd = Rotation.from_euler("xyz", dr).as_matrix()
        T = np.eye(4)
        if cfg.prior.frame == "camera":
            T[:3, :3] = Rd @ R0
            T[:3, 3] = t0 + dt
        else:
            T[:3, :3] = R0 @ Rd
            T[:3, 3] = t0 + R0 @ dt
        hyps.append(T)
    return hyps
