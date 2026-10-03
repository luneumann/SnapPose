from __future__ import annotations

import cv2
import numpy as np

from .geometry import project
from .onboarding import Model


def overlay(rgb: np.ndarray, model: Model, T: np.ndarray, K: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draw the CAD points under pose T on the image (rgb in, rgb out)."""
    img = rgb.copy()
    p = model.score_points @ T[:3, :3].T + T[:3, 3]
    n = model.score_normals @ T[:3, :3].T
    keep = np.einsum("ij,ij->i", n, p) < 0 if model.watertight else np.ones(len(p), bool)
    u, v = project(p[keep], K)
    h, w = img.shape[:2]
    for x, y in zip(np.round(u).astype(int), np.round(v).astype(int)):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 1, color, -1)
    c = project((T[:3, :3] @ model.center + T[:3, 3])[None], K)
    cv2.putText(img, model.object_id, (int(c[0][0]) - 20, max(int(c[1][0]) - int(model.radius * K[0, 0] / T[2, 3]) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
