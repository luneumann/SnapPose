"""Synthetic RGB-D rendering of a CAD model (point splatting) for demos, tests and benchmarks."""
from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .geometry import make_T, project

DEFAULT_K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
DEFAULT_SIZE = (480, 640)  # H, W


def make_bracket() -> trimesh.Trimesh:
    """Asymmetric test part (~80x50x35 mm): base plate, cylindrical boss, offset fin."""
    base = trimesh.creation.box(extents=[80, 50, 10])
    boss = trimesh.creation.cylinder(radius=12, height=25, transform=trimesh.transformations.translation_matrix([-20, 8, 17.5]))
    fin = trimesh.creation.box(extents=[6, 30, 20], transform=trimesh.transformations.translation_matrix([28, -5, 15]))
    mesh = trimesh.util.concatenate([base, boss, fin])
    mesh.merge_vertices()
    return mesh


def render_rgbd(mesh: trimesh.Trimesh, T: np.ndarray, K: np.ndarray = DEFAULT_K, size=DEFAULT_SIZE,
                background_z: float | None = None, noise_mm: float = 0.3, dropout: float = 0.01,
                seed: int = 0, n_dense: int = 600_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb HxWx3 uint8, depth HxW float32 mm). background_z adds a flat table plane."""
    rng = np.random.default_rng(seed)
    h, w = size
    pts, face = trimesh.sample.sample_surface(mesh, n_dense, seed=seed)
    nrm = mesh.face_normals[face]
    p = pts @ T[:3, :3].T + T[:3, 3]
    n = nrm @ T[:3, :3].T
    front = np.einsum("ij,ij->i", n, p) < 0
    p, n = p[front], n[front]
    u, v = project(p, K)
    shade = np.clip(-n[:, 2] * 0.6 + 0.25 - n[:, 1] * 0.2, 0.05, 1.0)

    # one pixel per point; depth = mean z of the nearest layer in the pixel (unbiased for tilted surfaces)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    p, ui, vi, shade = p[ok], ui[ok], vi[ok], shade[ok]
    flat = vi * w + ui
    zmin = np.full(h * w, np.inf)
    np.minimum.at(zmin, flat, p[:, 2])
    near = p[:, 2] <= zmin[flat] + 1.0
    cnt = np.bincount(flat[near], minlength=h * w)
    zsum = np.bincount(flat[near], weights=p[near, 2], minlength=h * w)
    ssum = np.bincount(flat[near], weights=shade[near], minlength=h * w)
    hit = cnt > 0
    zbuf = np.where(hit, zsum / np.maximum(cnt, 1), np.inf).reshape(h, w)
    col = np.where(hit, ssum / np.maximum(cnt, 1), 0.0).reshape(h, w)
    depth = np.where(np.isfinite(zbuf), zbuf, 0.0)
    img = (col * 255).astype(np.uint8)
    if background_z is not None:
        empty = depth == 0
        depth[empty] = background_z
        img[empty] = 70
    depth = depth.astype(np.float32)
    valid = depth > 0
    depth[valid] += rng.normal(0, noise_mm, valid.sum()).astype(np.float32)
    depth[rng.random(depth.shape) < dropout] = 0
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb, depth


def random_pose(rng: np.random.Generator, z_range=(450.0, 700.0)) -> np.ndarray:
    yaw = rng.uniform(-np.pi, np.pi)
    tilt = np.radians(rng.uniform(0, 40))
    tilt_dir = rng.uniform(-np.pi, np.pi)
    R = (Rotation.from_euler("z", yaw) * Rotation.from_rotvec([tilt * np.cos(tilt_dir), tilt * np.sin(tilt_dir), 0])).as_matrix()
    # object looks "up" at the camera: flip about x so that +z of the part points towards the camera
    R = Rotation.from_euler("x", np.pi).as_matrix() @ R
    t = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(*z_range)])
    return make_T(R, t)


def perturb(T: np.ndarray, rng: np.random.Generator, t_mm: float, r_deg: float) -> np.ndarray:
    """Random pose offset with per-axis magnitudes uniform in [-t_mm, t_mm] / [-r_deg, r_deg] (camera frame)."""
    dt = rng.uniform(-t_mm, t_mm, 3)
    dr = Rotation.from_euler("xyz", np.radians(rng.uniform(-r_deg, r_deg, 3))).as_matrix()
    return make_T(dr @ T[:3, :3], T[:3, 3] + dt)
