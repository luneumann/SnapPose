"""Pose math. Convention: OpenCV camera, mm, T_cam_obj is 4x4 float64 (p_cam = R p_obj + t)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rotvec_T(rotvec, t) -> np.ndarray:
    return make_T(Rotation.from_rotvec(rotvec).as_matrix(), np.asarray(t, float))


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_about(axis, angle_rad, origin) -> np.ndarray:
    """4x4 rotation by angle about the line through `origin` with direction `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    origin = np.asarray(origin, float)
    R = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    return make_T(R, origin - R @ origin)


@dataclass
class Symmetry:
    """Object symmetries in the CAD frame.

    JSON (list under "symmetries"): {"type":"cyclic","axis":[0,0,1],"order":4,"origin":[0,0,0]}
    or {"type":"continuous","axis":[0,0,1],"origin":[0,0,0]}.
    """
    discrete: list[np.ndarray] = field(default_factory=lambda: [np.eye(4)])
    continuous: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def is_trivial(self) -> bool:
        return len(self.discrete) == 1 and not self.continuous

    @classmethod
    def from_json(cls, path: str | Path) -> "Symmetry":
        spec = json.loads(Path(path).read_text())
        return cls.from_spec(spec.get("symmetries", []))

    @classmethod
    def from_spec(cls, items: list[dict]) -> "Symmetry":
        discrete = [np.eye(4)]
        continuous = []
        for it in items:
            origin = np.asarray(it.get("origin", [0, 0, 0]), float)
            axis = np.asarray(it["axis"], float)
            if it["type"] == "cyclic":
                n = int(it["order"])
                group = [rotation_about(axis, 2 * np.pi * k / n, origin) for k in range(1, n)]
                discrete = [D @ G for D in discrete for G in [np.eye(4)] + group]
            elif it["type"] == "continuous":
                continuous.append((axis / np.linalg.norm(axis), origin))
            else:
                raise ValueError(f"unknown symmetry type {it['type']!r}")
        return cls(discrete, continuous)

    def to_spec(self) -> list[dict]:
        # Only used for round-trips of continuous axes; discrete groups are stored via the source JSON.
        return [{"type": "continuous", "axis": a.tolist(), "origin": o.tolist()} for a, o in self.continuous]

    def canonicalize(self, T: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
        """Return the symmetric equivalent of T whose rotation is closest to T_ref's."""
        if self.is_trivial:
            return T
        best = min((T @ S for S in self.discrete), key=lambda c: rot_angle_deg(c[:3, :3], T_ref[:3, :3]))
        for axis, origin in self.continuous:
            angles = np.radians(np.arange(-180, 180, 0.5))
            cands = [best @ rotation_about(axis, a, origin) for a in angles]
            best = min(cands, key=lambda c: rot_angle_deg(c[:3, :3], T_ref[:3, :3]))
        return best

    def equivalents(self, T: np.ndarray) -> list[np.ndarray]:
        return [T @ S for S in self.discrete]


def pose_error(T_est: np.ndarray, T_gt: np.ndarray, sym: Symmetry | None = None) -> tuple[float, float]:
    """(translation error [mm], rotation error [deg]) minimised over discrete symmetries."""
    cands = sym.equivalents(T_est) if sym else [T_est]
    errs = [(float(np.linalg.norm(c[:3, 3] - T_gt[:3, 3])), rot_angle_deg(c[:3, :3], T_gt[:3, :3])) for c in cands]
    return min(errs, key=lambda e: e[0] / 1.0 + e[1])


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None,
                offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Pixel grid -> Nx3 camera points. `offset` = (row0, col0) of a crop."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    ok = z > 0 if mask is None else mask & (z > 0)
    u = u[ok] + offset[1]
    v = v[ok] + offset[0]
    z = z[ok]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    return K[0, 0] * points[:, 0] / zs + K[0, 2], K[1, 1] * points[:, 1] / zs + K[1, 2]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]
