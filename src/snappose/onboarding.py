"""Offline onboarding: CAD -> sampled points/normals + manifest cache (spec 3.4, steps 1-3, 5, 8)."""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import Symmetry

SCHEMA = 1
N_SCORE_POINTS = 4000


@dataclass
class Model:
    object_id: str
    points: np.ndarray        # Nx3 CAD frame, mm
    normals: np.ndarray       # Nx3 outward
    center: np.ndarray        # bbox centre
    radius: float             # half bbox diagonal around `center`
    extents: np.ndarray
    watertight: bool
    symmetry: Symmetry
    tree: cKDTree
    score_points: np.ndarray
    score_normals: np.ndarray


def _hash(cad_path: Path, scale: float, n_points: int | None) -> str:
    h = hashlib.sha256(cad_path.read_bytes())
    h.update(f"{scale}|{n_points}|{SCHEMA}".encode())
    return h.hexdigest()


def sample_mesh(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    pts, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, float), np.asarray(mesh.face_normals[face], float)


def build_model(object_id: str, mesh: trimesh.Trimesh, symmetry: Symmetry | None = None,
                n_points: int | None = None) -> Model:
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    extents = hi - lo
    if n_points is None:  # ~1.2 mm spacing, capped
        n_points = int(np.clip(mesh.area / 1.4, 5000, 60000))
    points, normals = sample_mesh(mesh, n_points)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(points), min(N_SCORE_POINTS, len(points)), replace=False)
    center = (lo + hi) / 2
    return Model(object_id, points, normals, center, float(np.linalg.norm(extents) / 2), extents,
                 bool(mesh.is_watertight), symmetry or Symmetry(), cKDTree(points),
                 points[sel], normals[sel])


def onboard(object_id: str, cad_path: str | Path, store: str | Path = "objects", scale: float = 1.0,
            symmetry_path: str | Path | None = None, n_points: int | None = None,
            force: bool = False) -> Model:
    """Import a mesh (STL/PLY/OBJ), sample it and cache the result under store/<object_id>/.

    `scale` converts CAD units to mm (e.g. 1000 for metres). Output poses are in the (scaled) CAD frame.
    """
    cad_path = Path(cad_path)
    out = Path(store) / object_id
    digest = _hash(cad_path, scale, n_points)
    manifest = out / "manifest.json"
    if not force and manifest.exists() and json.loads(manifest.read_text()).get("hash") == digest:
        return load_model(object_id, store)

    loaded = trimesh.load(cad_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{cad_path} contains no triangle mesh (STEP must be converted to STL/PLY first)")
    loaded.apply_scale(scale)
    ext = loaded.extents
    if ext.max() < 1.0 or ext.max() > 2000.0:
        warnings.warn(f"bounding box {ext.round(2)} mm looks implausible; check units (scale={scale})")
    sym = Symmetry.from_json(symmetry_path) if symmetry_path else Symmetry()
    model = build_model(object_id, loaded, sym, n_points)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "points.npz", points=model.points, normals=model.normals,
                        center=model.center, radius=model.radius, extents=model.extents,
                        watertight=model.watertight)
    spec = json.loads(Path(symmetry_path).read_text()) if symmetry_path else {"symmetries": []}
    (out / "symmetries.json").write_text(json.dumps(spec, indent=2))
    manifest.write_text(json.dumps({"object_id": object_id, "hash": digest, "schema": SCHEMA,
                                    "cad": cad_path.name, "scale": scale, "n_points": len(model.points),
                                    "extents_mm": model.extents.round(2).tolist()}, indent=2))
    return model


def load_model(object_id: str, store: str | Path = "objects") -> Model:
    out = Path(store) / object_id
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"object {object_id!r} is not onboarded in {store!s} (run `snappose onboard`)")
    d = np.load(out / "points.npz")
    spec = json.loads((out / "symmetries.json").read_text())
    rng = np.random.default_rng(0)
    pts, nrm = d["points"], d["normals"]
    sel = rng.choice(len(pts), min(N_SCORE_POINTS, len(pts)), replace=False)
    return Model(object_id, pts, nrm, d["center"], float(d["radius"]), d["extents"], bool(d["watertight"]),
                 Symmetry.from_spec(spec.get("symmetries", [])), cKDTree(pts), pts[sel], nrm[sel])
