"""Evaluate prior-mode matching on BOP-format datasets (LM, TUD-L, ITODD, ...).

Every ground-truth instance (visibility >= --min-visib) is one trial: the prior is the GT pose perturbed by
a random offset (default +-6 mm / +-3 deg per axis), the matcher gets only depth + intrinsics + that prior.
Reported: success at absolute tolerances, ADD(-S) < 0.1 d (BOP/LINEMOD convention), false-OK rate, timing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .api import PoseMatcher
from .geometry import Symmetry, make_T, pose_error
from .onboarding import build_model
from .synth import perturb

TOLS = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]


def find_root(path: Path, split: str = "test") -> Path:
    for cand in [path, *sorted(path.glob("*"))]:
        if (cand / "models").is_dir() and (cand / split).is_dir():
            return cand
    raise FileNotFoundError(f"no BOP dataset (models/ + {split}/) under {path}")


def add_error(T_est, T_gt, pts, symmetric: bool) -> float:
    a = pts @ T_est[:3, :3].T + T_est[:3, 3]
    b = pts @ T_gt[:3, :3].T + T_gt[:3, 3]
    if symmetric:
        return float(cKDTree(b).query(a)[0].mean())
    return float(np.linalg.norm(a - b, axis=1).mean())


def load_models(root: Path, obj_ids: set[int] | None):
    info = json.loads((root / "models" / "models_info.json").read_text())
    models = {}
    for key, inf in info.items():
        oid = int(key)
        if obj_ids and oid not in obj_ids:
            continue
        mesh = trimesh.load(root / "models" / f"obj_{oid:06d}.ply", force="mesh")
        sym = Symmetry.from_bop(inf)
        models[oid] = (build_model(f"obj_{oid:02d}", mesh, sym), inf)
    return models


def run(dataset: str, profile: str = "balanced", per_object: int = 30, prior_t: float = 6.0, prior_r: float = 3.0,
        min_visib: float = 0.7, obj_ids: list[int] | None = None, seed: int = 0, overrides: dict | None = None,
        bop19_only: bool = True, split: str = "test", use_image: bool = True) -> dict:
    root = find_root(Path(dataset), split)
    rng = np.random.default_rng(seed)
    models = load_models(root, set(obj_ids) if obj_ids else None)
    matcher = PoseMatcher(profile=profile)
    matcher.cfg = matcher.cfg.with_overrides({"prior": {"tolerance_t_mm": [prior_t + 2] * 3,
                                                        "tolerance_r_deg": [prior_r + 2] * 3}, **(overrides or {})})
    for m, _ in models.values():
        matcher.set_model(m)

    targets = None
    tfile = root / "test_targets_bop19.json"
    if bop19_only and tfile.exists():
        targets = {(t["scene_id"], t["im_id"], t["obj_id"]) for t in json.loads(tfile.read_text())}

    trials: dict[int, list] = defaultdict(list)
    for sdir in sorted((root / split).iterdir()):
        if not sdir.is_dir():
            continue
        sid = int(sdir.name)
        cams = json.loads((sdir / "scene_camera.json").read_text())
        gts = json.loads((sdir / "scene_gt.json").read_text())
        infos = json.loads((sdir / "scene_gt_info.json").read_text())
        for im, gt_list in gts.items():
            for gi, g in enumerate(gt_list):
                oid = g["obj_id"]
                if oid not in models or len(trials[oid]) >= 10 * per_object:
                    continue
                if targets is not None and (sid, int(im), oid) not in targets:
                    continue
                if infos[im][gi].get("visib_fract", 1.0) < min_visib:
                    continue
                trials[oid].append((sid, im, gi))

    rows = []
    for oid, lst in sorted(trials.items()):
        pick = rng.permutation(len(lst))[:per_object]
        model, inf = models[oid]
        sym = model.symmetry
        symmetric = not sym.is_trivial
        sub = model.points[rng.choice(len(model.points), min(2000, len(model.points)), replace=False)]
        for pi in sorted(pick):
            sid, im, gi = lst[pi]
            sdir = root / split / f"{sid:06d}"
            cam = json.loads((sdir / "scene_camera.json").read_text())[im]
            g = json.loads((sdir / "scene_gt.json").read_text())[im][gi]
            dpath = next(p for p in (sdir / "depth" / f"{int(im):06d}.png", sdir / "depth" / f"{int(im):06d}.tif") if p.exists())
            depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED).astype(np.float32) * cam.get("depth_scale", 1.0)
            K = np.array(cam["cam_K"], float).reshape(3, 3)
            # some BOP GT rotations (e.g. TUD-L) are only orthonormal to ~1e-3: project onto SO(3)
            R_gt = Rotation.from_matrix(np.array(g["cam_R_m2c"]).reshape(3, 3)).as_matrix()
            T_gt = make_T(R_gt, np.array(g["cam_t_m2c"]))
            prior = perturb(T_gt, rng, prior_t, prior_r)
            img = None
            if use_image:
                for kind in ("rgb", "gray"):
                    cand = [q for q in (sdir / kind / f"{int(im):06d}.png", sdir / kind / f"{int(im):06d}.tif",
                                        sdir / kind / f"{int(im):06d}.jpg") if q.exists()]
                    if cand:
                        img = cv2.imread(str(cand[0]), cv2.IMREAD_UNCHANGED)
                        if img is not None and img.ndim == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        break
            res = matcher.match(f"obj_{oid:02d}", depth, K, prior, rgb=img)
            et, er = pose_error(res.T_cam_obj, T_gt, sym)
            add = add_error(res.T_cam_obj, T_gt, sub, symmetric)
            rows.append({"obj": oid, "scene": sid, "im": int(im), "t_err": et, "r_err": er,
                         "add_ok": add < 0.1 * inf["diameter"], "status": res.status,
                         "ms": res.timing_ms["total"], "metrics": res.metrics, "conf": res.confidence, "prior_t_err": pose_error(prior, T_gt, sym)[0], "prior_r_err": pose_error(prior, T_gt, sym)[1]})
    out = summarize(rows, dataset, profile)
    out["rows"] = rows
    return out


def summarize(rows: list[dict], dataset: str, profile: str) -> dict:
    def agg(rs):
        e = np.array([[r["t_err"], r["r_err"]] for r in rs])
        ok = [r for r in rs if r["status"] == "OK"]
        bad_ok = [r for r in ok if r["t_err"] > 5 or r["r_err"] > 5]
        return {
            "n": len(rs),
            "success": {f"{t:g}mm/{r:g}deg": float(np.mean((e[:, 0] <= t) & (e[:, 1] <= r))) for t, r in TOLS},
            "add_s_0.1d": float(np.mean([r["add_ok"] for r in rs])),
            "prior_t_median_mm": float(np.median([r["prior_t_err"] for r in rs])),
            "prior_r_median_deg": float(np.median([r["prior_r_err"] for r in rs])),
            "t_err_median_mm": float(np.median(e[:, 0])), "r_err_median_deg": float(np.median(e[:, 1])),
            "status": {s: sum(r["status"] == s for r in rs) for s in ("OK", "UNSICHER", "NOK")},
            "false_ok_gt5": len(bad_ok), "ok_total": len(ok),
            "time_ms_mean": float(np.mean([r["ms"] for r in rs])), "time_ms_p95": float(np.percentile([r["ms"] for r in rs], 95)),
        }
    per_obj = {str(o): agg([r for r in rows if r["obj"] == o]) for o in sorted({r["obj"] for r in rows})}
    return {"dataset": dataset, "profile": profile, "overall": agg(rows) if rows else {}, "per_object": per_obj}


def format_report(res: dict) -> str:
    def line(name, a):
        s = a["success"]
        return (f"{name:>8} {a['n']:4d}  {s['1mm/1deg']*100:5.0f}% {s['2mm/2deg']*100:5.0f}% {s['5mm/5deg']*100:5.0f}%  "
                f"{a['add_s_0.1d']*100:6.0f}%  {a['prior_t_median_mm']:5.1f} {a['prior_r_median_deg']:5.1f} > {a['t_err_median_mm']:5.2f} {a['r_err_median_deg']:5.2f}  "
                f"{a['status']['OK']:3d}/{a['status']['UNSICHER']:3d}/{a['status']['NOK']:3d}  {a['false_ok_gt5']:2d}/{a['ok_total']:3d}  "
                f"{a['time_ms_mean']:6.0f} {a['time_ms_p95']:6.0f}")
    head = (f"{res['dataset']}  profile={res['profile']}\n"
            f"{'object':>8} {'n':>4}  {'1mm':>6} {'2mm':>6} {'5mm':>6}  {'ADD-S':>7}  prior(mm,deg) > result  OK/UNS/NOK  falseOK  t_ms  p95")
    rows = [line(f"obj {o}", a) for o, a in res["per_object"].items()]
    return "\n".join([head, *rows, line("ALL", res["overall"])])


def read_image(sdir: Path, im: int):
    """Colour (RGB) or grey image of a BOP frame, or None."""
    for kind in ("rgb", "gray"):
        for ext in ("png", "tif", "jpg"):
            q = sdir / kind / f"{im:06d}.{ext}"
            if q.exists():
                img = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
    return None


def read_depth(sdir: Path, im: int, scale: float) -> np.ndarray:
    for ext in ("png", "tif"):
        q = sdir / "depth" / f"{im:06d}.{ext}"
        if q.exists():
            return cv2.imread(str(q), cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
    raise FileNotFoundError(f"no depth image {im} in {sdir}")


def random_instance(dataset: str, obj_id: int | None, rng: np.random.Generator, split: str | None = None,
                    min_visib: float = 0.7) -> dict:
    """One random ground-truth instance of a BOP dataset: depth, image, K, T_gt, ids."""
    base = Path(dataset)
    split = split or ("test" if find_root_or_none(base, "test") else "val")
    root = find_root(base, split)
    scenes = [d for d in sorted((root / split).iterdir()) if d.is_dir()]
    for _ in range(200):
        sdir = scenes[int(rng.integers(len(scenes)))]
        gts = json.loads((sdir / "scene_gt.json").read_text())
        infos = json.loads((sdir / "scene_gt_info.json").read_text())
        cams = json.loads((sdir / "scene_camera.json").read_text())
        im = list(gts)[int(rng.integers(len(gts)))]
        cand = [(i, g) for i, g in enumerate(gts[im])
                if (obj_id is None or g["obj_id"] == obj_id) and infos[im][i].get("visib_fract", 1.0) >= min_visib]
        if not cand:
            continue
        gi, g = cand[int(rng.integers(len(cand)))]
        cam = cams[im]
        R = Rotation.from_matrix(np.array(g["cam_R_m2c"]).reshape(3, 3)).as_matrix()
        return {"depth": read_depth(sdir, int(im), cam.get("depth_scale", 1.0)), "rgb": read_image(sdir, int(im)),
                "K": np.array(cam["cam_K"], float).reshape(3, 3), "T_gt": make_T(R, np.array(g["cam_t_m2c"])),
                "obj_id": g["obj_id"], "scene": int(sdir.name), "im": int(im), "root": root}
    raise LookupError("no matching instance found")


def find_root_or_none(path: Path, split: str):
    try:
        return find_root(path, split)
    except FileNotFoundError:
        return None
