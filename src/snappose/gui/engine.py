"""GUI state and actions. HTTP handlers only call methods here; matching runs in a worker thread."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .. import bench, eval_bop
from ..api import PoseMatcher
from ..config import PROFILES
from ..geometry import make_T, pose_error, project
from ..onboarding import Model, build_model, load_model, onboard
from ..synth import DEFAULT_K, make_bracket, perturb, random_pose, render_rgbd

DEMO_ID = "demo-bracket"
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
COLORS = {"prior": (255, 150, 30), "result": (60, 230, 120), "gt": (90, 170, 255), "edges": (255, 255, 255)}


class Engine:
    def __init__(self, store: str | Path = "objects", bop_dir: str | Path = "data/bop"):
        self.store = Path(store)
        self.bop_dir = Path(bop_dir)
        self.matcher = PoseMatcher(store=self.store)
        self.lock = threading.RLock()
        self.settings = {"profile": "balanced", "time_budget_ms": None, "tol_t_mm": 10.0, "tol_r_deg": 5.0,
                         "edges": True, "prior_err_t": 6.0, "prior_err_r": 3.0}
        self.object_id: str | None = None
        self.frame: dict | None = None      # depth, K, rgb, T_gt, source, label
        self.prior: np.ndarray | None = None
        self.result: dict | None = None
        self.result_T: np.ndarray | None = None
        self.busy = ""                      # "" | "match" | "bench"
        self.error = ""
        self.bench_result: dict | None = None
        self._upload: dict = {}
        self._seed = 1
        self._demo_mesh = None

    # ------------------------------------------------------------------ objects
    def objects(self) -> list[dict]:
        out = [{"id": DEMO_ID, "label": "Demo-Teil (synthetisch)", "demo": True, "extents": [80, 50, 35]}]
        if self.store.exists():
            for d in sorted(self.store.iterdir()):
                mf = d / "manifest.json"
                if mf.exists():
                    m = json.loads(mf.read_text())
                    out.append({"id": d.name, "label": d.name, "demo": False, "extents": m.get("extents_mm"),
                                "n_points": m.get("n_points"), "cad": m.get("cad")})
        return out

    def _model(self, object_id: str) -> Model:
        if object_id == DEMO_ID:
            if DEMO_ID not in self.matcher._models:
                self._demo_mesh = make_bracket()
                self.matcher.set_model(build_model(DEMO_ID, self._demo_mesh))
            return self.matcher._models[DEMO_ID]
        return self.matcher.model(object_id)

    def select_object(self, object_id: str) -> None:
        if object_id not in {o["id"] for o in self.objects()}:
            raise ValueError("unbekanntes Objekt")
        with self.lock:
            self._model(object_id)
            self.object_id = object_id
            self.result = self.result_T = None
            if self.frame and self.frame.get("T_gt") is not None and self.frame["source"] != "demo":
                self.frame = None                # a BOP frame belongs to its own object

    def onboard_upload(self, name: str, ext: str, data: bytes, scale: float) -> None:
        if not NAME_RE.fullmatch(name) or name == DEMO_ID:
            raise ValueError("Name: 1–40 Zeichen, nur Buchstaben, Ziffern, _ und -")
        if ext.lower() not in ("stl", "ply", "obj"):
            raise ValueError("CAD-Format: STL, PLY oder OBJ (STEP bitte vorher zu STL konvertieren)")
        if not 0.0001 < scale < 1e6:
            raise ValueError("Maßstab ungültig")
        tmp = self.store / "_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        f = tmp / f"{name}.{ext.lower()}"
        f.write_bytes(data)
        try:
            self.matcher.onboard(name, f, scale=scale, force=True)
        except Exception as e:  # trimesh raises many types for bad meshes
            raise ValueError(f"CAD konnte nicht gelesen werden: {e}") from e
        self.select_object(name)

    def delete_object(self, name: str) -> None:
        if not NAME_RE.fullmatch(name) or name == DEMO_ID:
            raise ValueError("ungültiger Name")
        d = self.store / name
        if d.is_dir() and (d / "manifest.json").exists():
            shutil.rmtree(d)
        self.matcher._models.pop(name, None)
        if self.object_id == name:
            self.object_id = None

    # ------------------------------------------------------------------ frames
    def demo_frame(self, seed: int | None = None) -> None:
        with self.lock:
            self.select_object(DEMO_ID)
            self._seed = int(seed) if seed is not None else self._seed + 1
            rng = np.random.default_rng(self._seed)
            gt = random_pose(rng)
            rgb, depth = render_rgbd(self._demo_mesh, gt, background_z=gt[2, 3] + 60, seed=self._seed)
            self.frame = {"source": "demo", "label": f"Demo (Seed {self._seed})", "depth": depth, "rgb": rgb,
                          "K": DEFAULT_K.copy(), "T_gt": gt}
            self._roll_prior(rng)

    def bop_datasets(self) -> list[dict]:
        out = []
        if self.bop_dir.exists():
            for d in sorted(self.bop_dir.iterdir()):
                for split in ("test", "val"):
                    if eval_bop.find_root_or_none(d, split):
                        out.append({"id": d.name, "split": split})
                        break
        return out

    def bop_frame(self, dataset: str, obj_id: int | None) -> None:
        if not any(d["id"] == dataset for d in self.bop_datasets()):
            raise ValueError("Datensatz nicht gefunden")
        with self.lock:
            rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 31))
            inst = eval_bop.random_instance(str(self.bop_dir / dataset), obj_id, rng)
            oid = f"{dataset}-obj{inst['obj_id']:02d}"
            if oid not in self.matcher._models:
                models = eval_bop.load_models(inst["root"], {inst["obj_id"]})
                m, _ = models[inst["obj_id"]]
                m.object_id = oid
                self.matcher.set_model(m)
            self.object_id = oid
            self.frame = {"source": "bop", "label": f"{dataset} · Szene {inst['scene']} · Bild {inst['im']} · Obj {inst['obj_id']}",
                          "depth": inst["depth"], "rgb": inst["rgb"], "K": inst["K"], "T_gt": inst["T_gt"]}
            self._roll_prior(rng)

    def bop_objects(self, dataset: str) -> list[int]:
        for d in self.bop_datasets():
            if d["id"] == dataset:
                info = self.bop_dir / dataset
                root = eval_bop.find_root(info, d["split"])
                return sorted(int(k) for k in json.loads((root / "models" / "models_info.json").read_text()))
        return []

    def upload_part(self, kind: str, data: bytes) -> None:
        if kind == "depth":
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 2:
                raise ValueError("Tiefenbild muss ein einkanaliges PNG/TIF sein (16 Bit, mm)")
            self._upload["depth"] = img
        elif kind == "color":
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Farbbild nicht lesbar")
            self._upload["rgb"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        else:
            raise ValueError("unbekannte Art")

    def set_camera(self, d: dict) -> None:
        depth = self._upload.get("depth")
        if depth is None:
            raise ValueError("Zuerst ein Tiefenbild laden")
        fx, fy, cx, cy = (float(d[k]) for k in ("fx", "fy", "cx", "cy"))
        scale = float(d.get("depth_scale", 1.0))
        if min(fx, fy) <= 0:
            raise ValueError("Brennweite muss positiv sein")
        with self.lock:
            self.frame = {"source": "files", "label": "Eigene Dateien", "depth": depth.astype(np.float32) * scale,
                          "rgb": self._upload.get("rgb"), "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), "T_gt": None}
            self.prior = None
            self.result = self.result_T = None

    # ------------------------------------------------------------------ prior
    def _roll_prior(self, rng: np.random.Generator | None = None) -> None:
        f = self.frame
        if f is None or f.get("T_gt") is None:
            return
        rng = rng or np.random.default_rng()
        self.prior = perturb(f["T_gt"], rng, self.settings["prior_err_t"], self.settings["prior_err_r"])
        self.result = self.result_T = None

    def reroll_prior(self) -> None:
        with self.lock:
            self._roll_prior()

    def set_prior(self, t: list, euler_deg: list) -> None:
        if len(t) != 3 or len(euler_deg) != 3:
            raise ValueError("x, y, z und drei Winkel angeben")
        R = Rotation.from_euler("xyz", np.radians([float(v) for v in euler_deg])).as_matrix()
        with self.lock:
            self.prior = make_T(R, np.array([float(v) for v in t]))
            self.result = self.result_T = None

    def guess_prior(self) -> None:
        """Convenience for own files: centre of the nearest valid depth blob, identity-ish rotation."""
        with self.lock:
            f = self.frame
            if f is None:
                raise ValueError("Zuerst eine Aufnahme laden")
            d, K = f["depth"], f["K"]
            valid = d > 0
            if not valid.any():
                raise ValueError("Tiefenbild enthält keine gültigen Werte")
            z = np.percentile(d[valid], 5)
            ys, xs = np.nonzero(valid & (d < z + 40))
            u, v = xs.mean(), ys.mean()
            zc = float(np.median(d[ys, xs]))
            t = np.array([(u - K[0, 2]) * zc / K[0, 0], (v - K[1, 2]) * zc / K[1, 1], zc])
            self.prior = make_T(np.diag([1.0, -1.0, -1.0]), t)
            self.result = self.result_T = None

    # ------------------------------------------------------------------ settings
    def update_settings(self, d: dict) -> None:
        with self.lock:
            s = self.settings
            if "profile" in d:
                if d["profile"] not in PROFILES:
                    raise ValueError("unbekanntes Profil")
                s["profile"] = d["profile"]
            if "time_budget_ms" in d:
                v = d["time_budget_ms"]
                s["time_budget_ms"] = None if v in (None, "", 0) else float(v)
            for k, lo, hi in (("tol_t_mm", 0.5, 200), ("tol_r_deg", 0.5, 45), ("prior_err_t", 0, 100), ("prior_err_r", 0, 45)):
                if k in d:
                    s[k] = float(np.clip(float(d[k]), lo, hi))
            if "edges" in d:
                s["edges"] = bool(d["edges"])

    # ------------------------------------------------------------------ match / bench
    def start_match(self) -> None:
        with self.lock:
            if self.busy:
                raise ValueError("Es läuft bereits eine Berechnung")
            if not self.object_id:
                raise ValueError("Zuerst ein Objekt wählen")
            if self.frame is None:
                raise ValueError("Zuerst eine Aufnahme laden")
            if self.prior is None:
                raise ValueError("Startlage fehlt (Schritt 3)")
            self.busy, self.error = "match", ""
        threading.Thread(target=self._run_match, daemon=True).start()

    def _run_match(self) -> None:
        try:
            s, f = self.settings, self.frame
            res = self.matcher.match(
                self.object_id, f["depth"], f["K"], self.prior, rgb=f["rgb"],
                profile=s["profile"], time_budget_ms=s["time_budget_ms"],
                overrides={"prior": {"tolerance_t_mm": [s["tol_t_mm"]] * 3, "tolerance_r_deg": [s["tol_r_deg"]] * 3},
                           "edges": {"enabled": s["edges"]}})
            out = res.to_dict()
            if f.get("T_gt") is not None:
                et, er = pose_error(res.T_cam_obj, f["T_gt"], self._model(self.object_id).symmetry)
                out["error_vs_gt"] = {"t_mm": round(et, 3), "r_deg": round(er, 3)}
                pt, pr = pose_error(self.prior, f["T_gt"], self._model(self.object_id).symmetry)
                out["prior_error_vs_gt"] = {"t_mm": round(pt, 3), "r_deg": round(pr, 3)}
            with self.lock:
                self.result, self.result_T = out, res.T_cam_obj
        except Exception as e:  # shown in the GUI, not swallowed
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = ""

    def start_bench(self, n: int) -> None:
        with self.lock:
            if self.busy:
                raise ValueError("Es läuft bereits eine Berechnung")
            self.busy, self.error = "bench", ""
        threading.Thread(target=self._run_bench, args=(int(np.clip(n, 5, 60)),), daemon=True).start()

    def _run_bench(self, n: int) -> None:
        try:
            self.bench_result = bench.run(n=n)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = ""

    # ------------------------------------------------------------------ status / image
    def snapshot(self) -> dict:
        with self.lock:
            f = self.frame
            return {
                "busy": self.busy, "error": self.error, "settings": dict(self.settings), "profiles": sorted(PROFILES),
                "object": self.object_id, "objects": self.objects(), "bop": self.bop_datasets(),
                "frame": None if f is None else {"source": f["source"], "label": f["label"],
                                                 "size": list(f["depth"].shape[::-1]), "has_rgb": f["rgb"] is not None,
                                                 "has_gt": f.get("T_gt") is not None},
                "prior": None if self.prior is None else {
                    "t": self.prior[:3, 3].round(2).tolist(),
                    "euler_deg": np.degrees(Rotation.from_matrix(self.prior[:3, :3]).as_euler("xyz")).round(2).tolist()},
                "result": self.result, "bench": self.bench_result,
                "profile_budget_ms": {k: v["time_budget_ms"] for k, v in PROFILES.items()},
            }

    def image_jpeg(self, view: str, layers: set[str], width: int = 960) -> bytes | None:
        with self.lock:
            f = self.frame
            if f is None:
                return None
            model = self._model(self.object_id) if self.object_id else None
            poses = {"prior": self.prior, "result": self.result_T, "gt": f.get("T_gt") if f["source"] != "files" else None}
            depth, rgb, K = f["depth"], f["rgb"], f["K"]
        h, w = depth.shape
        if view == "rgb" and rgb is not None:
            base = rgb if rgb.ndim == 3 else cv2.cvtColor(_to_u8(rgb), cv2.COLOR_GRAY2RGB)
            if base.dtype != np.uint8:
                base = cv2.cvtColor(_to_u8(rgb), cv2.COLOR_GRAY2RGB)
        else:
            valid = depth > 0
            lo, hi = (np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1))
            d8 = np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            base = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
            base[~valid] = 0
        if base.shape[:2] != (h, w):
            base = cv2.resize(base, (w, h))

        # crop around the object (it is usually tiny in a full frame)
        ref = poses["result"] if poses["result"] is not None else poses["prior"]
        x0, y0, x1, y1 = 0, 0, w, h
        if ref is not None and model is not None:
            c = ref[:3, :3] @ model.center + ref[:3, 3]
            if c[2] > 1:
                u, v = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]
                half = max(model.radius * K[0, 0] / c[2] * 1.7, 80)
                bw, bh = half * 2, half * 2 * 0.75
                if bw < w:
                    x0 = int(np.clip(u - bw / 2, 0, w - bw)); x1 = int(x0 + bw)
                    y0 = int(np.clip(v - bh / 2, 0, h - bh)); y1 = int(y0 + bh)
        crop = base[y0:y1, x0:x1]
        s = width / crop.shape[1]
        img = cv2.resize(crop, (width, int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        K2 = np.array([[K[0, 0] * s, 0, (K[0, 2] - x0 + 0.5) * s - 0.5], [0, K[1, 1] * s, (K[1, 2] - y0 + 0.5) * s - 0.5], [0, 0, 1]])
        if model is not None:
            for name in ("gt", "prior", "result"):
                if name in layers and poses[name] is not None:
                    _draw(img, model, poses[name], K2, COLORS[name], edges=False)
            if "edges" in layers and poses["result"] is not None:
                _draw(img, model, poses["result"], K2, COLORS["edges"], edges=True)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None


def _to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _draw(img: np.ndarray, model: Model, T: np.ndarray, K: np.ndarray, color, edges: bool) -> None:
    if edges:
        pts, n1, n2 = model.edge_points, model.edge_n1, model.edge_n2
        if pts is None or len(pts) == 0:
            return
        p = pts @ T[:3, :3].T + T[:3, 3]
        keep = (np.einsum("ij,ij->i", n1 @ T[:3, :3].T, p) < 0) | (np.einsum("ij,ij->i", n2 @ T[:3, :3].T, p) < 0) \
            if model.watertight else np.ones(len(p), bool)
        r = 1
    else:
        p = model.score_points @ T[:3, :3].T + T[:3, 3]
        keep = np.einsum("ij,ij->i", model.score_normals @ T[:3, :3].T, p) < 0 if model.watertight else np.ones(len(p), bool)
        r = 2
    u, v = project(p[keep], K)
    hh, ww = img.shape[:2]
    for x, y in zip(np.round(u).astype(int), np.round(v).astype(int)):
        if 0 <= x < ww and 0 <= y < hh:
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
