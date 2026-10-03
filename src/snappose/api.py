from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .budget import Budget
from .config import Config
from .onboarding import Model, load_model, onboard
from .stages.window import Window
from .stages import s5b_edges
from .stages import s0_preprocess, s2_hypotheses, s3_refine, s4_score, s5_icp, s6_verify
from .types import MatchResult


class PoseMatcher:
    """Single-shot 6D pose matching of a CAD object in one RGB-D frame (prior mode)."""

    def __init__(self, config: "str | Path | dict | Config | None" = None, store: str | Path = "objects",
                 profile: str | None = None):
        self.cfg = Config.load(config, profile=profile)
        self.store = Path(store)
        self._models: dict[str, Model] = {}

    def onboard(self, object_id: str, cad_path: str | Path, **kw) -> Model:
        self._models[object_id] = onboard(object_id, cad_path, store=self.store, **kw)
        return self._models[object_id]

    def set_model(self, model: Model) -> None:
        """Register an in-memory model (tests, synthetic data)."""
        self._models[model.object_id] = model

    def _solve(self, model, scene, hyps, cfg, budget, window, prior, em, tm):
        """S3 refinement + S5 fine ICP (+ edge refinement if em) + final scoring for one candidate."""
        edge = None
        if em is not None:
            px_per_mm = scene.K[0, 0] / max(prior[2, 3], 1.0)
            cap0 = float(np.clip(1.2 * max(cfg.prior.tolerance_t_mm) * px_per_mm, cfg.edges.max_px, 80.0))
            edge = (em, scene.depth, scene.K, cap0)
        t0 = time.perf_counter()
        ref = s3_refine.run(hyps, scene, model, cfg, budget, window, edge)
        tm['s3'] = tm.get('s3', 0.0) + (time.perf_counter() - t0) * 1000.0
        best = ref.ranked[0]
        T = best.T
        t0 = time.perf_counter()
        if cfg.s5.icp_enabled and len(scene.roi_points) > 0:
            pts = s0_preprocess.voxelize(scene, cfg.s5.icp_voxel_mm)
            tau1 = max(cfg.s5.icp_voxel_mm, 0.8)
            tau0 = min(max(2.0 * cfg.s3.coarse_voxel_mm, 3.0 * tau1), ref.tau0)
            if em is None:
                T, _ = s5_icp.icp(T, pts, model, cfg.s5.icp_iters, tau0, tau1, deadline=budget.deadline,
                                  min_iters=5, tol_mm=cfg.s3.early_stop_delta_mm, window=window)
            else:
                T, _ = s5b_edges.refine(T, pts, model, em, scene.depth, scene.K, cfg.s5.icp_iters, tau0, tau1,
                                        weight=cfg.edges.weight, max_px=cfg.edges.max_px, min_px=3.0, window=window,
                                        deadline=budget.deadline, tol_mm=cfg.s3.early_stop_delta_mm)
        t1 = time.perf_counter()
        tm['s5'] = tm.get('s5', 0.0) + (t1 - t0) * 1000.0
        sc = s4_score.score_pose(T, scene.depth, scene.K, model, scene.tau_mm,
                                 s0_preprocess.voxelize(scene, cfg.s3.coarse_voxel_mm),
                                 (em, cfg.edges.score_weight) if em is not None else None)
        tm['s4'] = tm.get('s4', 0.0) + (time.perf_counter() - t1) * 1000.0
        return T, sc, best, ref

    def model(self, object_id: str) -> Model:
        if object_id not in self._models:
            self._models[object_id] = load_model(object_id, self.store)
        return self._models[object_id]

    def match(self, object_id: str, depth: np.ndarray, K: np.ndarray, prior: np.ndarray | None = None,
              rgb: np.ndarray | None = None, overrides: dict | None = None,
              profile: str | None = None, time_budget_ms: float | None = None) -> MatchResult:
        """depth: HxW float, mm, 0 = invalid, registered to the colour image. rgb is accepted for API
        stability but unused in V1 (depth-only matching)."""
        cfg = self.cfg
        if overrides or profile or time_budget_ms is not None:
            cfg = cfg.with_overrides(overrides, profile=profile, time_budget_ms=time_budget_ms)
        if prior is None:
            raise NotImplementedError("global mode (no prior) is not implemented yet; pass a prior pose")
        model = self.model(object_id)
        prior = np.asarray(prior, float)
        budget = Budget(cfg.time_budget_ms)
        timing: dict[str, float] = {}

        def lap(name: str, t_start: float) -> None:
            timing[name] = (time.perf_counter() - t_start) * 1000.0

        t = time.perf_counter()
        scene = s0_preprocess.run(depth, K, prior, model, cfg)
        em = None
        if rgb is not None and cfg.edges.enabled:
            em = s5b_edges.prepare(rgb, scene.depth.shape)
        lap("s0", t)

        t = time.perf_counter()
        hyps = s2_hypotheses.run(prior, cfg)
        lap("s2", t)

        window = Window(prior, cfg, enabled=cfg.s5.use_window)
        T, sc, best, ref = self._solve(model, scene, hyps, cfg, budget, window, prior, None, timing)
        edge_used = False
        if em is not None and budget.remaining_ms() > 0:
            # second candidate that also uses image edges; adopted only if depth consistency is kept
            T2, sc2, best2, ref2 = self._solve(model, scene, hyps, cfg, budget, window, prior, em, timing)
            e1 = _edge_ratio(T, model, em, scene, cfg)
            if (sc2.inlier_ratio >= sc.inlier_ratio - cfg.edges.max_depth_loss
                    and sc2.edge_ratio >= e1 + cfg.edges.min_edge_gain):
                T, sc, best, ref, edge_used = T2, sc2, best2, ref2, True
            else:
                sc.edge_ratio = e1

        t = time.perf_counter()
        best.T, best.score = T, sc.score
        margin = s6_verify.score_margin(best, ref.ranked[1:], model.symmetry)
        T_out = model.symmetry.canonicalize(T, prior)
        exhausted = ref.budget_exhausted or budget.exhausted()
        status = s6_verify.status(sc, margin, exhausted, cfg, scene.tau_mm)
        lap("s6", t)
        timing["total"] = budget.elapsed_ms()

        from . import __version__
        return MatchResult(
            object_id=object_id, status=status, T_cam_obj=T_out, confidence=sc.score,
            metrics={"inlier_ratio": sc.inlier_ratio, "explained_ratio": sc.explained, "edge_ratio": sc.edge_ratio,
                     "residual_mm": sc.residual_mm, "score_margin": margin, "edges_used": float(edge_used), "tau_mm": scene.tau_mm},
            mode="prior", profile=cfg.profile, symmetry_canonicalized=not model.symmetry.is_trivial,
            budget_exhausted=exhausted, timing_ms=timing,
            versions={"pipeline": __version__, "refiner": cfg.s3.refiner, "backbone": "none"})



def _edge_ratio(T, model, em, scene, cfg) -> float:
    return s5b_edges.edge_inlier_ratio(T, model, em, scene.depth, scene.K)
