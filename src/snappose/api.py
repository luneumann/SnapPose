from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .budget import Budget
from .config import Config
from .onboarding import Model, load_model, onboard
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
        lap("s0", t)

        t = time.perf_counter()
        hyps = s2_hypotheses.run(prior, cfg)
        lap("s2", t)

        t = time.perf_counter()
        ref = s3_refine.run(hyps, scene, model, cfg, budget)
        lap("s3", t)
        best = ref.ranked[0]

        t = time.perf_counter()
        T = best.T
        if cfg.s5.icp_enabled and len(scene.roi_points) > 0:
            pts = s0_preprocess.voxelize(scene, cfg.s5.icp_voxel_mm)
            tau1 = max(cfg.s5.icp_voxel_mm, 0.8)
            tau0 = max(2.0 * cfg.s3.coarse_voxel_mm, 3.0 * tau1)
            tau0 = min(tau0, ref.tau0)
            T, _ = s5_icp.icp(T, pts, model, cfg.s5.icp_iters, tau0, tau1, deadline=budget.deadline,
                              min_iters=5, tol_mm=cfg.s3.early_stop_delta_mm)
        lap("s5", t)

        t = time.perf_counter()
        sc = s4_score.score_pose(T, scene.depth, scene.K, model, cfg.s4.inlier_tau_mm,
                                 s0_preprocess.voxelize(scene, cfg.s3.coarse_voxel_mm))
        lap("s4", t)

        t = time.perf_counter()
        best.T, best.score = T, sc.score
        margin = s6_verify.score_margin(best, ref.ranked[1:], model.symmetry)
        T_out = model.symmetry.canonicalize(T, prior)
        exhausted = ref.budget_exhausted or budget.exhausted()
        status = s6_verify.status(sc, margin, exhausted, cfg)
        lap("s6", t)
        timing["total"] = budget.elapsed_ms()

        from . import __version__
        return MatchResult(
            object_id=object_id, status=status, T_cam_obj=T_out, confidence=sc.score,
            metrics={"inlier_ratio": sc.inlier_ratio, "explained_ratio": sc.explained, "residual_mm": sc.residual_mm, "score_margin": margin},
            mode="prior", profile=cfg.profile, symmetry_canonicalized=not model.symmetry.is_trivial,
            budget_exhausted=exhausted, timing_ms=timing,
            versions={"pipeline": __version__, "refiner": cfg.s3.refiner, "backbone": "none"})

