"""S3: multi-hypothesis refinement with scoring and successive halving.

Refiners are pluggable (registry). V1 ships a geometric refiner (coarse ICP rounds). A render-and-compare
network refiner (MegaPose) can be added as another registry entry; see docs/SPEC.md section 3.2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..budget import Budget
from ..config import Config
from ..onboarding import Model
from ..registry import REFINERS
from ..types import Hypothesis
from . import s4_score
from .s0_preprocess import Scene, voxelize
from .s5_icp import icp


@REFINERS.register("icp", license="own")
def refine_icp(T, scene_pts, model, cfg: Config, tau0: float, tau1: float, deadline):
    T, _ = icp(T, scene_pts, model, cfg.s3.iters_per_round, tau0, tau1, deadline=deadline,
               min_iters=2, tol_mm=cfg.s3.early_stop_delta_mm)
    return T


@REFINERS.register("megapose", license="apache-2.0")
def refine_megapose(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError("MegaPose refiner is not integrated yet (planned, see README roadmap)")


@dataclass
class RefineOutput:
    ranked: list[Hypothesis]       # best first, incl. pruned hypotheses (last known pose/score)
    budget_exhausted: bool
    tau0: float


def auto_tau0(cfg: Config) -> float:
    return float(np.clip(0.6 * max(cfg.prior.tolerance_t_mm), 4.0, 15.0))


def run(hyps: list[np.ndarray], scene: Scene, model: Model, cfg: Config, budget: Budget) -> RefineOutput:
    refine = REFINERS.get(cfg.s3.refiner, cfg.license_mode)
    pts = voxelize(scene, cfg.s3.coarse_voxel_mm)
    tau0 = auto_tau0(cfg)
    tau_end = max(2.0 * cfg.s3.coarse_voxel_mm, cfg.s4.inlier_tau_mm)
    tau_end = min(tau_end, tau0)
    rounds = max(cfg.s3.refine_iters, 1)
    ratio = tau_end / tau0

    active = [Hypothesis(T) for T in hyps]
    pruned: list[Hypothesis] = []
    s3_deadline = None
    if budget.total_ms is not None:
        s3_deadline = budget.deadline - budget.reserve_ms("s4", "s5") / 1000.0
    exhausted = False
    round_ms = 0.0

    for r in range(rounds):
        if r > 0 and s3_deadline is not None and time.perf_counter() + round_ms / 1000.0 > s3_deadline:
            exhausted = True
            break
        t_round = time.perf_counter()
        t0_r, t1_r = tau0 * ratio ** (r / rounds), tau0 * ratio ** ((r + 1) / rounds)
        for h in active:
            h.T = refine(h.T, pts, model, cfg, t0_r, t1_r, s3_deadline)
            h.score = s4_score.score_pose(h.T, scene.depth, scene.K, model, cfg.s4.inlier_tau_mm, pts).score
        active.sort(key=lambda h: -h.score)
        # budget pressure forces pruning even for prune_schedule "none"
        tight = s3_deadline is not None and (s3_deadline - time.perf_counter()) < 1.5 * (time.perf_counter() - t_round)
        if len(active) > 1 and r < rounds - 1 and (cfg.s3.prune_schedule == "halving" or tight):
            keep = max((len(active) + 1) // 2, 1)
            pruned += active[keep:]
            active = active[:keep]
        round_ms = (time.perf_counter() - t_round) * 1000.0

    ranked = sorted(active + pruned, key=lambda h: -h.score)
    return RefineOutput(ranked, exhausted, tau0)
