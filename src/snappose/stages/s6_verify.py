"""S6: confidence, status and symmetry canonicalisation."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..geometry import Symmetry, pose_error
from ..types import Hypothesis
from .s4_score import Score

DISTINCT_T_MM = 5.0
DISTINCT_R_DEG = 5.0


def score_margin(best: Hypothesis, others: list[Hypothesis], sym: Symmetry) -> float:
    """Score gap to the best hypothesis that is a genuinely different pose (symmetry-aware)."""
    for h in sorted(others, key=lambda h: -h.score):
        et, er = pose_error(h.T, best.T, sym)
        if et > DISTINCT_T_MM or er > DISTINCT_R_DEG:
            return float(best.score - h.score)
    return 1.0


def status(sc: Score, margin: float, budget_exhausted: bool, cfg: Config) -> str:
    v = cfg.verification
    ok = (sc.inlier_ratio >= v.min_inlier_ratio and sc.explained >= v.min_explained_ratio
          and sc.residual_mm <= v.max_residual_mm
          and margin >= v.min_score_margin)
    if sc.n_visible == 0 or sc.inlier_ratio < 0.5 * v.min_inlier_ratio or sc.residual_mm > 3.0 * v.max_residual_mm:
        return "NOK"
    if ok and not budget_exhausted:
        return "OK"
    return "UNSICHER"
