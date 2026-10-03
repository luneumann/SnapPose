from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import quat_xyzw


@dataclass
class Hypothesis:
    T: np.ndarray            # 4x4 T_cam_obj
    score: float = 0.0


@dataclass
class MatchResult:
    object_id: str
    status: str              # OK | UNSICHER | NOK
    T_cam_obj: np.ndarray
    confidence: float
    metrics: dict
    mode: str
    profile: str
    symmetry_canonicalized: bool
    budget_exhausted: bool
    timing_ms: dict
    versions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        T = self.T_cam_obj
        return {
            "object_id": self.object_id,
            "status": self.status,
            "T_cam_obj": T.tolist(),
            "translation_mm": T[:3, 3].round(3).tolist(),
            "quaternion_xyzw": quat_xyzw(T[:3, :3]).round(6).tolist(),
            "confidence": round(float(self.confidence), 4),
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "mode": self.mode,
            "profile": self.profile,
            "symmetry_canonicalized": self.symmetry_canonicalized,
            "budget_exhausted": self.budget_exhausted,
            "timing_ms": {k: round(float(v), 1) for k, v in self.timing_ms.items()},
            "versions": self.versions,
        }
