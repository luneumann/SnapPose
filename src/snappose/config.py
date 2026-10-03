"""Config: profile -> YAML overrides -> call overrides -> time budget."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class S0:
    input_scale: float = 0.75
    depth_min_mm: float = 100.0
    depth_max_mm: float = 3000.0
    flying_pixel_mm: float = 8.0


@dataclass
class S2:
    n_perturbations: int = 5
    seed: int = 0


@dataclass
class S3:
    refiner: str = "icp"
    refine_iters: int = 4          # rounds; hypotheses are scored/pruned after each round
    iters_per_round: int = 6
    prune_schedule: str = "halving"  # halving | none
    coarse_voxel_mm: float = 3.0
    early_stop_delta_mm: float = 0.02


@dataclass
class S4:
    scorer: str = "depth_residual"
    inlier_tau_mm: float | None = None   # None = estimate from the sensor's depth step size (clip 2..8 mm)


@dataclass
class S5:
    icp_enabled: bool = True
    icp_iters: int = 30
    icp_voxel_mm: float = 1.5
    use_window: bool = True      # clamp refinement to the prior tolerance window (x1.25)


@dataclass
class Edges:
    enabled: bool = True          # only active if an image is passed to match()
    weight: float = 1.0           # relative influence of the edge group vs the depth group
    score_weight: float = 0.7     # share of the image-edge term in the hypothesis score
    iters: int = 20
    max_depth_loss: float = 0.10  # edge result must keep the depth inlier ratio within this of the depth-only result
    min_edge_gain: float = 0.10   # ... and improve the edge inlier ratio by at least this
    max_px: float = 12.0          # edge residuals above this are ignored (shrinks to 3 px)


@dataclass
class Prior:
    tolerance_t_mm: list = field(default_factory=lambda: [10.0, 10.0, 10.0])
    tolerance_r_deg: list = field(default_factory=lambda: [5.0, 5.0, 5.0])
    frame: str = "camera"           # camera | object


@dataclass
class Verification:
    min_inlier_ratio: float = 0.85
    min_explained_ratio: float = 0.5
    max_residual_mm: float = 1.5
    min_score_margin: float = 0.0


PROFILES: dict[str, dict] = {
    "fast": {
        "time_budget_ms": 400,
        "s0": {"input_scale": 0.5},
        "s2": {"n_perturbations": 1},
        "s3": {"refine_iters": 2, "prune_schedule": "halving", "coarse_voxel_mm": 4.0},
        "s4": {"scorer": "depth_residual"},
        "s5": {"icp_iters": 15, "icp_voxel_mm": 2.5},
    },
    "balanced": {
        "time_budget_ms": 1200,
        "s0": {"input_scale": 0.75},
        "s2": {"n_perturbations": 5},
        "s3": {"refine_iters": 4, "prune_schedule": "halving", "coarse_voxel_mm": 3.0},
        "s4": {"scorer": "depth_residual"},
        "s5": {"icp_iters": 30, "icp_voxel_mm": 1.5},
    },
    "precise": {
        "time_budget_ms": 4000,
        "s0": {"input_scale": 1.0},
        "s2": {"n_perturbations": 15},
        "s3": {"refine_iters": 6, "prune_schedule": "none", "coarse_voxel_mm": 2.0},
        "s4": {"scorer": "depth_residual"},
        "s5": {"icp_iters": 60, "icp_voxel_mm": 1.0},
    },
}

_SECTIONS = {"s0": S0, "s2": S2, "s3": S3, "s4": S4, "s5": S5, "edges": Edges, "prior": Prior, "verification": Verification}


@dataclass
class Config:
    license_mode: str = "commercial"    # commercial | research
    profile: str = "balanced"
    mode: str = "auto"                  # auto | prior  (global: planned, see README)
    time_budget_ms: float = 800
    s0: S0 = field(default_factory=S0)
    s2: S2 = field(default_factory=S2)
    s3: S3 = field(default_factory=S3)
    s4: S4 = field(default_factory=S4)
    s5: S5 = field(default_factory=S5)
    edges: Edges = field(default_factory=Edges)
    prior: Prior = field(default_factory=Prior)
    verification: Verification = field(default_factory=Verification)

    @classmethod
    def load(cls, source: "str | Path | dict | Config | None" = None, *, profile: str | None = None,
             overrides: dict | None = None, time_budget_ms: float | None = None) -> "Config":
        if isinstance(source, Config):
            return source
        raw: dict = {}
        if isinstance(source, (str, Path)):
            raw = yaml.safe_load(Path(source).read_text()) or {}
        elif isinstance(source, dict):
            raw = copy.deepcopy(source)
        prof = profile or raw.get("profile", "balanced")
        if prof not in PROFILES:
            raise ValueError(f"unknown profile {prof!r}; available: {sorted(PROFILES)}")
        merged = copy.deepcopy(PROFILES[prof])
        top = {k: raw[k] for k in ("license_mode", "mode", "time_budget_ms") if k in raw}
        sections = {k: raw[k] for k in ("prior", "verification") if k in raw}
        _merge(merged, top)
        _merge(merged, sections)
        _merge(merged, raw.get("overrides", {}))
        _merge(merged, overrides or {})
        if time_budget_ms is not None:
            merged["time_budget_ms"] = time_budget_ms
        return cls._build(prof, merged)

    def with_overrides(self, overrides: dict | None = None, *, profile: str | None = None,
                       time_budget_ms: float | None = None) -> "Config":
        """New config: optionally switch profile (resets profile values), then apply overrides."""
        if profile and profile != self.profile:
            base = {"license_mode": self.license_mode, "mode": self.mode, "prior": asdict(self.prior),
                    "verification": asdict(self.verification)}
            return Config.load(base, profile=profile, overrides=overrides, time_budget_ms=time_budget_ms)
        d = {k: asdict(getattr(self, k)) for k in _SECTIONS}
        d.update(license_mode=self.license_mode, mode=self.mode, time_budget_ms=self.time_budget_ms)
        _merge(d, overrides or {})
        if time_budget_ms is not None:
            d["time_budget_ms"] = time_budget_ms
        return self._build(self.profile, d)

    @classmethod
    def _build(cls, profile: str, d: dict) -> "Config":
        known = {f.name for f in fields(cls)} - {"profile"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs = {}
        for name, sec_cls in _SECTIONS.items():
            vals = d.get(name, {})
            valid = {f.name for f in fields(sec_cls)}
            bad = set(vals) - valid
            if bad:
                raise ValueError(f"unknown keys in section {name!r}: {sorted(bad)} (valid: {sorted(valid)})")
            kwargs[name] = sec_cls(**vals)
        scalars = {k: d[k] for k in ("license_mode", "mode", "time_budget_ms") if k in d}
        cfg = cls(profile=profile, **scalars, **kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.license_mode not in ("commercial", "research"):
            raise ValueError("license_mode must be 'commercial' or 'research'")
        if self.mode not in ("auto", "prior"):
            raise ValueError("mode must be 'auto' or 'prior' (global mode is not implemented yet)")
        if not 0.1 <= self.s0.input_scale <= 1.0:
            raise ValueError("s0.input_scale must be in [0.1, 1.0]")
        if self.s3.prune_schedule not in ("halving", "none"):
            raise ValueError("s3.prune_schedule must be 'halving' or 'none'")
        if self.prior.frame not in ("camera", "object"):
            raise ValueError("prior.frame must be 'camera' or 'object'")
        for key in ("tolerance_t_mm", "tolerance_r_deg"):
            if len(getattr(self.prior, key)) != 3:
                raise ValueError(f"prior.{key} needs 3 values")


def _merge(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
