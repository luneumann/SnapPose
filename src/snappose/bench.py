"""Synthetic benchmark: success rate at fixed tolerances, error percentiles and timing per profile."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .api import PoseMatcher
from .config import PROFILES
from .geometry import pose_error
from .onboarding import build_model
from .synth import DEFAULT_K, make_bracket, perturb, random_pose, render_rgbd

TOLERANCES = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]


def run(n: int = 20, profiles=None, prior_t_mm: float = 6.0, prior_r_deg: float = 3.0, seed: int = 0,
        warm: bool = True) -> dict:
    mesh = make_bracket()
    model = build_model("bracket", mesh)
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        gt = random_pose(rng)
        _, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60, seed=seed + i)
        cases.append((gt, perturb(gt, rng, prior_t_mm, prior_r_deg), depth))

    out = {}
    for prof in profiles or list(PROFILES):
        m = PoseMatcher(profile=prof)
        m.cfg = m.cfg.with_overrides({"prior": {"tolerance_t_mm": [prior_t_mm + 2] * 3,
                                                "tolerance_r_deg": [prior_r_deg + 2] * 3}})
        m.set_model(model)
        if warm:
            m.match("bracket", cases[0][2], DEFAULT_K, cases[0][1])
        errs, times, stats = [], [], []
        for gt, prior, depth in cases:
            r = m.match("bracket", depth, DEFAULT_K, prior)
            errs.append(pose_error(r.T_cam_obj, gt))
            times.append(r.timing_ms["total"])
            stats.append(r.status)
        e = np.array(errs)
        out[prof] = {
            "n": n,
            "success": {f"{t:g}mm/{r:g}deg": float(np.mean((e[:, 0] <= t) & (e[:, 1] <= r))) for t, r in TOLERANCES},
            "t_err_mm": {"median": float(np.median(e[:, 0])), "p95": float(np.percentile(e[:, 0], 95))},
            "r_err_deg": {"median": float(np.median(e[:, 1])), "p95": float(np.percentile(e[:, 1], 95))},
            "time_ms": {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95))},
            "status": {s: stats.count(s) for s in ("OK", "UNSICHER", "NOK")},
        }
    return out


def format_table(res: dict) -> str:
    keys = list(next(iter(res.values()))["success"])
    head = f"{'profile':9} " + " ".join(f"{k:>14}" for k in keys) + "  t_med[mm]  r_med[deg]  t_mean[ms]  t_p95[ms]  OK/UNS/NOK"
    lines = [head]
    for p, r in res.items():
        s = r["status"]
        lines.append(f"{p:9} " + " ".join(f"{r['success'][k] * 100:13.0f}%" for k in keys)
                     + f"  {r['t_err_mm']['median']:9.2f}  {r['r_err_deg']['median']:10.2f}"
                     + f"  {r['time_ms']['mean']:10.0f}  {r['time_ms']['p95']:9.0f}  {s['OK']}/{s['UNSICHER']}/{s['NOK']}")
    return "\n".join(lines)
