import numpy as np
import pytest

from snappose import PoseMatcher
from snappose.geometry import pose_error
from snappose.synth import DEFAULT_K, perturb, random_pose, render_rgbd


def _case(mesh, rng, seed=0, t=6.0, r=3.0):
    gt = random_pose(rng)
    _, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60, seed=seed)
    return gt, perturb(gt, rng, t, r), depth


def _matcher(model, profile):
    m = PoseMatcher(profile=profile)
    m.cfg = m.cfg.with_overrides({"prior": {"tolerance_t_mm": [8] * 3, "tolerance_r_deg": [5] * 3}})
    m.set_model(model)
    return m


@pytest.mark.parametrize("profile,tol", [("balanced", 2.0), ("precise", 1.5)])
def test_prior_match_accuracy(mesh, model, rng, profile, tol):
    m = _matcher(model, profile)
    for i in range(6):
        gt, prior, depth = _case(mesh, rng, i)
        res = m.match("bracket", depth, DEFAULT_K, prior)
        et, er = pose_error(res.T_cam_obj, gt)
        assert et < tol and er < tol, (i, et, er)
        assert res.status in ("OK", "UNSICHER")


def test_wrong_prior_is_not_ok(mesh, model, rng):
    gt, _, depth = _case(mesh, rng)
    far = perturb(gt, rng, 60, 40)                   # far outside the tolerance window
    res = _matcher(model, "balanced").match("bracket", depth, DEFAULT_K, far)
    assert res.status != "OK"


def test_empty_depth_gives_nok(model, rng):
    prior = random_pose(rng)
    res = _matcher(model, "fast").match("bracket", np.zeros((480, 640), np.float32), DEFAULT_K, prior)
    assert res.status == "NOK"


def test_time_budget_is_respected(mesh, model, rng):
    gt, prior, depth = _case(mesh, rng)
    m = _matcher(model, "precise")
    m.match("bracket", depth, DEFAULT_K, prior)       # warm-up
    res = m.match("bracket", depth, DEFAULT_K, prior, time_budget_ms=60)
    assert res.budget_exhausted
    assert res.status != "OK"                         # exhausted budget never yields OK
    assert res.timing_ms["total"] < 60 * 4            # soft budget: min. ICP iterations + scoring still run
    assert pose_error(res.T_cam_obj, gt)[0] < 8.0     # still returns a sensible pose


def test_result_json_schema(mesh, model, rng):
    gt, prior, depth = _case(mesh, rng)
    d = _matcher(model, "fast").match("bracket", depth, DEFAULT_K, prior).to_dict()
    for k in ("object_id", "status", "T_cam_obj", "translation_mm", "quaternion_xyzw", "confidence",
              "metrics", "mode", "profile", "budget_exhausted", "timing_ms", "versions"):
        assert k in d
    assert set(d["timing_ms"]) >= {"s0", "s2", "s3", "s4", "s5", "s6", "total"}
    assert np.array(d["T_cam_obj"]).shape == (4, 4)


def test_global_mode_not_implemented(model):
    with pytest.raises(NotImplementedError):
        _matcher(model, "fast").match("bracket", np.ones((480, 640), np.float32), DEFAULT_K, prior=None)


def test_deterministic(mesh, model, rng):
    gt, prior, depth = _case(mesh, rng)
    a = _matcher(model, "balanced").match("bracket", depth, DEFAULT_K, prior).T_cam_obj
    b = _matcher(model, "balanced").match("bracket", depth, DEFAULT_K, prior).T_cam_obj
    assert np.allclose(a, b)
