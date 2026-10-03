import numpy as np
import trimesh

from snappose import PoseMatcher
from snappose.config import Config
from snappose.geometry import pose_error, rotvec_T
from snappose.onboarding import build_model, crease_edges
from snappose.stages import s0_preprocess
from snappose.stages.window import Window
from snappose.synth import DEFAULT_K, perturb, random_pose, render_rgbd


def test_crease_edges_of_box():
    box = trimesh.creation.box(extents=[20, 10, 5])
    pts, n1, n2 = crease_edges(box, spacing=1.0)
    assert len(pts) > 100
    assert np.allclose(np.linalg.norm(n1, axis=1), 1) and np.allclose(np.linalg.norm(n2, axis=1), 1)
    # edge points lie on the box edges: at least two coordinates at +-half extent
    on = (np.isclose(np.abs(pts), np.array([10, 5, 2.5]), atol=1e-6)).sum(axis=1)
    assert (on >= 2).all()


def test_window_clamps_translation_and_rotation():
    cfg = Config.load(overrides={"prior": {"tolerance_t_mm": [5, 5, 5], "tolerance_r_deg": [2, 2, 2]}})
    prior = rotvec_T([0, 0, 0], [0, 0, 500])
    w = Window(prior, cfg)
    far = rotvec_T([0.2, 0, 0], [30, 0, 500])
    R, t = w.clamp(far[:3, :3], far[:3, 3])
    assert abs(t[0] - 6.25) < 1e-9                       # 5 mm * 1.25 margin
    assert pose_error(rotvec_T([0, 0, 0], t) @ np.eye(4), prior)[1] < 1e-6
    assert np.degrees(np.linalg.norm(__import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_matrix(R).as_rotvec())) < 2.6
    assert w.inside(prior)


def test_estimate_tau_follows_depth_quantisation():
    rng = np.random.default_rng(0)
    base = 800 + np.add.outer(np.linspace(0, 40, 200), np.linspace(0, 40, 200))   # tilted plane
    fine = (base + rng.normal(0, 0.05, base.shape)).astype(np.float32)
    coarse = (np.round(base / 3.0) * 3.0).astype(np.float32)                      # 3 mm steps
    assert s0_preprocess.estimate_tau(fine) == 2.0                               # clipped to the floor
    assert 2.5 <= s0_preprocess.estimate_tau(coarse) <= 4.0


def test_match_with_image_uses_edges_without_hurting(mesh, rng):
    model = build_model("bracket", mesh)
    gt = random_pose(rng)
    rgb, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60, seed=1)
    prior = perturb(gt, rng, 5, 2)
    m = PoseMatcher(profile="balanced")
    m.cfg = m.cfg.with_overrides({"prior": {"tolerance_t_mm": [8] * 3, "tolerance_r_deg": [5] * 3}})
    m.set_model(model)
    res = m.match("bracket", depth, DEFAULT_K, prior, rgb=rgb)
    et, er = pose_error(res.T_cam_obj, gt)
    assert et < 2.5 and er < 2.5
    assert "edge_ratio" in res.metrics and res.metrics["edge_ratio"] > 0.5
    assert res.metrics["tau_mm"] >= 2.0
