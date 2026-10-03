import json

import cv2
import numpy as np

from snappose.cli import main
from snappose.synth import DEFAULT_K, make_bracket, random_pose, render_rgbd


def test_demo_runs(tmp_path, capsys):
    assert main(["demo", "-p", "fast", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "demo_overlay.png").exists()


def test_onboard_and_match_files(tmp_path, capsys):
    mesh = make_bracket()
    cad = tmp_path / "b.stl"
    mesh.export(cad)
    rng = np.random.default_rng(7)
    gt = random_pose(rng)
    rgb, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60)
    cv2.imwrite(str(tmp_path / "d.png"), depth.astype(np.uint16))
    cv2.imwrite(str(tmp_path / "c.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (tmp_path / "k.json").write_text(json.dumps({"K": DEFAULT_K.tolist()}))
    (tmp_path / "prior.json").write_text(json.dumps({"T_cam_obj": gt.tolist()}))

    assert main(["onboard", "--object", "b", "--cad", str(cad), "--store", str(tmp_path / "o")]) == 0
    rc = main(["match", "--object", "b", "--store", str(tmp_path / "o"), "--depth", str(tmp_path / "d.png"),
               "--rgb", str(tmp_path / "c.png"), "--intrinsics", str(tmp_path / "k.json"),
               "--prior", str(tmp_path / "prior.json"), "--out", str(tmp_path / "r.json"),
               "--overlay", str(tmp_path / "ov.png"), "-p", "balanced", "-s", "s3.refine_iters=3"])
    res = json.loads((tmp_path / "r.json").read_text())
    assert rc in (0, 1) and res["status"] in ("OK", "UNSICHER")
    assert np.linalg.norm(np.array(res["translation_mm"]) - gt[:3, 3]) < 2.0
    assert (tmp_path / "ov.png").exists()
