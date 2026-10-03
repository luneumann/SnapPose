import json

import cv2
import numpy as np

from snappose import eval_bop
from snappose.synth import DEFAULT_K, make_bracket, random_pose, render_rgbd


def _make_bop(root, n=4):
    mesh = make_bracket()
    (root / "models").mkdir(parents=True)
    mesh.export(root / "models" / "obj_000001.ply")
    (root / "models" / "models_info.json").write_text(json.dumps({"1": {"diameter": 100.0}}))
    sd = root / "test" / "000001"
    (sd / "depth").mkdir(parents=True)
    rng = np.random.default_rng(3)
    cams, gts, infos = {}, {}, {}
    for i in range(n):
        gt = random_pose(rng)
        _, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60, seed=i)
        cv2.imwrite(str(sd / "depth" / f"{i:06d}.png"), depth.astype(np.uint16))
        cams[str(i)] = {"cam_K": DEFAULT_K.flatten().tolist(), "depth_scale": 1.0}
        gts[str(i)] = [{"cam_R_m2c": gt[:3, :3].flatten().tolist(), "cam_t_m2c": gt[:3, 3].tolist(), "obj_id": 1}]
        infos[str(i)] = [{"visib_fract": 1.0}]
    for name, d in (("scene_camera", cams), ("scene_gt", gts), ("scene_gt_info", infos)):
        (sd / f"{name}.json").write_text(json.dumps(d))


def test_bop_adapter_on_synthetic_dataset(tmp_path):
    _make_bop(tmp_path)
    res = eval_bop.run(str(tmp_path), profile="balanced", per_object=4, prior_t=4, prior_r=2)
    o = res["overall"]
    assert o["n"] == 4
    assert o["success"]["5mm/5deg"] == 1.0
    assert "1" in res["per_object"]
    assert "ALL" in eval_bop.format_report(res)
