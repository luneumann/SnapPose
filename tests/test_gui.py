import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from snappose.gui.engine import Engine
from snappose.gui.server import Handler
from snappose.synth import DEFAULT_K, make_bracket, random_pose, render_rgbd


@pytest.fixture
def server(tmp_path):
    Handler.engine = Engine(tmp_path / "objs", tmp_path / "nobop")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{Handler.port}", Handler.engine
    srv.shutdown()


def call(base, path, body=None, raw=None, headers=None, method=None):
    h = {"X-Requested-With": "snappose"}
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body or {}).encode() if body is not None or method == "POST" else None)
    if data is not None and raw is None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=h, method=method or ("POST" if data is not None else "GET"))
    return urllib.request.urlopen(req)


def wait_idle(base, timeout=60):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = json.loads(call(base, "/api/status").read())
        if not s["busy"]:
            return s
        time.sleep(0.1)
    raise TimeoutError


def test_page_and_status(server):
    base, _ = server
    assert b"SnapPose" in call(base, "/").read()
    s = json.loads(call(base, "/api/status").read())
    assert s["objects"][0]["id"] == "demo-bracket" and s["frame"] is None


def test_demo_match_and_overlay(server):
    base, _ = server
    s = json.loads(call(base, "/api/frame/demo", {"seed": 3}).read())
    assert s["frame"]["has_gt"] and s["prior"]
    call(base, "/api/match", {})
    s = wait_idle(base)
    assert s["error"] == "" and s["result"]["status"] in ("OK", "UNSICHER")
    assert s["result"]["error_vs_gt"]["t_mm"] < 3
    jpg = call(base, "/api/image.jpg?view=rgb&layers=prior,result,gt,edges").read()
    assert jpg[:2] == b"\xff\xd8"
    assert call(base, "/api/image.jpg?view=depth&layers=result").read()[:2] == b"\xff\xd8"
    assert json.loads(call(base, "/api/result.json").read())["status"] == s["result"]["status"]


def test_rejects_foreign_host_and_missing_header(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/match", {}, headers={"X-Requested-With": ""})
    assert e.value.code == 403
    req = urllib.request.Request(base + "/api/status", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


def test_errors_are_reported_not_crashing(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/match", {})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/object/onboard?name=../x&ext=stl&scale=1", raw=b"x", headers={"Content-Type": "application/octet-stream"})
    assert e.value.code == 400


def test_upload_cad_and_own_files(server, tmp_path):
    base, eng = server
    mesh = make_bracket()
    f = tmp_path / "b.stl"
    mesh.export(f)
    call(base, "/api/object/onboard?name=teil_1&ext=stl&scale=1", raw=f.read_bytes(), headers={"Content-Type": "application/octet-stream"})
    s = json.loads(call(base, "/api/status").read())
    assert s["object"] == "teil_1" and any(o["id"] == "teil_1" for o in s["objects"])

    gt = random_pose(np.random.default_rng(5))
    rgb, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60)
    _, dpng = cv2.imencode(".png", depth.astype(np.uint16))
    call(base, "/api/frame/upload?kind=depth", raw=dpng.tobytes(), headers={"Content-Type": "application/octet-stream"})
    call(base, "/api/frame/camera", {"fx": 600, "fy": 600, "cx": 320, "cy": 240, "depth_scale": 1})
    euler = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_matrix(gt[:3, :3]).as_euler("xyz", degrees=True)
    call(base, "/api/prior/set", {"t": (gt[:3, 3] + 3).tolist(), "euler_deg": (euler + 1).tolist()})
    call(base, "/api/match", {})
    s = wait_idle(base)
    assert s["error"] == "" and np.linalg.norm(np.array(s["result"]["translation_mm"]) - gt[:3, 3]) < 3
    call(base, "/api/object/delete", {"id": "teil_1"})
    assert not any(o["id"] == "teil_1" for o in json.loads(call(base, "/api/status").read())["objects"])
