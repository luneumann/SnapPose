import json

import numpy as np

from snappose.geometry import Symmetry, inv_T, pose_error, rot_angle_deg, rotation_about, rotvec_T


def test_inverse_roundtrip():
    T = rotvec_T([0.3, -0.2, 0.9], [10, 20, 500])
    assert np.allclose(T @ inv_T(T), np.eye(4), atol=1e-9)


def test_cyclic_symmetry_canonicalize(tmp_path):
    spec = {"symmetries": [{"type": "cyclic", "axis": [0, 0, 1], "order": 4}]}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(spec))
    sym = Symmetry.from_json(f)
    assert len(sym.discrete) == 4
    ref = rotvec_T([0.1, 0, 0], [0, 0, 500])
    T = ref @ rotation_about([0, 0, 1], np.pi, [0, 0, 0])      # 180 deg about z: physically identical
    out = sym.canonicalize(T, ref)
    assert rot_angle_deg(out[:3, :3], ref[:3, :3]) < 1e-6
    assert pose_error(T, ref, sym)[1] < 1e-6


def test_continuous_symmetry():
    sym = Symmetry.from_spec([{"type": "continuous", "axis": [0, 0, 1]}])
    ref = rotvec_T([0, 0, 0], [0, 0, 400])
    T = ref @ rotation_about([0, 0, 1], 1.234, [0, 0, 0])
    assert rot_angle_deg(sym.canonicalize(T, ref)[:3, :3], ref[:3, :3]) < 0.6
