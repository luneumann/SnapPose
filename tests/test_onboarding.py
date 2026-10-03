import numpy as np
import pytest

from snappose.onboarding import load_model, onboard
from snappose.synth import make_bracket


def test_onboard_cache_roundtrip(tmp_path):
    cad = tmp_path / "part.stl"
    make_bracket().export(cad)
    m1 = onboard("part", cad, store=tmp_path / "objs")
    mtime = (tmp_path / "objs" / "part" / "manifest.json").stat().st_mtime_ns
    m2 = onboard("part", cad, store=tmp_path / "objs")             # cache hit
    assert (tmp_path / "objs" / "part" / "manifest.json").stat().st_mtime_ns == mtime
    assert np.allclose(m1.points, m2.points)
    assert np.allclose(load_model("part", tmp_path / "objs").extents, [80, 50, 35], atol=0.5)


def test_scale_changes_hash(tmp_path):
    cad = tmp_path / "part.stl"
    make_bracket().export(cad)
    a = onboard("p", cad, store=tmp_path)
    b = onboard("p", cad, store=tmp_path, scale=2.0)
    assert np.allclose(b.extents, 2 * a.extents, atol=0.5)


def test_missing_object(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("nope", tmp_path)
