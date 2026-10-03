import pytest

from snappose.config import Config
from snappose.registry import REFINERS


def test_profiles_and_overrides():
    c = Config.load(profile="fast")
    assert c.s3.refine_iters == 2 and c.time_budget_ms == 300
    c = Config.load({"profile": "balanced", "overrides": {"s3": {"refine_iters": 3}}}, time_budget_ms=123)
    assert c.s3.refine_iters == 3 and c.time_budget_ms == 123
    assert c.s5.icp_iters == 30            # untouched profile value


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        Config.load({"overrides": {"s3": {"nope": 1}}})
    with pytest.raises(ValueError):
        Config.load(profile="turbo")


def test_with_overrides_switches_profile():
    c = Config.load(profile="fast").with_overrides({"s3": {"refine_iters": 5}}, profile="precise")
    assert c.profile == "precise" and c.s3.refine_iters == 5 and c.s5.icp_iters == 60


def test_license_mode_blocks_noncommercial():
    REFINERS.register("fp_test", license="non-commercial")(lambda *a, **k: None)
    with pytest.raises(PermissionError):
        REFINERS.get("fp_test", "commercial")
    assert REFINERS.get("fp_test", "research")
