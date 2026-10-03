import numpy as np
import pytest

from snappose.onboarding import build_model
from snappose.synth import make_bracket


@pytest.fixture(scope="session")
def mesh():
    return make_bracket()


@pytest.fixture(scope="session")
def model(mesh):
    return build_model("bracket", mesh)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
