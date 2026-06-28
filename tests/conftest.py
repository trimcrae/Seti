import pytest

from seti.config import load_config
from seti.sample import make_sample


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample():
    return make_sample(seed=7)
