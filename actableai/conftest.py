import pytest
import ray

import numpy as np

from actableai.utils.testing import init_ray as _init_ray


@pytest.fixture(scope="function")
def np_rng(random_state=0):
    yield np.random.default_rng(random_state)


@pytest.fixture(scope="function")
def init_ray():
    if not ray.is_initialized():
        _init_ray()

        yield None

        ray.shutdown()
    else:
        yield None
