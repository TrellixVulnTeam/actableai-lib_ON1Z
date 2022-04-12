from itertools import product
from unittest.mock import patch

import pytest
import ray

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask
from actableai.utils.resources.profile import ResourceProfilerType
from actableai.utils.testing import init_ray


@pytest.fixture(scope="class")
def init_ray_class():
    ray_shutdown = False
    if not ray.is_initialized():
        init_ray(resources={"gpu_memory": 10000000}, num_gpus=1)
        ray_shutdown = True

    yield None

    if ray_shutdown:
        ray.shutdown()


@pytest.fixture(scope="class")
def resources_predictors_actor(init_ray_class):
    class MockResourcesPredictorsActor:
        async def get_model_metrics(self, *args, **kwargs):
            return {"NRMSE": 0.5}

        async def predict(self, *args, **kwargs):
            return 1000.0

        def add_data(self, *args, **kwargs):
            pass

    actor = (
        ray.remote(MockResourcesPredictorsActor)
        .options(name="MockResourcesPredictorsActor", lifetime="detached")
        .remote()
    )

    yield actor


@pytest.fixture(scope="class")
def aai_simple_task():
    def mock_profile_function(
        resource_profiled, include_children, function, *args, **kwargs
    ):
        class MockResourceProfilerResults:
            def get_max_profiled(self, resource_profiled):
                return 1.0

        kwargs["execution_type"] = resource_profiled
        return MockResourceProfilerResults(), function(*args, **kwargs)

    class AAITestTask(AAITask):
        @AAITask.run_with_ray_remote(TaskType.CAUSAL_INFERENCE)
        def run(self, arg_1, arg_2, arg_3=3, arg_4=None, execution_type="basic"):
            return arg_1, arg_2, arg_3, arg_4, execution_type

    with patch(
        "actableai.utils.resources.predict.features.extract_features",
        return_value=({"feat1": 1.0}, None),
    ):
        with patch(
            "actableai.utils.resources.profile.profile_function",
            new=mock_profile_function,
        ):
            yield AAITestTask


@pytest.mark.parametrize(
    "use_ray,optimize_memory_allocation,optimize_gpu_memory_allocation,collect_memory_usage,collect_gpu_memory_usage",
    [
        *[(True, *combination) for combination in product([True, False], repeat=4)],
        (False, False, False, False, False),
    ],
)
@pytest.mark.parametrize("optimize_memory_allocation_nrmse_threshold", [0.0, 1.0])
@pytest.mark.parametrize("optimize_gpu_memory_allocation_nrmse_threshold", [0.0, 1.0])
class TestBaseTask:
    def test_simple(
        self,
        aai_simple_task,
        use_ray,
        optimize_memory_allocation,
        collect_memory_usage,
        optimize_memory_allocation_nrmse_threshold,
        optimize_gpu_memory_allocation,
        collect_gpu_memory_usage,
        optimize_gpu_memory_allocation_nrmse_threshold,
        resources_predictors_actor,
    ):
        task = aai_simple_task(
            use_ray=use_ray,
            optimize_memory_allocation=optimize_memory_allocation,
            collect_memory_usage=collect_memory_usage,
            optimize_memory_allocation_nrmse_threshold=optimize_memory_allocation_nrmse_threshold,
            optimize_gpu_memory_allocation=optimize_gpu_memory_allocation,
            collect_gpu_memory_usage=collect_gpu_memory_usage,
            optimize_gpu_memory_allocation_nrmse_threshold=optimize_gpu_memory_allocation_nrmse_threshold,
            resources_predictors_actor=resources_predictors_actor,
        )
        arg_1, arg_2, arg_3, arg_4, execution_type = task.run(1, 2, arg_4=4)

        assert arg_1 == 1
        assert arg_2 == 2
        assert arg_3 == 3
        assert arg_4 == 4

        if not use_ray:
            assert execution_type == "basic"
        elif collect_memory_usage and collect_gpu_memory_usage:
            assert (
                execution_type
                == ResourceProfilerType.RSS_MEMORY
                | ResourceProfilerType.SWAP_MEMORY
                | ResourceProfilerType.GPU_MEMORY
            )
        elif collect_memory_usage:
            assert (
                execution_type
                == ResourceProfilerType.RSS_MEMORY | ResourceProfilerType.SWAP_MEMORY
            )
        elif collect_gpu_memory_usage:
            assert execution_type == ResourceProfilerType.GPU_MEMORY
        else:
            assert execution_type == "basic"
