import pytest
import torch
import logging

from actableai.utils.resources.profile import ResourceProfilerType, profile_function
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    NVMLError,
    nvmlShutdown,
)


def _function(sleep=False, n_child=0, child_sleep=False, cpu=True, gpu=False):
    import time
    from multiprocessing import Process

    if gpu:
        import torch

    def _child_function():
        if gpu:
            big_cuda_tensor_2 = torch.cuda.FloatTensor(100, 100).fill_(0)
        if cpu:
            big_array_2 = [1] * 1000

        if child_sleep:
            time.sleep(0.2)

    if gpu:
        big_cuda_tensor = torch.cuda.FloatTensor(1000, 1000).fill_(0)
    if cpu:
        big_array = [1] * 10000

    process_list = [Process(target=_child_function) for _ in range(n_child)]
    for process in process_list:
        process.start()

    if sleep:
        time.sleep(0.2)

    for process in process_list:
        process.join()

    return True


@pytest.fixture(scope="session")
def is_gpu_available():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


@pytest.fixture(scope="session")
def is_nvml_available(is_gpu_available):
    if not is_gpu_available:
        return False

    try:
        nvmlInit()

        device_count = nvmlDeviceGetCount()
        for device_index in range(device_count):
            device_handle = nvmlDeviceGetHandleByIndex(device_index)
            nvmlDeviceGetComputeRunningProcesses(device_handle)

        nvmlShutdown()

        return True
    except NVMLError:
        return False


class TestResourcesProfilers:
    @pytest.mark.parametrize("resource_profiled_int", range(1 << 7))
    def test_resource_profiler_types(self, resource_profiled_int):
        resource_profiled = ResourceProfilerType(resource_profiled_int)
        profiling_results, _ = profile_function(resource_profiled, False, _function)
        df_profiling_results = profiling_results.df_profiling_results

        if len(df_profiling_results) > 0:
            assert (
                df_profiling_results["resource_type"].unique().sum() & resource_profiled
            ) != 0

    def test_return_value(self):
        _, results = profile_function(ResourceProfilerType(0), False, _function)
        assert results

    def test_return_columns(self):
        profiling_results, _ = profile_function(
            ResourceProfilerType.MEMORY, False, _function, sleep=True
        )
        df_profiling_results = profiling_results.df_profiling_results
        columns = df_profiling_results.columns

        assert "timestamp" in columns
        assert "pid" in columns
        assert "resource_type" in columns
        assert "child_process" in columns

    def test_memory_profiling_max(self):
        profiling_results, _ = profile_function(
            ResourceProfilerType.MEMORY, False, _function
        )
        assert profiling_results.get_max_profiled(ResourceProfilerType.MEMORY) > 0

    def test_memory_profiling_len(self):
        profiling_results, _ = profile_function(
            ResourceProfilerType.MEMORY, False, _function, sleep=True
        )
        assert len(profiling_results.df_profiling_results) > 0

    @pytest.mark.parametrize("n_child", [1, 2, 3])
    def test_memory_profiling_child(self, n_child):
        profiling_results, _ = profile_function(
            ResourceProfilerType.MEMORY,
            True,
            _function,
            n_child=n_child,
            child_sleep=True,
        )
        df_profiling_results = profiling_results.df_profiling_results

        assert len(df_profiling_results["pid"].unique()) >= n_child + 1
        assert len(df_profiling_results["child_process"].unique()) == 2

    def test_gpu_memory_profiling_max(self, is_nvml_available):
        if is_nvml_available:
            profiling_results, _ = profile_function(
                ResourceProfilerType.GPU_MEMORY, False, _function, cpu=False, gpu=True
            )
            assert profiling_results.get_max_profiled(ResourceProfilerType.MEMORY) > 0
        else:
            logging.warning(
                "Skipping test `test_gpu_memory_profiling_max`, no GPU available or NVML not available"
            )

    def test_gpu_memory_profiling_len(self, is_nvml_available):
        if is_nvml_available:
            profiling_results, _ = profile_function(
                ResourceProfilerType.GPU_MEMORY,
                False,
                _function,
                sleep=True,
                cpu=False,
                gpu=True,
            )
            assert len(profiling_results.df_profiling_results) > 0
        else:
            logging.warning(
                "Skipping test `test_gpu_memory_profiling_len`, no GPU available or NVML not available"
            )

    @pytest.mark.parametrize("n_child", [1, 2, 3])
    def test_gpu_memory_profiling_child(self, n_child, is_nvml_available):
        if is_nvml_available:
            profiling_results, _ = profile_function(
                ResourceProfilerType.GPU_MEMORY,
                True,
                _function,
                sleep=False,
                n_child=n_child,
                child_sleep=True,
                cpu=False,
                gpu=True,
            )
            df_profiling_results = profiling_results.df_profiling_results

            assert len(df_profiling_results["pid"].unique()) >= n_child + 1
            assert len(df_profiling_results["child_process"].unique()) == 2
        else:
            logging.warning(
                "Skipping test `test_gpu_memory_profiling_child`, no GPU available or NVML not available"
            )
