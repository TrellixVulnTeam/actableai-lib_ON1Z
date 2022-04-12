import logging
from typing import List, Tuple

import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    NVMLError,
    nvmlShutdown,
)

from actableai.utils.resources.profile import ResourceProfilerType
from actableai.utils.resources.profile.base import ResourceProfiler


class GPUMemoryProfiler(ResourceProfiler):
    """
    GPU Memory profiler
    """

    def __init__(self, resource_profiled: ResourceProfilerType):
        """
        Constructor for the GPUMemoryProfiler class

        Parameters
        ----------
        resource_profiled:
            The resource to profile, ignored in this case
        """
        super().__init__(resource_profiled)

        self.device_count = 0

        try:
            nvmlInit()
            self.device_count = nvmlDeviceGetCount()
        except NVMLError:
            logging.warning("Error while initializing the NVML library")

    def _device_handle_generator(self):
        """
        Generator to iterate through the nvidia devices
        """
        for device_index in range(self.device_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(device_index)
            except NVMLError:
                logging.warning(
                    "Error while trying to fetch nvidia Device, skipping ..."
                )
                continue

            yield handle

    def __call__(
        self, process_list: List[psutil.Process]
    ) -> List[Tuple[int, ResourceProfilerType, float]]:
        """
        Function called to profile processes

        Parameters
        ----------
        process_list:
            The list of process objects to profile


        Returns
        -------
        List containing the resource profiled, each element of the list contain three elements:
        - The pid of the process
        - The resource profiled
        - The profiled value
        """
        process_pid_set = set({process.pid for process in process_list})

        profiled_gpu_memory = []

        for device_handle in self._device_handle_generator():
            process_info_list = []

            try:
                process_info_list = nvmlDeviceGetComputeRunningProcesses(device_handle)
            except NVMLError:
                logging.warning(
                    "Error while trying to profile GPU memory, your OS is probably not compatible with this feature"
                )

            for process_info in process_info_list:
                if process_info.pid in process_pid_set:
                    profiled_gpu_memory.append(
                        (
                            process_info.pid,
                            ResourceProfilerType.GPU_MEMORY,
                            process_info.usedGpuMemory,
                        )
                    )

        return profiled_gpu_memory

    def shutdown(self):
        """
        Function called after using the profiler, in that case shutdowns the NVML library
        """
        try:
            nvmlShutdown()
        except NVMLError:
            logging.warning("Error while trying to shutdown NVML")
