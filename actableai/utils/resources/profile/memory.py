from queue import Queue
from typing import List, Tuple

import psutil

from actableai.utils.resources.profile import ResourceProfilerType
from actableai.utils.resources.profile.base import ResourceProfiler


class MemoryProfiler(ResourceProfiler):
    """
    Memory profiler
    profiles the following resources:
    - RSS_MEMORY
    - VMS_MEMORY
    - SHARED_MEMORY
    - USS_MEMORY
    - PSS_MEMORY
    - SWAP_MEMORY
    """

    _profiling_info_to_resource_type = {
        "rss": ResourceProfilerType.RSS_MEMORY,
        "vms": ResourceProfilerType.VMS_MEMORY,
        "shared": ResourceProfilerType.SHARED_MEMORY,
        "uss": ResourceProfilerType.USS_MEMORY,
        "pss": ResourceProfilerType.PSS_MEMORY,
        "swap": ResourceProfilerType.SWAP_MEMORY,
    }

    def __init__(self, resource_profiled: ResourceProfilerType):
        """
        Constructor for the MemoryProfiler class

        Parameters
        ----------
        resource_profiled:
            The resource to profile
        """
        super().__init__(resource_profiled)

        self.need_full_info = (
            (
                ResourceProfilerType.USS_MEMORY
                | ResourceProfilerType.PSS_MEMORY
                | ResourceProfilerType.SWAP_MEMORY
            )
            & resource_profiled
        ) != 0

        self.profiling_info_list = [
            info_name
            for info_name, resource_type in self._profiling_info_to_resource_type.items()
            if resource_type in resource_profiled
        ]

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

        profiled_memory = []

        for process in process_list:
            memory_info = {}
            try:
                if self.need_full_info:
                    memory_info = process.memory_full_info()
                else:
                    memory_info = process.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            for info_name in self.profiling_info_list:
                if hasattr(memory_info, info_name):
                    info = getattr(memory_info, info_name)
                    profiled_memory.append(
                        (
                            process.pid,
                            self._profiling_info_to_resource_type[info_name],
                            info,
                        )
                    )

        return profiled_memory
