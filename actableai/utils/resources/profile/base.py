from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple

import psutil

from actableai.utils.resources.profile import ResourceProfilerType


class ResourceProfiler(ABC):
    """
    Abstract class representing a resource profiler
    """

    def __init__(self, resource_profiled: ResourceProfilerType):
        """
        Constructor for the ResourceProfiler class

        Parameters
        ----------
        resource_profiled:
            The resource to profile, used because some Resource Profiler implementation might profile different
            resources at the same time (MemoryProfiler for instance)
        """
        self.resource_profiled = resource_profiled

    @abstractmethod
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
        raise NotImplementedError

    def shutdown(self):
        """
        Function called after using the profiler, it is assumed that the __call__ function cannot be called once
        shutdown is called
        """
        pass
