from enum import IntFlag, unique, auto
from queue import Queue
from typing import Callable, Tuple, Any, Dict, Type, Optional

import pandas as pd


@unique
class ResourceProfilerType(IntFlag):
    """
    Enum representing the different resource that can be profiled
    """

    RSS_MEMORY = auto()
    VMS_MEMORY = auto()
    SHARED_MEMORY = auto()
    USS_MEMORY = auto()
    PSS_MEMORY = auto()
    SWAP_MEMORY = auto()
    GPU_MEMORY = auto()
    MEMORY = (
        RSS_MEMORY | VMS_MEMORY | SHARED_MEMORY | USS_MEMORY | PSS_MEMORY | SWAP_MEMORY
    )


class ResourceProfilerResults:
    """
    Class representing the results returned by the profiling of a function
    """

    def __init__(self, df_profiling_results: pd.DataFrame):
        """
        Constructor for the ResourceProfilerResults class

        Parameters
        ----------
        df_profiling_results:
            The pandas DataFrame representing the profiling results
        """
        self.df_profiling_results = df_profiling_results

    def get_max_profiled(self, resource_profiled: ResourceProfilerType) -> float:
        """
        Return the maximum profiled value of a specific resource

        Parameters
        ----------
        resource_profiled:
            The resource type to get the maximum value from

        Returns
        -------
        The maximum profiled value, 0 if not profiled
        """

        df_mask = (self.df_profiling_results["resource_type"] & resource_profiled) != 0
        df_resource_profiling_results = self.df_profiling_results[df_mask]

        if len(df_resource_profiling_results) == 0:
            return 0

        df_resource_profiling_results = df_resource_profiling_results[
            ["timestamp", "value"]
        ]
        df_resource_profiling_results = df_resource_profiling_results.groupby(
            "timestamp"
        ).sum()
        max_profiled = df_resource_profiling_results.max()

        if max_profiled.isna().all():
            return 0

        return max_profiled.iat[0]


from actableai.utils.resources.profile.gpu_memory import GPUMemoryProfiler
from actableai.utils.resources.profile.memory import MemoryProfiler
from actableai.utils.resources.profile.base import ResourceProfiler

_resource_profilers: Dict[ResourceProfilerType, Type[ResourceProfiler]] = {
    ResourceProfilerType.MEMORY: MemoryProfiler,
    ResourceProfilerType.GPU_MEMORY: GPUMemoryProfiler,
}


def _run_profiler(
    resource_profiled: ResourceProfilerType,
    include_children: bool,
    pid: int,
    queue: Queue,
):
    """
    Run the profiler for a specific pid

    Stops when an element is pushed in the queue and returns the results in the same queue

    Parameters
    ----------
    resource_profiled:
        The resources to profile
    include_children:
        If True will also returns the profiled resources for the children of the pid
    pid:
        The pid to profile
    queue:
        Queue used to transfer information
    """
    from time import sleep, time
    import psutil
    import pandas as pd

    main_process = psutil.Process(pid)

    # Instantiate the profilers for the different resources
    profilers = []
    for resource, resource_profiler in _resource_profilers.items():
        resource &= resource_profiled
        if resource != 0:
            profilers.append(resource_profiler(resource))

    df_profiling_results = pd.DataFrame(
        columns=["timestamp", "pid", "resource_type", "value"]
    )

    # Run profiler while there is nothing in the queue
    while True:
        timestamp = time()

        process_list = [main_process]
        if include_children:
            process_list += main_process.children(recursive=True)

        for profiler in profilers:
            profiler_results = profiler(process_list)
            for pid, resource_type, value in profiler_results:
                df_profiling_results = df_profiling_results.append(
                    {
                        "timestamp": timestamp,
                        "pid": pid,
                        "resource_type": resource_type,
                        "value": value,
                    },
                    ignore_index=True,
                )

        if not queue.empty():
            break
        sleep(0.1)

    for profiler in profilers:
        profiler.shutdown()

    queue.get()
    queue.put(df_profiling_results)


def profile_function(
    resource_profiled: ResourceProfilerType,
    include_children: bool,
    function: Callable,
    *args,
    **kwargs
) -> Tuple[ResourceProfilerResults, Any]:
    """
    Profile a function

    Will call the function and start a new thread that will profile different resources of the function

    Parameters
    ----------
    resource_profiled:
        The resources to profile
    include_children:
        If True will also profile the children (if any) created when calling the function
    function:
        The function to profile
    args:
        The arguments to call the function with
    kwargs:
        The named arguments to call the function with

    Returns
    -------
    dataframe:
        DataFrame containing the profiling information of the function over time (in bytes)
    result:
        The return value of the function
    """
    import os
    from queue import Queue
    from threading import Thread
    import pandas as pd

    main_process_pid = os.getpid()

    # Start the profiler in another thread
    queue = Queue()
    thread = Thread(
        target=_run_profiler,
        args=(resource_profiled, include_children, main_process_pid, queue),
    )
    thread.start()

    # Run the function to profile
    result = function(*args, **kwargs)

    # When the function is finished add an element in the queue to notify the profiler thread
    queue.put(True)
    # Wait for the profiler to finish
    thread.join()

    # Fetch the profiling data from the queue
    df_profiling_results = queue.get()

    if len(df_profiling_results) > 0:
        # Process the profiling data
        df_profiling_results["timestamp"] = pd.to_datetime(
            df_profiling_results["timestamp"], unit="s"
        )
        df_profiling_results["pid"] = df_profiling_results["pid"].astype(int)
        df_profiling_results["resource_type"] = df_profiling_results[
            "resource_type"
        ].astype(int)

        # We do that to avoid having duplicates (for instance GPU memory profiler will add one row per device)
        df_profiling_results = df_profiling_results.groupby(
            ["timestamp", "pid", "resource_type"], as_index=False
        ).sum()

        df_profiling_results["child_process"] = True
        df_profiling_results.loc[
            df_profiling_results["pid"] == main_process_pid, "child_process"
        ] = False

    return ResourceProfilerResults(df_profiling_results), result
