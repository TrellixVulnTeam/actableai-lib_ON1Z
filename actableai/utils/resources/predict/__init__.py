from enum import Enum, unique


@unique
class ResourcePredictorType(str, Enum):
    """
    Enum representing the different type of resource that are predictable
    """

    MAX_MEMORY = "max_memory"
    MAX_GPU_MEMORY = "max_gpu_memory"
