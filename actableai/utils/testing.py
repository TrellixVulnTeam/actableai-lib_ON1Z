from copy import deepcopy

import psutil
import ray


def unittest_hyperparameters():
    return {"RF": {}}


def init_ray(**kwargs):
    num_cpus = psutil.cpu_count()
    ray.init(num_cpus=num_cpus, namespace="aai", **kwargs)
