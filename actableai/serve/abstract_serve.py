import logging

logger = logging.getLogger(__name__)


class AbstractRayDeployment:
    """
    Class to implement for RayDeployment.
    When implementing, you should implement the __init__ method and then any method you
    want to expose.
    When using the class, you should call the deploy method to deploy the class on Ray
    with the desired number of replicas, Ray options and the same arguments as the
    __init__ method.
    """

    @classmethod
    def deploy(cls, num_replicas, ray_options, *args):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            num_replicas=num_replicas,
            ray_actor_options=ray_options,
            init_args=args,
        ).deploy()

    @classmethod
    def get_handle(cls):
        """
        TODO write documentation
        """
        return cls.get_deployment().get_handle()

    @classmethod
    def get_deployment(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.get_deployment(cls.__name__)

    def __init__(self) -> None:
        """
        TODO write documentation
        """
        raise NotImplementedError()
