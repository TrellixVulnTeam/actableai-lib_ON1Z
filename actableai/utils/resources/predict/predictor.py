import ray

from typing import Dict, Union, Optional, List

from actableai.tasks import TaskType
from actableai.utils.resources.predict import ResourcePredictorType


class ResourcesPredictorsActor:
    """
    Class used to store different resources predictors
    Is is assumed that this class will be used as a ray Actor
    """

    _version = "2.0"

    @classmethod
    def get_actor(
        cls,
        s3_resources_predictors_bucket: str = None,
        s3_resources_predictors_prefix: str = None,
        resources_predictors_data_path: str = None,
        random_state: int = None,
        backup_state_probability: float = None,
    ) -> ray.actor.ActorHandle:
        """
        Get the ray Actor, will create it if it does not exist already

        Parameters
        ----------
        s3_resources_predictors_bucket:
            The AWS S3 bucket name where the resources predictors data are stored
        s3_resources_predictors_prefix:
            The AWS S3 prefix to use when fetching the resources predictors data
        resources_predictors_data_path:
            The path where the resources predictors data are located, this will be used only if the S3 AWS parameters
            are None
        random_state:
            The random state to use for the random generator, if None will use the current timestamp
        backup_state_probability:
            The probability to start a backup of the resources predictors data when calling the `add_data` method,
            default: 0.1 (10%)

        Returns
        -------
        The Actor Handle
        """
        resources_predictors_actor = None

        # Load or create the resources predictors actor
        try:
            resources_predictors_actor = ray.get_actor(name=cls.__name__)
        except ValueError:
            resources_predictors_actor = (
                ray.remote(cls)
                .options(name=cls.__name__, lifetime="detached")
                .remote(
                    s3_resources_predictors_bucket=s3_resources_predictors_bucket,
                    s3_resources_predictors_prefix=s3_resources_predictors_prefix,
                    resources_predictors_data_path=resources_predictors_data_path,
                    random_state=random_state,
                    backup_state_probability=backup_state_probability,
                )
            )

        return resources_predictors_actor

    def __init__(
        self,
        s3_resources_predictors_bucket: str = None,
        s3_resources_predictors_prefix: str = None,
        resources_predictors_data_path: str = None,
        random_state: int = None,
        backup_state_probability: float = None,
        override_models: bool = False,
    ):
        """
        Constructor for the Resources Predictors Actor

        Parameters
        ----------
        s3_resources_predictors_bucket:
            The AWS S3 bucket name where the resources predictors data are stored
        s3_resources_predictors_prefix:
            The AWS S3 prefix to use when fetching the resources predictors data
        resources_predictors_data_path:
            The path where the resources predictors data are located, this will be used only if the S3 AWS parameters
            are None
        random_state:
            The random state to use for the random generator, if None will use the current timestamp
        backup_state_probability:
            The probability to start a backup of the resources predictors data when calling the `add_data` method,
            default: 0.1 (10%)
        override_models:
            If True will not load the models from AWS or filesystem, it will create new models
        """
        import json
        import pickle
        import boto3
        from queue import Queue
        import numpy as np
        from river.metrics import Metrics, RMSE, R2
        from actableai.utils.resources.predict.models import create_pipeline
        from actableai.utils.river import NRMSE

        if s3_resources_predictors_prefix is None:
            s3_resources_predictors_prefix = ""
        if resources_predictors_data_path is None:
            resources_predictors_data_path = "/tmp/resources_predictors_data_path/"
        if backup_state_probability is None:
            backup_state_probability = 0.1

        # FIXME create real data interface to store data
        self.s3_resources_predictors_prefix = s3_resources_predictors_prefix
        self.resources_predictors_data_path = resources_predictors_data_path
        self.bucket = None
        if s3_resources_predictors_bucket is not None:
            s3 = boto3.resource("s3")
            self.bucket = s3.Bucket(s3_resources_predictors_bucket)

        # Migrate if version file does not correspond to current version
        if self.bucket is not None and not override_models:
            version_file = self._read_file(self._get_version_path())
            if (
                version_file is None
                or json.loads(version_file).get("version", None) != self._version
            ):
                self.migrate(
                    s3_resources_predictors_bucket, [s3_resources_predictors_prefix]
                )

        # Load all the models, when a model is not available it is represented by None
        # This is done to avoid keeping track of which models are implemented
        self.models_data = {}
        for resource_predicted in ResourcePredictorType:
            self.models_data[resource_predicted] = {}

            for task in TaskType:
                self.models_data[resource_predicted][task] = None

                # Create or read the model
                model = self._read_file(
                    self._get_model_path(resource_predicted, task), True
                )
                if model is None or override_models:
                    model = create_pipeline(resource_predicted, task)
                else:
                    model = pickle.loads(model)

                # This means that there is no model for this combination of resource_predicted, task
                if model is None:
                    continue

                # Create or read the model metrics
                model_metrics = self._read_file(
                    self._get_model_metrics_path(resource_predicted, task), True
                )
                if model_metrics is None or override_models:
                    model_metrics = Metrics([RMSE(), NRMSE(), R2()])
                else:
                    model_metrics = pickle.loads(model_metrics)

                self.models_data[resource_predicted][task] = {
                    "model": model,
                    "model_metrics": model_metrics,
                    "training_data_queue": Queue(),
                }

        self.random_generator = np.random.default_rng(random_state)
        self.backup_state_probability = backup_state_probability

    async def get_model_metrics(
        self, resource_predicted: ResourcePredictorType, task: TaskType
    ) -> Dict[str, float]:
        """
        Get the metrics of one specific model

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for

        Returns
        -------
        Dictionary with the metrics names as a key and the metrics values as value
        """
        from actableai.utils.river import metrics_to_dict

        model_data = self.models_data[resource_predicted][task]
        if model_data is None:
            raise NotImplementedError(
                f"Predictor not implemented ({resource_predicted}, {task})"
            )

        return metrics_to_dict(model_data["model_metrics"])

    async def predict(
        self, resource_predicted: ResourcePredictorType, task: TaskType, features: dict
    ) -> float:
        """
        Make a prediction for a resource usage

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        features:
            The features to feed the model with

        Returns
        -------
        The predicted value
        """
        model_data = self.models_data[resource_predicted][task]
        if model_data is None:
            raise NotImplementedError(
                f"Predictor not implemented ({resource_predicted}, {task})"
            )

        return model_data["model"].predict_one(features, learn_unsupervised=False)

    def add_data(
        self,
        resource_predicted: ResourcePredictorType,
        task: TaskType,
        features: dict,
        target: float,
        prediction: float = None,
        timestamp: int = None,
        full_features: dict = None,
    ):
        """
        Add data to the prediction model (train)

        Will backup the resources predictors data to either AWS S3 or the filesystem with the pre-defined probability

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        features:
            The features to train the model with
        target:
            The observed value to train with (ground truth)
        prediction:
            The predicted value with these features before training (if None will be re-computed)
        timestamp:
            The timestamp used for backup-ing the data, current timestamp by default
        full_features:
            All the collected features, these are not used to train the model, just to keep track of everything to
            eventually add new features in the future
        """
        from time import time_ns
        from actableai.utils.river import metrics_to_dict

        model_data = self.models_data[resource_predicted][task]
        if model_data is None:
            raise NotImplementedError(
                f"Predictor not implemented ({resource_predicted}, {task})"
            )

        if prediction is None:
            prediction = model_data["model"].predict_one(
                features, learn_unsupervised=False
            )
        model_data["model"].learn_one(features, target, learn_unsupervised=True)

        model_data["model_metrics"].update(target, prediction)

        if timestamp is None:
            timestamp = time_ns()

        training_data_dict = {
            "timestamp": str(timestamp),
            "X": features,
            "y": target,
            "y_pred": prediction,
            "metrics": metrics_to_dict(model_data["model_metrics"]),
        }

        if full_features is not None:
            training_data_dict["X_full"] = full_features

        model_data["training_data_queue"].put(training_data_dict)

        # Save the state with a pre-defined probability
        if self.random_generator.uniform(0, 1) < self.backup_state_probability:
            self._backup_state()

    @staticmethod
    def _get_backup_folder_path(backup_timestamp: int):
        """
        Get the path where the backup folder is

        Parameters
        ----------
        backup_timestamp:
            Timestamp representing the creation of the backup folder

        Returns
        -------
        The path
        """
        return f"backup_{backup_timestamp}"

    @classmethod
    def _get_model_data_folder_path(
        cls,
        resource_predicted: ResourcePredictorType,
        task: TaskType,
        backup_timestamp: int = None,
    ) -> str:
        """
        Get the path where the resources predictors data are stored

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        backup_timestamp:
            If set will be used to write into a backup folder, None by default

        Returns
        -------
        The path
        """
        import os

        path = ""
        if backup_timestamp is not None:
            path = os.path.join(path, cls._get_backup_folder_path(backup_timestamp))
        return os.path.join(path, resource_predicted, task)

    @classmethod
    def _get_model_path(
        cls,
        resource_predicted: ResourcePredictorType,
        task: TaskType,
        backup_timestamp: int = None,
    ) -> str:
        """
        Get the path where the model object is stored

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        backup_timestamp:
            If set will be used to write into a backup folder, None by default

        Returns
        -------
        The path
        """
        import os

        return os.path.join(
            cls._get_model_data_folder_path(resource_predicted, task, backup_timestamp),
            "model.p",
        )

    @classmethod
    def _get_model_metrics_path(
        cls,
        resource_predicted: ResourcePredictorType,
        task: TaskType,
        backup_timestamp: int = None,
    ) -> str:
        """
        Get the path where the metrics object is stored

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        backup_timestamp:
            If set will be used to write into a backup folder, None by default

        Returns
        -------
        The path
        """
        import os

        return os.path.join(
            cls._get_model_data_folder_path(resource_predicted, task, backup_timestamp),
            "model_metrics.p",
        )

    @classmethod
    def _get_training_data_path(
        cls,
        resource_predicted: ResourcePredictorType,
        task: TaskType,
        timestamp: str,
        backup_timestamp: int = None,
    ) -> str:
        """
        Get the path where the training data are stored

        Parameters
        ----------
        resource_predicted:
            The resource to predict
        task:
            The task to predict for
        timestamp:
            Timestamp representing the time when the training data were used
        backup_timestamp:
            If set will be used to write into a backup folder, None by default

        Returns
        -------
        The path
        """
        import os

        return os.path.join(
            cls._get_model_data_folder_path(resource_predicted, task, backup_timestamp),
            "training_data",
            f"{timestamp}.json",
        )

    @classmethod
    def _get_version_path(cls, backup_timestamp: int = None):
        """
        Get the path where the version file is stored

        Parameters
        ----------
        backup_timestamp:
            If set will be used to write into a backup folder, None by default

        Returns
        -------
        The path
        """
        import os

        path = ""
        if backup_timestamp is not None:
            path = os.path.join(path, cls._get_backup_folder_path(backup_timestamp))
        return os.path.join(path, "version.json")

    def _write_file(self, path: str, file_content: Union[str, bytes]):
        """
        Write a file to a specific path, either on AWS S3 or in the filesystem

        Parameters
        ----------
        path:
            The path to save the file to
        file_content:
            The content of the file to save
        """
        import os

        if self.bucket is None:
            path = os.path.join(self.resources_predictors_data_path, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            file_mode = "w"
            if isinstance(file_content, bytes):
                file_mode += "b"

            with open(path, file_mode) as file:
                file.write(file_content)
        else:
            path = os.path.join(self.s3_resources_predictors_prefix, path)

            self.bucket.put_object(Key=path, Body=file_content)

    def _read_file(
        self, path: str, byte_file: bool = False
    ) -> Optional[Union[str, bytes]]:
        """
        Read a file from a specific path in either AWS S3 or the filesystem

        Parameters
        ----------
        path:
            The path the read from
        byte_file:
            True if the file to read is to be read as bytes

        Returns
        -------
        The file content, returns None if the file does not exist
        """
        import os
        from pathlib import Path
        from botocore.exceptions import ClientError

        # Read from file system
        if self.bucket is None:
            path = os.path.join(self.resources_predictors_data_path, path)

            if not Path(path).exists():
                return None

            file_mode = "r"
            if byte_file:
                file_mode += "b"
            with open(path, file_mode) as model_file:
                return model_file.read()

        # Read from AWS S3
        path = os.path.join(self.s3_resources_predictors_prefix, path)
        obj = self.bucket.Object(path)

        try:
            return obj.get()["Body"].read()
        except ClientError as exception:
            if exception.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def _backup_state(self):
        """
        Backup the resources predictors data in either AWS S3 or the filesystem
        """
        import json
        import pickle

        for resource_predicted, resource_models_data in self.models_data.items():
            for task, model_data in resource_models_data.items():
                if model_data is None:
                    continue

                model_path = self._get_model_path(resource_predicted, task)
                model_metrics_path = self._get_model_metrics_path(
                    resource_predicted, task
                )

                self._write_file(model_path, pickle.dumps(model_data["model"]))
                self._write_file(
                    model_metrics_path, pickle.dumps(model_data["model_metrics"])
                )

                while not model_data["training_data_queue"].empty():
                    training_data = model_data["training_data_queue"].get()
                    training_data_path = self._get_training_data_path(
                        resource_predicted, task, training_data["timestamp"]
                    )
                    self._write_file(training_data_path, json.dumps(training_data))

    @classmethod
    def migrate(cls, s3_bucket: str, s3_prefix_list: List[str]):
        """
        Update models and backup old training data


        Here is how to call this function:
        ```python
        from actableai.utils.resources.predict.predictor import ResourcesPredictorsActor
        ResourcesPredictorsActor.migrate("actable-ai-resources-predictors", ["dev"])
        ```

        In this example we are migrating the dev folder, but you can replace it by prod or even update both at the same
        time.


        How it works?
        This function will create a backup folder and will move every models, model metrics, and training data already
        existing in this backup folder.
        Once this is done it will create brand new pipelines and will train each of these pipelines using the training
        data collected previously and uploaded in AWS.
        Finally it will save these new pipelines overriding the previous ones (which are backed up).

        Parameters
        ----------
        s3_bucket:
            The AWS S3 bucket name where the resources predictors data are stored
        s3_prefix_list:
            The list of AWS S3 prefix to use when fetching the resources predictors data
        """
        import boto3
        import json
        from time import time_ns

        s3_client = boto3.client("s3")
        s3_resource = boto3.resource("s3")
        bucket = s3_resource.Bucket(s3_bucket)

        backup_timestamp = time_ns()

        print(f"Start migration, backup timestamp: {backup_timestamp}")

        for s3_prefix in s3_prefix_list:
            print("----------------------------------------")
            print(f"Start migration for {s3_prefix}")

            resources_predictors = cls(
                s3_resources_predictors_bucket=s3_bucket,
                s3_resources_predictors_prefix=s3_prefix,
                backup_state_probability=0.0,
                override_models=True,
            )

            # Backup version file
            version_file = resources_predictors._read_file(
                resources_predictors._get_version_path()
            )
            if version_file is not None:
                print(f"Backup version file")
                resources_predictors._write_file(
                    resources_predictors._get_version_path(
                        backup_timestamp=backup_timestamp
                    ),
                    version_file,
                )

            # Create new version file
            new_version = {
                "version": str(cls._version),
                "created_timestamp": backup_timestamp,
            }
            resources_predictors._write_file(
                resources_predictors._get_version_path(), json.dumps(new_version)
            )

            for resource_predicted in ResourcePredictorType:
                for task in TaskType:
                    # Backup model
                    model = resources_predictors._read_file(
                        resources_predictors._get_model_path(resource_predicted, task),
                        True,
                    )
                    if model is not None:
                        print(
                            f"Backup model, resource: {resource_predicted}, task: {task}"
                        )
                        model_backup_path = resources_predictors._get_model_path(
                            resource_predicted, task, backup_timestamp=backup_timestamp
                        )
                        resources_predictors._write_file(model_backup_path, model)

                    # Backup model metrics
                    model_metrics = resources_predictors._read_file(
                        resources_predictors._get_model_metrics_path(
                            resource_predicted, task
                        ),
                        True,
                    )
                    if model_metrics is not None:
                        print(
                            f"Backup model metrics, resource: {resource_predicted}, task: {task}"
                        )
                        model_metrics_backup_path = (
                            resources_predictors._get_model_metrics_path(
                                resource_predicted,
                                task,
                                backup_timestamp=backup_timestamp,
                            )
                        )
                        resources_predictors._write_file(
                            model_metrics_backup_path, model_metrics
                        )

                    # Backup training data and train new model
                    response = s3_client.list_objects_v2(
                        Bucket=s3_bucket,
                        Prefix=f"{s3_prefix}/{resource_predicted}/{task}/training_data/",
                    )
                    if response is not None and "Contents" in response:
                        training_data_dict = {}

                        print(
                            "Start backup training data"
                            + f", len: {len(response['Contents'])}"
                            + f", resource: {resource_predicted}"
                            + f", task: {task}"
                        )

                        # Backup training data
                        for obj in response["Contents"]:
                            bucket_obj = bucket.Object(obj["Key"])
                            training_data = json.load(bucket_obj.get()["Body"])

                            training_data_backup_path = (
                                resources_predictors._get_training_data_path(
                                    resource_predicted,
                                    task,
                                    training_data["timestamp"],
                                    backup_timestamp=backup_timestamp,
                                )
                            )
                            resources_predictors._write_file(
                                training_data_backup_path, json.dumps(training_data)
                            )

                            # Add training data to dict
                            training_data_dict[
                                training_data["timestamp"]
                            ] = training_data

                        print(
                            "Start training new model"
                            + f", len: {len(response['Contents'])}"
                            + f", resource: {resource_predicted}"
                            + f", task: {task}"
                        )
                        # Train new model with training data
                        for timestamp in sorted(training_data_dict.keys()):
                            training_data = training_data_dict[timestamp]

                            resources_predictors.add_data(
                                resource_predicted,
                                task,
                                training_data["X"],
                                training_data["y"],
                                timestamp=training_data["timestamp"],
                                full_features=training_data.get("X_full", None),
                            )

            print(f"Save new models and new training data")
            # Save new models
            resources_predictors._backup_state()
