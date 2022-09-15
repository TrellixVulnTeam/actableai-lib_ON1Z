from typing import Dict

from actableai.models.aai_predictor import AAIModel, AAITimeSeriesForecasterModel


class AAIModelInference:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(cls, num_replicas, ray_options, s3_bucket, s3_prefix=""):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            num_replicas=num_replicas,
            ray_actor_options=ray_options,
            init_args=(s3_bucket, s3_prefix),
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

    def __init__(self, s3_bucket, s3_prefix=""):
        """
        TODO write documentation
        """
        import boto3

        self.task_models = {}
        self.s3_prefix = s3_prefix
        self.s3_bucket_name = s3_bucket

        s3 = boto3.resource("s3")
        self.bucket = s3.Bucket(self.s3_bucket_name)

    def _get_model_path(self, task_id):
        """
        TODO write documentation
        """
        import os

        return os.path.join(self.s3_prefix, task_id, "model.p")

    def _load_model(self, task_id):
        """
        TODO write documentation
        """
        import pickle
        from botocore.exceptions import ClientError

        path = self._get_model_path(task_id)
        obj = self.bucket.Object(path)
        try:
            raw_model = obj.get()["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            else:
                raise

        self.task_models[task_id] = pickle.loads(raw_model)
        return True

    def _get_model(self, task_id, raise_error=True):
        """
        TODO write documentation
        """
        from actableai.exceptions.models import InvalidTaskIdError

        if task_id not in self.task_models and not self._load_model(task_id):
            if raise_error:
                raise InvalidTaskIdError()
            else:
                return None

        return self.task_models[task_id]

    def predict(
        self,
        task_id,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
    ):
        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )

        task_model = self._get_model(task_id)

        # We used to pickle TabularPredictor. This check is for legacy
        if isinstance(task_model, AAIModel):
            if isinstance(task_model, AAITabularModel):
                pred = self._predict(
                    task_id,
                    task_model.predictor,
                    df,
                    return_probabilities,
                    probability_threshold,
                    positive_label,
                )
                # Intervention effect part
                if isinstance(task_model, AAITabularModelInterventional) and (
                    f"intervened_{task_model.intervened_column}" in df
                    or f"expected_{task_model.predictor.label}" in df
                ):
                    pred["intervention"] = task_model.intervention_effect(df, pred)
                return pred
            if isinstance(task_model, AAITimeSeriesForecasterModel):
                return task_model.model.predict(df=df)

        # Run legacy task_model directly
        return self._predict(
            task_id,
            task_model,
            df,
            return_probabilities,
            probability_threshold,
            positive_label,
        )

    def _predict(
        self,
        task_id,
        task_model,
        df,
        return_probabilities=False,
        probability_threshold=0.5,
        positive_label=None,
    ) -> Dict:
        """
        TODO write documentation
        """
        from actableai.exceptions.models import InvalidPositiveLabelError

        result = {}

        df_proba = self.predict_proba(task_id, df)

        class_labels = list(df_proba.columns)

        # Regression
        if len(class_labels) <= 1:
            result["prediction"] = df_proba
            return result

        # Multiclass
        if len(class_labels) > 2:
            prediction = df_proba.idxmax(axis="columns").to_frame(name=task_model.label)
            if return_probabilities:
                result["df_proba"] = df_proba
            result["prediction"] = prediction
            return result

        # Binary
        if positive_label is None:
            positive_label = class_labels[1]

        if positive_label not in class_labels:
            raise InvalidPositiveLabelError()

        negative_label = list(set(class_labels).difference({positive_label}))[0]

        df_true_label = df_proba[positive_label] >= probability_threshold
        df_true_label = (
            df_true_label.astype(str)
            .map({"True": positive_label, "False": negative_label})
            .to_frame(name=task_model.label)
        )

        if return_probabilities:
            result["df_proba"] = df_proba
        result["prediction"] = df_true_label
        return result

    def predict_proba(self, task_id, df):
        from actableai.models.aai_predictor import AAITabularModel

        task_model = self._get_model(task_id)

        if isinstance(task_model, AAITabularModel):
            return self._predict_proba(task_model.predictor, df)
        return self._predict_proba(task_model, df)

    def _predict_proba(self, task_model, df):
        """
        TODO write documentation
        """
        import pandas as pd
        from actableai.exceptions.models import MissingFeaturesError

        # FIXME this considers that the model we have is an AutoGluon which will not
        #  be the case in the future
        feature_generator = task_model._learner.feature_generator
        first_feature_links = feature_generator.get_feature_links()

        missing_features = list(
            set(first_feature_links.keys()).difference(set(df.columns))
        )

        if len(missing_features) > 0:
            raise MissingFeaturesError(missing_features=missing_features)

        df_proba = task_model.predict_proba(df, as_multiclass=True)
        if isinstance(df_proba, pd.Series):
            df_proba = df_proba.to_frame(name=task_model.label)
        return df_proba

    def get_metadata(self, task_id):
        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )

        task_model = self._get_model(task_id)
        if isinstance(task_model, AAITabularModel):
            metadata = self._get_metadata(task_model.predictor)

            if (
                isinstance(task_model, AAITabularModelInterventional)
                and task_model.causal_model is not None
                and task_model.intervened_column is not None
            ):
                metadata["intervened_column"] = task_model.intervened_column
                metadata["discrete_treatment"] = task_model.discrete_treatment
            return metadata
        return self._get_metadata(task_model)

    def _get_metadata(self, task_model):
        """
        TODO write documentation
        """

        # FIXME this considers that the model we have is an AutoGluon which will not
        #  be the case in the future
        feature_generator = task_model._learner.feature_generator
        first_feature_links = feature_generator.get_feature_links()
        features = list(first_feature_links.keys())

        problem_type = task_model.problem_type

        metadata = {
            "features": features,
            "problem_type": problem_type,
            "class_labels": None,
            "prediction_target": task_model.label,
        }

        if problem_type == "multiclass" or problem_type == "binary":
            class_labels = task_model.class_labels
            metadata["class_labels"] = class_labels

        return metadata

    def is_model_available(self, task_id):
        """
        TODO write documentation
        """
        import boto3

        if self.is_model_loaded(task_id):
            return True

        s3_client = boto3.client("s3")
        object_list = s3_client.list_objects_v2(
            Bucket=self.s3_bucket_name, Prefix=self._get_model_path(task_id)
        )

        return "Contents" in object_list and len(object_list["Contents"]) > 0

    def is_model_loaded(self, task_id):
        """
        TODO write documentation
        """
        return task_id in self.task_models

    def unload_model(self, task_id):
        """
        TODO write documentation
        """
        if self.is_model_loaded(task_id):
            del self.task_models[task_id]
