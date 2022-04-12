from copy import deepcopy
from typing import Dict, List, Type

from ray.util.metrics import Metric
from river.base import Regressor, MultiOutputMixin
from river.base.typing import RegTarget
from river.compose import Pipeline
from river.metrics import RMSE, Metrics
from river.stats import Mean


class NRMSE(RMSE):
    """
    Normalized RMSE class (wrapper around river's RMSE class)
    """

    def __init__(self):
        super().__init__()
        self._observed_mean = Mean()

    def update(self, y_true, y_pred, sample_weight=1.0):
        self._observed_mean.update(x=y_true)
        return super().update(y_true, y_pred, sample_weight)

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self._observed_mean.revert(x=y_true)
        return super().revert(y_true, y_pred, sample_weight)

    def get(self):
        observed_mean = self._observed_mean.get()
        observed_mean = observed_mean if observed_mean != 0 else 1

        return super().get() / observed_mean


def metrics_to_dict(metrics_object: Metrics) -> Dict[str, float]:
    """
    Transform a river metrics object to a dictionary

    Parameters
    ----------
    metrics_object:
        The metrics object containing the metrics

    Returns
    -------
    The metrics values as a dict
    """

    metrics = {}
    for metric in metrics_object:
        # Apparently the only way to gather the name of the metric,
        #   see: https://github.com/online-ml/river/blob/master/river/metrics/base.py#L59
        metrics[metric.__class__.__name__] = metric.get()

    return metrics


class MultiOutputRegressor(Regressor, MultiOutputMixin):
    """
    Class representing a regressor with multiple output, one regressor per output
    """

    def __init__(self, models: list):
        """
        Constructor for the MultiOutputRegressor class

        Parameters
        ----------
        models:
            The list of regressor, one regressor per output
        """
        super().__init__()
        self.models = models

    def __len__(self) -> int:
        """
        Override len function, will return the number of outputs
        """
        return len(self.models)

    def learn_one(self, x: dict, y: RegTarget, **kwargs) -> "Regressor":
        """
        Learn one data point
        """
        for model in self.models:
            model.learn_one(x, y, **kwargs)

        return self

    def predict_one(self, x: dict) -> List[RegTarget]:
        """
        Predict one data point
        """
        prediction_list = []

        for model in self.models:
            prediction_list.append(model.predict_one(x))

        return prediction_list


class MultiOutputPipeline:
    """
    Wrapper around a pipeline with a multi output regressor or classifier
    """

    def __init__(self, pipeline: Pipeline, metric_class: Type[Metric]):
        """
        Constructor for the MultiOutputPipeline class

        Parameters
        ----------
        pipeline:
            The pipeline to wrap
        metric_class:
            One metric will be created per output in order to use the best one
        """
        self.pipeline = pipeline

        # Get the output size of the pipeline, if zero means that it's not a multi output
        _, last_element = next(reversed(pipeline.steps.items()))
        self.output_size = len(last_element) if hasattr(last_element, "__len__") else 0

        # Instantiate one metric per output/model
        self.metric_list = [metric_class() for _ in range(self.output_size)]

    def learn_one(self, x: dict, y, learn_unsupervised=False, **params):
        """
        Learn one data point and update metrics
        """
        prediction_list = []
        if self.output_size > 0:
            prediction_list = self.pipeline.predict_one(x, learn_unsupervised=False)

        self.pipeline.learn_one(x, y, learn_unsupervised, **params)

        for output_index in range(self.output_size):
            self.metric_list[output_index].update(y, prediction_list[output_index])

    def predict_one(self, x: dict, learn_unsupervised=True):
        """
        Predict one data point, use the internal metrics to give the best output
        """
        prediction_list = self.pipeline.predict_one(x, learn_unsupervised)
        if type(prediction_list) is not list:
            return prediction_list

        best_model_index = -1
        best_model_metric = 0

        for output_index in range(self.output_size):
            output_metric = self.metric_list[output_index].get()
            if best_model_index == -1 or output_metric < best_model_metric:
                best_model_index = output_index
                best_model_metric = output_metric

        return prediction_list[best_model_index]
