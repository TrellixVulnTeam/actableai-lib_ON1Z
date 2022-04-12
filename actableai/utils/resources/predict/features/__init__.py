from typing import Callable

from actableai.tasks import TaskType
from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.resources.predict.features.forecast import (
    ForecastFeaturesExtractor,
)
from actableai.utils.resources.predict.features.method import MethodFeaturesExtractor
from actableai.utils.resources.predict.features.regression import (
    RegressionFeaturesExtractor,
    RegressionTrainFeaturesExtractor,
)
from actableai.utils.resources.predict.features.bayesian_regression import (
    BayesianRegressionFeaturesExtractor,
)
from actableai.utils.resources.predict.features.classification import (
    ClassificationFeaturesExtractor,
    ClassificationTrainFeaturesExtractor,
)
from actableai.utils.resources.predict.features.clustering import (
    ClusteringFeaturesExtractor,
    DECAnchorClusteringFeaturesExtractor,
)
from actableai.utils.resources.predict.features.correlation import (
    CorrelationFeaturesExtractor,
)
from actableai.utils.resources.predict.features.data_imputation import (
    DataImputationFeaturesExtractor,
)
from actableai.utils.resources.predict.features.sentiment_analysis import (
    SentimentAnalysisFeaturesExtractor,
)

_features_extractors = {
    TaskType.CLASSIFICATION: ClassificationFeaturesExtractor,
    TaskType.CLASSIFICATION_TRAIN: ClassificationTrainFeaturesExtractor,
    TaskType.CLUSTERING: ClusteringFeaturesExtractor,
    TaskType.DEC_ANCHOR_CLUSTERING: DECAnchorClusteringFeaturesExtractor,
    TaskType.CORRELATION: CorrelationFeaturesExtractor,
    TaskType.DATA_IMPUTATION: DataImputationFeaturesExtractor,
    TaskType.FORECAST: ForecastFeaturesExtractor,
    TaskType.REGRESSION: RegressionFeaturesExtractor,
    TaskType.REGRESSION_TRAIN: RegressionTrainFeaturesExtractor,
    TaskType.BAYESIAN_REGRESSION: BayesianRegressionFeaturesExtractor,
    TaskType.SENTIMENT_ANALYSIS: SentimentAnalysisFeaturesExtractor,
}


def get_features_extractor(
    resource_predicted: ResourcePredictorType, task: TaskType, function: Callable
) -> MethodFeaturesExtractor:
    """
    Get a features extractor object corresponding to the predicted resources and the task

    Parameters
    ----------
    resource_predicted:
        The resource to predict
    task:
        The task to predict for
    function:
        The function to extract the features from (used to get the signature)

    Returns
    -------
    Features extractor class
    """
    features_extractor = _features_extractors.get(task, None)
    if features_extractor is None:
        raise NotImplementedError(
            f"The features extractor for this task ({task}) is not implemented"
        )
    return features_extractor(resource_predicted=resource_predicted, function=function)


def extract_features(
    resource_predicted: ResourcePredictorType,
    task: TaskType,
    function: Callable,
    *args,
    **kwargs,
) -> dict:
    """
    Extract features from function

    Parameters
    ----------
    resource_predicted:
        The resource to predict
    task:
        The task to predict for
    function:
        The function to extract the features from (used to get the signature)
    args:
        The arguments passed to the function
    kwargs:
        The named arguments passed to the function

    Returns
    -------
    The extracted features
    """
    return get_features_extractor(resource_predicted, task, function)(*args, **kwargs)
