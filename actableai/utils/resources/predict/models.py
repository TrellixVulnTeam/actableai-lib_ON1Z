from numbers import Number
from typing import Optional

from river.compose import SelectType
from river.dummy import StatisticRegressor
from river.ensemble import AdaptiveRandomForestRegressor
from river.metrics import RMSE
from river.preprocessing import AdaptiveStandardScaler, OneHotEncoder, Normalizer
from river.stats import Mean

from actableai.tasks import TaskType
from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.river import MultiOutputRegressor, MultiOutputPipeline


def _create_default_pipeline() -> MultiOutputPipeline:
    """
    Create a basic pipeline that can be used to predict resources
    """
    numerical_pre_processing = SelectType(Number) | AdaptiveStandardScaler()
    categorical_pre_processing = SelectType(str) | OneHotEncoder(sparse=True)
    pre_processing = (
        numerical_pre_processing + categorical_pre_processing
    ) | Normalizer()

    regressor = MultiOutputRegressor(
        [
            AdaptiveRandomForestRegressor(leaf_prediction="adaptive"),
            StatisticRegressor(Mean()),
        ]
    )

    return MultiOutputPipeline(pre_processing | regressor, RMSE)


def create_pipeline(
    resource_predicted: ResourcePredictorType, task: TaskType
) -> Optional[MultiOutputPipeline]:
    """
    Create a pipeline (model) for resource prediction

    Parameters
    ----------
    resource_predicted:
        The resource to predict
    task:
        The task to predict for

    Returns
    -------
    The river pipeline to use for prediction or None if not available
    """
    pipeline = None

    if resource_predicted == ResourcePredictorType.MAX_MEMORY:
        if task == TaskType.CLASSIFICATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CLASSIFICATION_TRAIN:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CLUSTERING:
            pipeline = _create_default_pipeline()
        elif task == TaskType.DEC_ANCHOR_CLUSTERING:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CORRELATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.DATA_IMPUTATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.FORECAST:
            pipeline = _create_default_pipeline()
        elif task == TaskType.REGRESSION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.REGRESSION_TRAIN:
            pipeline = _create_default_pipeline()
        elif task == TaskType.BAYESIAN_REGRESSION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.SENTIMENT_ANALYSIS:
            pipeline = _create_default_pipeline()
    elif resource_predicted == ResourcePredictorType.MAX_GPU_MEMORY:
        if task == TaskType.CLASSIFICATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CLASSIFICATION_TRAIN:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CLUSTERING:
            pipeline = _create_default_pipeline()
        elif task == TaskType.DEC_ANCHOR_CLUSTERING:
            pipeline = _create_default_pipeline()
        elif task == TaskType.CORRELATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.DATA_IMPUTATION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.FORECAST:
            pipeline = _create_default_pipeline()
        elif task == TaskType.REGRESSION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.REGRESSION_TRAIN:
            pipeline = _create_default_pipeline()
        elif task == TaskType.BAYESIAN_REGRESSION:
            pipeline = _create_default_pipeline()
        elif task == TaskType.SENTIMENT_ANALYSIS:
            pipeline = _create_default_pipeline()

    return pipeline
