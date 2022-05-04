from typing import List, Dict, Tuple, Any, Optional

from abc import ABC, abstractmethod

import pandas as pd

from mxnet.context import Context

from actableai.timeseries.util import find_gluonts_freq
from actableai.timeseries.models.params import BaseParams


class AAITimeSeriesBaseModel(ABC):
    """Time Series Model interface."""

    def __init__(
        self,
        target_columns: List[str],
        prediction_length: int,
        freq: str,
        group_dict: Optional[Dict[Tuple[Any], int]] = None,
        real_static_feature_dict: Optional[Dict[Tuple[Any], List[float]]] = None,
        cat_static_feature_dict: Optional[Dict[Tuple[Any], List[Any]]] = None,
        real_dynamic_feature_columns: Optional[List[str]] = None,
        cat_dynamic_feature_columns: Optional[List[str]] = None,
    ):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            target_columns: List of columns to forecast.
            prediction_length: Length of the prediction to forecast.
            freq: Frequency of the time series.
            group_dict: Dictionary containing the unique label for each group.
            real_static_feature_dict: Dictionary containing a list of real features for
                each group.
            cat_static_feature_dict: Dictionary containing a list of categorical
                features for each group.
            real_dynamic_feature_columns: List of columns containing real features.
            cat_dynamic_feature_columns: List of columns containing categorical
                features.
        """
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.freq = freq

        self.group_dict = group_dict
        self.real_static_feature_dict = real_static_feature_dict
        self.cat_static_feature_dict = cat_static_feature_dict
        self.real_dynamic_feature_columns = real_dynamic_feature_columns
        self.cat_dynamic_feature_columns = cat_dynamic_feature_columns

        if self.group_dict is None:
            self.group_dict = {}
        if self.real_static_feature_dict is None:
            self.real_static_feature_dict = {}
        if self.cat_static_feature_dict is None:
            self.cat_static_feature_dict = {}
        if self.real_dynamic_feature_columns is None:
            self.real_dynamic_feature_columns = []
        if self.cat_dynamic_feature_columns is None:
            self.cat_dynamic_feature_columns = []

        self.freq_gluon = find_gluonts_freq(self.freq)

        self.has_dynamic_features = (
            len(self.real_dynamic_feature_columns)
            + len(self.cat_dynamic_feature_columns)
            > 0
        )

    @abstractmethod
    def fit(
        self,
        df_dict: Dict[Tuple[Any], pd.DataFrame],
        model_params: List[BaseParams],
        mx_ctx: Context,
        *,
        loss: str = "mean_wQuantileLoss",
        trials: int = 3,
        max_concurrent: Optional[int] = None,
        use_ray: bool = True,
        tune_samples: int = 3,
        sampling_method: str = "random",
        random_state: Optional[int] = None,
        ray_tune_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fit_full: bool = True,
    ) -> float:
        """Tune and fit the model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            model_params: List of models parameters to run the tuning search on.
            mx_ctx: mxnet context.
            loss: Loss to minimize when tuning.
            trials: Number of trials for hyperparameter search.
            max_concurrent: Maximum number of concurrent ray task.
            use_ray: If True ray will be used for hyperparameter tuning.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].
            random_state: Random state to use for reproducibility.
            ray_tune_kwargs: Named parameters to pass to ray's `tune` function.
            verbose: Verbose level.
            fit_full: If True the model will be fit after tuning using all the data
                (tuning data).

        Returns:
            Total time spent for tuning.
        """
        raise NotImplementedError()

    @abstractmethod
    def refit(self, df_dict: Dict[Tuple[Any], pd.DataFrame]):
        """Fit previously tuned model.

        Args:
            df_dict: Dictionary containing the time series for each group.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        raise NotImplementedError()

    @abstractmethod
    def score(
        self,
        df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        num_samples: int = 100,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        num_workers: Optional[int] = None,
    ) -> Tuple[
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], pd.DataFrame],
        pd.DataFrame,
    ]:
        """Evaluate model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            num_samples: Number of dataset samples to use for evaluation
            quantiles: List of quantiles to use for evaluation.
            num_workers: Maximum number of workers to use, if None no parallelization
                will be done.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            - Dictionary containing the predicted time series for each group.
            - Dictionary containing the metrics for each target for each group.
            - Dataframe containing the aggregated metrics for each target.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self,
        df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        """Make a prediction using the model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Dictionary containing the predicted time series for each group.
        """
        raise NotImplementedError()
