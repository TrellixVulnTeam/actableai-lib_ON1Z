from typing import List, Optional, Dict, Tuple, Any

import pandas as pd

from mxnet.context import Context

from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.simple_model import AAITimeSeriesSimpleModel
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.models.params import BaseParams


class AAITimeSeriesMultiTargetModel(AAITimeSeriesBaseModel):
    """Multi-Target Time Series Model, can be used for univariate and multivariate
    forecasting. It will internally use one AAITimeSeriesSimpleModel for each
    target, using the other target as features for every model.
    """

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
        """AAITimeSeriesMultiTargetModel Constructor.

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
        super().__init__(
            target_columns,
            prediction_length,
            freq,
            group_dict,
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

        self.predictor_dict = {}

        self.shift_target_columns_dict = self._get_shift_target_columns()
        self.shift_target_columns = list(self.shift_target_columns_dict.values())

    def _get_shift_target_columns(self) -> Dict[str, str]:
        """Create look-up table (dictionary) to associate target columns with their
            shifted corresponding columns.

        Returns:
            The dictionary.
        """
        return {
            target_column: f"_{target_column}_shift"
            for target_column in self.target_columns
        }

    def _pre_process_data(
        self, df_dict: Dict[Tuple[Any], pd.DataFrame], keep_future: bool = True
    ) -> Dict[Tuple[Any], pd.DataFrame]:
        """Pre-process data to add the shifted target as features.

        Args:
            df_dict: Dictionary containing the time series for each group.
            keep_future: If False the future (shifted) values are trimmed out.

        Returns:
            New dictionary containing the time series for each group (along with the
                features).
        """
        if len(self.target_columns) <= 1:
            return df_dict

        df_dict_new = {}
        for group in df_dict.keys():
            # Create the shifted dataframe
            df_shift = df_dict[group]
            if keep_future and self.has_dynamic_features:
                df_shift = df_shift.iloc[: -self.prediction_length]

            df_shift = df_shift[self.target_columns].shift(
                self.prediction_length, freq=self.freq
            )
            # Rename columns
            df_shift = df_shift.rename(columns=self.shift_target_columns_dict)

            if not keep_future:
                df_shift = df_shift.loc[df_shift.index.isin(df_dict[group].index)]

            # Add new features
            df_dict_new[group] = pd.concat([df_dict[group], df_shift], axis=1)

            df_dict_new[group] = df_dict_new[group].iloc[self.prediction_length :]

        return df_dict_new

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
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        total_time = 0

        for target_column in self.target_columns:
            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            self.predictor_dict[target_column] = AAITimeSeriesSimpleModel(
                target_columns=[target_column],
                prediction_length=self.prediction_length,
                freq=self.freq,
                group_dict=self.group_dict,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            )
            target_total_time = self.predictor_dict[target_column].fit(
                df_dict=df_dict_clean,
                model_params=model_params,
                mx_ctx=mx_ctx,
                loss=loss,
                trials=trials,
                max_concurrent=max_concurrent,
                use_ray=use_ray,
                tune_samples=tune_samples,
                sampling_method=sampling_method,
                random_state=random_state,
                ray_tune_kwargs=ray_tune_kwargs,
                verbose=verbose,
                fit_full=fit_full,
            )

            total_time += target_total_time

        return total_time

    def refit(self, df_dict: Dict[Tuple[Any], pd.DataFrame]):
        """Fit previously tuned model.

        Args:
            df_dict: Dictionary containing the time series for each group.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()
            self.predictor_dict[target_column].refit(df_dict_clean)

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
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}
        df_item_metrics_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}
        df_agg_metrics = pd.DataFrame()

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()

            (
                df_target_predictions_dict,
                df_target_item_metrics_dict,
                df_target_agg_metrics,
            ) = self.predictor_dict[target_column].score(
                df_dict_clean,
                num_samples=num_samples,
                quantiles=quantiles,
                num_workers=num_workers,
            )

            for group in df_dict_clean.keys():
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )
                df_item_metrics_dict[group] = pd.concat(
                    [df_item_metrics_dict[group], df_target_item_metrics_dict[group]],
                    ignore_index=True,
                )
            df_agg_metrics = pd.concat(
                [df_agg_metrics, df_target_agg_metrics], ignore_index=True
            )

        return df_predictions_dict, df_item_metrics_dict, df_agg_metrics

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
        df_dict_clean = self._pre_process_data(df_dict, keep_future=True)

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()

            df_target_predictions_dict = self.predictor_dict[target_column].predict(
                df_dict_clean, quantiles=quantiles
            )

            for group in df_dict_clean.keys():
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )

        return df_predictions_dict
