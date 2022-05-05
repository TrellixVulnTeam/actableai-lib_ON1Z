from typing import List, Union, Tuple, Any, Dict, Optional

import pandas as pd
from time import time

from mxnet.context import Context

from actableai.timeseries.models import (
    AAITimeSeriesMultiTargetModel,
    AAITimeSeriesSimpleModel,
)
from actableai.timeseries.models.params import BaseParams
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.utils import handle_datetime_column, find_freq


class AAITimeSeriesForecaster:
    """Time Series Forecaster Model."""

    def __init__(
        self,
        date_column: str,
        target_columns: List[str],
        prediction_length: int,
        group_by: List[str] = None,
        real_static_feature: Optional[
            Union[List[float], Dict[Tuple[Any, ...], List[float]]]
        ] = None,
        cat_static_feature: Optional[
            Union[List[Any], Dict[Tuple[Any, ...], List[Any]]]
        ] = None,
        real_dynamic_feature_columns: Optional[List[str]] = None,
        cat_dynamic_feature_columns: Optional[List[str]] = None,
    ):
        """AAITimeSeriesForecaster Constructor.

        Args:
            date_column: Column containing the date/datetime/time component of the time
                series.
            target_columns: List of columns to forecast, if None all the columns will
                be selected.
            prediction_length: Length of the prediction to forecast.
            group_by: List of columns to use to separate different time series/groups.
            real_static_feature: Dictionary or List containing the real static features,
                if dictionary it represents the features for each group, if list it
                means that there is no groups.
            cat_static_feature: Dictionary or List containing the categorical static
                features, if dictionary it represents the features for each group, if
                list it means that there is no groups.
            real_dynamic_feature_columns: List of columns containing real features.
            cat_dynamic_feature_columns: List of columns containing categorical
                features.
        """
        self.date_column = date_column
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.group_by = group_by
        self.real_static_feature_dict = real_static_feature
        self.cat_static_feature_dict = cat_static_feature
        self.real_dynamic_feature_columns = real_dynamic_feature_columns
        self.cat_dynamic_feature_columns = cat_dynamic_feature_columns

        if self.real_static_feature_dict is None:
            self.real_static_feature_dict = {}
        if self.cat_static_feature_dict is None:
            self.cat_static_feature_dict = {}
        if self.real_dynamic_feature_columns is None:
            self.real_dynamic_feature_columns = []
        if self.cat_dynamic_feature_columns is None:
            self.cat_dynamic_feature_columns = []
        if self.group_by is None:
            self.group_by = []

        if isinstance(self.real_static_feature_dict, list):
            self.real_static_feature_dict = {("data",): self.real_static_feature_dict}
        if isinstance(self.cat_static_feature_dict, list):
            self.cat_static_feature_dict = {("data",): self.cat_static_feature_dict}

        self.model = None

    @staticmethod
    def pre_process_data(
        df: pd.DataFrame,
        date_column: str,
        group_by: Optional[List[str]] = None,
        inplace: bool = True,
    ) -> Tuple[
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], int],
        Dict[Tuple[Any, ...], str],
    ]:
        """Pre-process dataframe to separate groups and handle datetime.

        Args:
            df: Input DataFrame
            date_column: Column containing the date/datetime/time component of the time
                series.
            group_by: List of columns to use to separate different time series/groups.
            inplace: If True this function will modify the original DataFrame.

        Returns:
            - Dictionary containing the time series for each group.
            - Dictionary containing the unique label for each group.
            - Dictionary containing the frequency of each group.
        """
        if group_by is None:
            group_by = []

        df_dict = {}
        group_dict = {}
        freq_dict = {}

        # Create groups
        if len(group_by) > 0:
            for group_index, (group, grouped_df) in enumerate(df.groupby(group_by)):
                if len(group_by) == 1:
                    group = (group,)

                group_dict[group] = group_index
                df_dict[group] = grouped_df.reset_index(drop=True)
        else:
            df_dict[("data",)] = df

        # Process groups
        for group in df_dict.keys():
            # Handle datetime
            pd_date, _ = handle_datetime_column(df_dict[group][date_column])

            # Find frequency
            freq = find_freq(pd_date)
            freq_dict[group] = freq

            if not inplace:
                df_dict[group] = df_dict[group].copy()

            # Sort dataframe
            df_dict[group].index = pd_date
            df_dict[group].name = date_column
            df_dict[group].sort_index(inplace=True)

        return df_dict, group_dict, freq_dict

    def fit(
        self,
        model_params: List[BaseParams],
        mx_ctx: Context,
        *,
        df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
        freq: Optional[str] = None,
        group_dict: Optional[Dict[Tuple[Any, ...], int]] = None,
        loss: str = "mean_wQuantileLoss",
        trials: int = 3,
        max_concurrent: Optional[int] = None,
        use_ray: bool = True,
        tune_samples: int = 3,
        sampling_method: str = "random",
        random_state: Optional[int] = None,
        ray_tune_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ) -> float:
        """Tune and fit the model.

        Args:
            model_params: List of models parameters to run the tuning search on.
            mx_ctx: mxnet context.
            df: Input DataFrame. If None `df_dict`, `freq`, and `group_dict` must be
                provided.
            df_dict: Dictionary containing the time series for each group. If None `df`
                must be provided.
            freq: Frequency of the time series.
            group_dict: Dictionary containing the unique label for each group.
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

        Returns:
            Total time spent for tuning.
        """
        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")
        if df is None and freq is None:
            raise Exception("freq cannot be None if df is None")
        if df is None and group_dict is None:
            raise Exception("group_dict cannot be None if df is None")

        if df_dict is None:
            df_dict, group_dict, freq_dict = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                group_by=self.group_by,
                inplace=False,
            )
            first_group = list(df_dict.keys())[0]
            freq = freq_dict[first_group]

        univariate_model_params = []
        multivariate_model_params = []
        for model_param in model_params:
            if model_param.is_multivariate_model:
                multivariate_model_params.append(model_param)
            else:
                univariate_model_params.append(model_param)

        multi_target_model = AAITimeSeriesMultiTargetModel(
            target_columns=self.target_columns,
            prediction_length=self.prediction_length,
            freq=freq,
            group_dict=group_dict,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
        )
        multi_target_fit_time = multi_target_model.fit(
            df_dict=df_dict,
            model_params=univariate_model_params,
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
            fit_full=False,
        )

        multivariate_model = None
        multivariate_fit_time = 0
        if len(self.target_columns) > 1 and len(multivariate_model_params) > 0:
            multivariate_model = AAITimeSeriesSimpleModel(
                target_columns=self.target_columns,
                prediction_length=self.prediction_length,
                freq=freq,
                group_dict=group_dict,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=self.real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            )

            multivariate_fit_time = multivariate_model.fit(
                df_dict=df_dict,
                model_params=multivariate_model_params,
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
                fit_full=False,
            )

        start_time = time()

        if multivariate_model is not None:
            _, _, df_multi_target_agg_metrics = multi_target_model.score(df_dict)
            _, _, df_multivariate_agg_metrics = multivariate_model.score(df_dict)

            multi_target_loss = df_multi_target_agg_metrics[loss].mean(axis=0)
            multivariate_loss = df_multivariate_agg_metrics[loss].mean(axis=0)

            if multi_target_loss < multivariate_loss:
                self.model = multi_target_model
            else:
                self.model = multivariate_model
        else:
            self.model = multi_target_model

        self.model.refit(df_dict)

        return multi_target_fit_time + multivariate_fit_time + time() - start_time

    def refit(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
    ):
        """Fit previously tuned model.

        Args:
            df: Input DataFrame. If None `df_dict` must be provided.
            df_dict: Dictionary containing the time series for each group. If None `df`
                must be provided.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df is not None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                group_by=self.group_by,
                inplace=False,
            )

        self.model.refit(df_dict)

    def score(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
        num_samples: int = 100,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        num_workers: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Evaluate model.

        Args:
            df: Input DataFrame. If None `df_dict` must be provided.
            df_dict: Dictionary containing the time series for each group. If None `df`
                must be provided.
            num_samples: Number of dataset samples to use for evaluation
            quantiles: List of quantiles to use for evaluation.
            num_workers: Maximum number of workers to use, if None no parallelization
                will be done.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            - Predicted time series.
            - Metrics for each target and groups.
            - Aggregated metrics for each target.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df is not None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict, df_item_metrics_dict, df_agg_metrics = self.model.score(
            df_dict,
            num_samples=num_samples,
            quantiles=quantiles,
            num_workers=num_workers,
        )

        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        df_item_metrics = pd.DataFrame()
        for group, df_group in df_item_metrics_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_item_metrics = pd.concat([df_item_metrics, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)
            df_item_metrics[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )
        df_item_metrics = df_item_metrics.drop(columns="_group")

        return df_predictions, df_item_metrics, df_agg_metrics

    def predict(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> pd.DataFrame:
        """Make a prediction using the model.

        Args:
            df: Input DataFrame. If None `df_dict` must be provided.
            df_dict: Dictionary containing the time series for each group. If None `df`
                must be provided.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Predicted time series.
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df is not None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict = self.model.predict(df_dict, quantiles=quantiles)

        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )

        return df_predictions
