import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from actableai.callbacks.time_series import AAITimeSeriesCallback
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIForecastTask(AAITask):
    """Forecast (time series) Task"""

    @staticmethod
    def _split_static_dynamic_features(
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        df_unique: pd.DataFrame,
        real_feature_columns: List[str],
        cat_feature_columns: List[str],
    ) -> Tuple[
        Dict[Tuple[Any, ...], List[float]],
        Dict[Tuple[Any, ...], List[Any]],
        List[str],
        List[str],
    ]:
        """Split features columns into two groups, static and dynamic.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            df_unique: DataFrame containing the number of unique values for each column.
            real_feature_columns: List of real feature columns.
            cat_feature_columns: List of categorical feature columns.

        Returns:
            - Dictionary containing a list of real static features for each group.
            - Dictionary containing a list of categorical static features for each
                group.
            - List of columns containing real dynamic features.
            - List of columns containing categorical dynamic features.
        """
        real_static_feature_dict = {group: [] for group in group_df_dict.keys()}
        cat_static_feature_dict = {group: [] for group in group_df_dict.keys()}
        real_dynamic_feature_columns = []
        cat_dynamic_feature_columns = []

        # Real columns
        for column in real_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in group_df_dict.items():
                    group_feature_value = df_group[column].loc[
                        df_group[column].first_valid_index()
                    ]
                    real_static_feature_dict[group].append(group_feature_value)
            else:
                real_dynamic_feature_columns.append(column)

        # Categorical columns
        for column in cat_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in group_df_dict.items():
                    group_feature_value = df_group[column].loc[
                        df_group[column].first_valid_index()
                    ]
                    cat_static_feature_dict[group].append(group_feature_value)
            else:
                cat_dynamic_feature_columns.append(column)

        return (
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

    @staticmethod
    def _split_train_valid_predict(
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        freq_dict: Dict[Tuple[Any, ...], str],
        prediction_length: int,
        predicted_columns: List[str],
        group_by: List[str],
        real_dynamic_feature_columns: List[str],
        cat_dynamic_feature_columns: List[str],
    ) -> Tuple[
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], pd.DataFrame],
    ]:
        """Split dataset into three sub datasets, train, validation, and prediction.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            freq_dict: Dictionary containing the frequency of each group.
            prediction_length: Length of the prediction to forecast.
            predicted_columns: List of columns to forecast.
            group_by: List of columns to use to separate different time series/groups.
            real_dynamic_feature_columns: List of columns containing real dynamic
                features.
            cat_dynamic_feature_columns: List of columns containing categorical dynamic
                features.

        Returns:
            - Dictionary containing the training time series for each group.
            - Dictionary containing the validation time series for each group.
            - Dictionary containing the prediction time series for each group.
        """
        from actableai.timeseries.utils import interpolate

        has_dynamic_features = (
            len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) > 0
        )
        group_df_train_dict = {}
        group_df_valid_dict = {}
        group_df_predict_dict = {}
        for group in group_df_dict.keys():
            # Filter Dataframe
            group_df_dict[group] = group_df_dict[group][
                predicted_columns
                + real_dynamic_feature_columns
                + cat_dynamic_feature_columns
                + group_by
            ]

            last_valid_index = (
                -prediction_length
                if has_dynamic_features
                else group_df_dict[group].shape[0]
            )

            # Interpolate missing values
            group_df_dict[group] = pd.concat(
                [
                    interpolate(
                        group_df_dict[group].iloc[:last_valid_index], freq_dict[group]
                    ),
                    group_df_dict[group].iloc[last_valid_index:],
                ]
            )

            # Split train/validation/test
            group_df_train_dict[group] = group_df_dict[group].iloc[
                : last_valid_index - prediction_length
            ]
            group_df_valid_dict[group] = group_df_dict[group].iloc[:last_valid_index]
            group_df_predict_dict[group] = group_df_dict[group]

        return group_df_train_dict, group_df_valid_dict, group_df_predict_dict

    @staticmethod
    def _get_default_model_params(
        train_size: int,
        prediction_length: int,
        predicted_columns: List[str],
        real_dynamic_feature_columns: List[str],
        cat_dynamic_feature_columns: List[str],
    ) -> List[object]:
        """Get/generate default model parameters.

        Args:
            train_size: size of the training dataset.
            prediction_length: Length of the prediction to forecast.
            predicted_columns: List of columns to forecast.
            real_dynamic_feature_columns: List of columns containing real dynamic
                features.
            cat_dynamic_feature_columns: List of columns containing categorical dynamic
                features.

        Returns:
            List containing the default parameters.
        """
        from actableai.timeseries.models import params

        model_params = [
            params.ProphetParams(),
            params.RForecastParams(
                method_name=("thetaf", "stlar", "arima", "ets"),
            ),
            # params.TreePredictorParams(
            #    use_feat_dynamic_cat=len(cat_dynamic_feature_columns) > 0,
            #    use_feat_dynamic_real=len(real_dynamic_feature_columns) > 0
            #    or len(predicted_columns) > 1,
            #    method=("QRX", "QuantileRegression"),
            #    context_length=(1, 2 * prediction_length),
            # ),
            params.DeepVARParams(
                epochs=(5, 20),
                num_layers=(1, 3),
                num_cells=(1, 20),
                scaling=False,
                context_length=(prediction_length, 2 * prediction_length),
            ),
        ]

        if train_size >= 1000:
            model_params.append(
                params.DeepARParams(
                    context_length=(1, 2 * prediction_length),
                    epochs=(1, 20),
                    num_layers=(1, 3),
                    num_cells=(1, 10),
                    use_feat_dynamic_real=len(real_dynamic_feature_columns) > 0
                    or len(predicted_columns) > 1,
                ),
            )

        return model_params

    @staticmethod
    def _convert_to_legacy_output(
        df_item_metrics: pd.DataFrame,
        df_val_predictions: pd.DataFrame,
        df_predictions: pd.DataFrame,
        date_column: str,
        prediction_length: int,
        group_by: List[str],
        group_df_valid_dict: Dict[Tuple[Any, ...], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Convert time series forecasting scoring to 'legacy' output.

        Args:
            df_item_metrics: Metrics for each target and groups.
            df_val_predictions: Predicted time series for validation.
            df_predictions: Predicted time series.
            date_column: Column containing the date/datetime/time component of the time
                series.
            prediction_length: Length of the prediction to forecast.
            group_by: List of columns to use to separate different time series/groups.
            group_df_valid_dict: Dictionary containing the validation time series for
                each group.

        Returns:
            Legacy output.
        """
        # TODO REMOVE LEGACY CODE/FUNCTION
        val_dates = [
            df_group_valid_dict.index[-prediction_length:]
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
            for df_group_valid_dict in group_df_valid_dict.values()
        ]
        if len(group_by) <= 0:
            val_dates = val_dates[0]

        df_item_metrics["item_id"] = df_item_metrics["target"]
        df_item_metrics.index = df_item_metrics["item_id"]

        df_val_predictions_items = {}
        df_predictions_items = {}
        if len(group_by) > 0:
            df_val_predictions_items = [
                (group if len(group_by) > 1 else (group,), group_df)
                for group, group_df in df_val_predictions.groupby(group_by)
            ]
            df_predictions_items = [
                (group if len(group_by) > 1 else (group,), group_df)
                for group, group_df in df_predictions.groupby(group_by)
            ]
        else:
            df_val_predictions_items = [(("data",), df_val_predictions)]
            df_predictions_items = [(("data",), df_predictions)]

        data = {
            "predict": [
                [
                    {
                        "name": target,
                        "group": group,
                        "value": {
                            "data": {
                                "date": group_df_valid_dict[group]
                                .index.strftime("%Y-%m-%d %H:%M:%S")[
                                    -4 * prediction_length :
                                ]
                                .tolist(),
                                "value": group_df_valid_dict[group][target][
                                    -4 * prediction_length :
                                ].tolist(),
                            },
                            "prediction": {
                                "date": df_group_target_predictions.sort_values(
                                    by=date_column
                                )[date_column]
                                .dt.strftime("%Y-%m-%d %H:%M:%S")
                                .tolist(),
                                "min": df_group_target_predictions.sort_values(
                                    by=date_column
                                )["0.05"].tolist(),
                                "median": df_group_target_predictions.sort_values(
                                    by=date_column
                                )["0.5"].tolist(),
                                "max": df_group_target_predictions.sort_values(
                                    by=date_column
                                )["0.95"].tolist(),
                            },
                        },
                    }
                    for target, df_group_target_predictions in df_group_predictions.groupby(
                        "target"
                    )
                ]
                for group, df_group_predictions in df_predictions_items
            ],
            "evaluate": {
                "dates": val_dates,
                "values": [
                    [
                        {
                            "q5": df_group_target_predictions.sort_values(date_column)[
                                "0.05"
                            ].tolist(),
                            "q50": df_group_target_predictions.sort_values(date_column)[
                                "0.5"
                            ].tolist(),
                            "q95": df_group_target_predictions.sort_values(date_column)[
                                "0.95"
                            ].tolist(),
                        }
                        for _, df_group_target_predictions in df_group_predictions.groupby(
                            "target"
                        )
                    ]
                    for _, df_group_predictions in df_val_predictions_items
                ],
                "agg_metrics": None,
                # Not used in the frontend, and not compatible with multivariate
                "item_metrics": df_item_metrics.to_dict(),
            },
        }

        return data

    @AAITask.run_with_ray_remote(TaskType.FORECAST)
    def run(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        date_column: Optional[str] = None,
        predicted_columns: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        ray_tune_kwargs: Optional[Dict] = None,
        max_concurrent: int = 3,
        trials: int = 1,
        model_params: Optional[List[object]] = None,
        use_ray: bool = True,
        tune_samples: int = 20,
        refit_full: bool = True,
        verbose: int = 3,
        seed: int = 123,
        sampling_method: str = "random",
        callback: Optional[AAITimeSeriesCallback] = None,
    ) -> Dict[str, Any]:
        """Run time series forecasting task and return results.

        Args:
            df: Input DataFrame.
            prediction_length: Length of the prediction to forecast.
            date_column: Column containing the date/datetime/time component of the time
                series.
            predicted_columns: List of columns to forecast, if None all the columns will
                be selected.
            group_by: List of columns to use to separate different time series/groups.
                This list is used by the `groupby` function of the pandas library.
            feature_columns: List of columns containing extraneous features used to
                forecast. If one or more feature columns contain dynamic features
                (features that change over time) the dataset must contain
                `prediction_length` features data points in the future.
            ray_tune_kwargs: Named parameters to pass to ray's `tune` function.
            max_concurrent: Maximum number of concurrent ray task.
            trials: Number of trials for hyperparameter search.
            model_params: List of model parameters to run the tuning search on. If None
                some default models will be used.
            use_ray: If True ray will be used for hyperparameter tuning.
            tune_samples: Number of dataset samples to use when tuning.
            refit_full: If True the final model will be fitted using all the data
                (including the validation set).
            verbose: Verbose level.
            seed: Random seed to use.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].

        Returns:
            Dict: Dictionary containing the results.
        """
        import time
        import mxnet as mx
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        from actableai.timeseries.models import AAITimeSeriesForecaster
        from actableai.data_validation.params import (
            TimeSeriesDataValidator,
            TimeSeriesPredictionDataValidator,
        )
        from actableai.data_validation.base import CheckLevels
        from actableai.timeseries.utils import (
            handle_datetime_column,
            find_freq,
        )
        from actableai.utils.sanitize import sanitize_timezone
        from actableai.utils import get_type_special

        # FIXME random seed not working here
        np.random.seed(seed)

        # FIXME this should not be needed
        pd.set_option("chained_assignment", "warn")
        start_time = time.time()

        # Pre process parameters
        if predicted_columns is None:
            predicted_columns = df.columns
        if feature_columns is None:
            feature_columns = []
        if group_by is None:
            group_by = []
        if ray_tune_kwargs is None:
            ray_tune_kwargs = {
                "resources_per_trial": {
                    "cpu": 3,
                    "gpu": 0,
                },
            }

        if "raise_on_failed_trial" not in ray_tune_kwargs:
            ray_tune_kwargs["raise_on_failed_trial"] = False

        # To resolve any issues of access rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        if date_column is None:
            df["_date"] = df.index
            df = df.reset_index(drop=True)

        # First parameters validation
        data_validation_results = TimeSeriesDataValidator().validate(
            df, date_column, predicted_columns, feature_columns, group_by
        )
        failed_checks = [x for x in data_validation_results if x is not None]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start_time,
                "data": {},
            }

        # Separate cat and real feature columns
        real_feature_columns = []
        cat_feature_columns = []
        for column in feature_columns:
            if get_type_special(df[column]) == "category":
                cat_feature_columns.append(column)
            else:
                real_feature_columns.append(column)

        # Encode categorical columns
        if len(cat_feature_columns) > 0:
            df[cat_feature_columns] = df[cat_feature_columns].apply(
                LabelEncoder().fit_transform
            )

        df_unique = None
        if len(group_by) > 0:
            df_unique = df.groupby(group_by).nunique().max()
        else:
            df_unique = df.nunique()

        (
            group_df_dict,
            group_label_dict,
            freq_dict,
        ) = AAITimeSeriesForecaster.pre_process_data(
            df=df,
            date_column=date_column,
            group_by=group_by,
            inplace=True,
        )

        # Separate static from dynamic feature columns
        (
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        ) = self._split_static_dynamic_features(
            group_df_dict, df_unique, real_feature_columns, cat_feature_columns
        )

        # Split dataset
        (
            group_df_train_dict,
            group_df_valid_dict,
            group_df_predict_dict,
        ) = self._split_train_valid_predict(
            group_df_dict,
            freq_dict,
            prediction_length,
            predicted_columns,
            group_by,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

        # Second Data Validation (for the prediction part of the data which needed pre-processing)
        data_prediction_validation_results = (
            TimeSeriesPredictionDataValidator().validate(
                group_df_train_dict,
                group_df_valid_dict,
                group_df_predict_dict,
                freq_dict,
                real_dynamic_feature_columns + cat_dynamic_feature_columns,
                predicted_columns,
                prediction_length,
            )
        )
        failed_checks = [x for x in data_prediction_validation_results if x is not None]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start_time,
                "data": {},
            }

        first_group = list(group_df_train_dict.keys())[0]
        freq = freq_dict[first_group]

        ray_gpu_per_trial = 0
        if "resources_per_trial" in ray_tune_kwargs:
            ray_gpu_per_trial = ray_tune_kwargs["resources_per_trial"].get("gpu", 0)
        mx_ctx = mx.gpu() if ray_gpu_per_trial > 0 else mx.cpu()

        if model_params is None:
            model_params = self._get_default_model_params(
                len(group_df_train_dict[first_group]),
                prediction_length,
                predicted_columns,
                real_dynamic_feature_columns,
                cat_dynamic_feature_columns,
            )

        model = AAITimeSeriesForecaster(
            date_column=date_column,
            target_columns=predicted_columns,
            prediction_length=prediction_length,
            group_by=group_by,
            real_static_feature=real_static_feature_dict,
            cat_static_feature=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        )
        total_trials_times = model.fit(
            model_params=model_params,
            mx_ctx=mx_ctx,
            group_df_dict=group_df_train_dict,
            freq=freq,
            group_label_dict=group_label_dict,
            loss="mean_wQuantileLoss",
            trials=trials,
            max_concurrent=max_concurrent,
            use_ray=use_ray,
            tune_samples=tune_samples,
            sampling_method=sampling_method,
            random_state=seed,
            ray_tune_kwargs=ray_tune_kwargs,
            verbose=verbose,
            callback=callback,
        )

        start = time.time()

        # Generate validation results
        (
            df_val_predictions,
            df_item_metrics,
            df_agg_metrics,
        ) = model.score(group_df_dict=group_df_valid_dict)

        # Refit with validation data
        if refit_full:
            model.refit(
                group_df_dict=group_df_valid_dict,
                mx_ctx=mx_ctx,
            )

        # Generate predictions
        df_predictions = model.predict(group_df_dict=group_df_predict_dict)

        # TODO REMOVE LEGACY CODE
        data = self._convert_to_legacy_output(
            df_item_metrics,
            df_val_predictions,
            df_predictions,
            date_column,
            prediction_length,
            group_by,
            group_df_valid_dict,
        )

        runtime = time.time() - start + total_trials_times

        return {
            "status": "SUCCESS",
            "messenger": "",
            "data_v2": {
                "predict": df_predictions,
                "validation": {
                    "predict": df_val_predictions,
                    "agg_metrics": df_agg_metrics,
                    "item_metrics": df_item_metrics,
                },
            },
            "data": data,  # TODO remove legacy code
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "runtime": runtime,
        }
