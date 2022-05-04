from typing import Dict, List, Optional, Any
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask

import pandas as pd


class AAIForecastTask(AAITask):
    """Forecast (time series) Task"""

    @AAITask.run_with_ray_remote(TaskType.FORECAST)
    def run(
        self,
        df: pd.DataFrame,
        date_column: str,
        prediction_length: int,
        predicted_columns: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        RAY_CPU_PER_TRIAL: int = 3,
        RAY_GPU_PER_TRIAL: int = 0,
        RAY_MAX_CONCURRENT: int = 3,
        trials: int = 1,
        model_params: Optional[List[object]] = None,
        use_ray: bool = True,
        tune_samples: int = 20,
        refit_full: bool = True,
        verbose: int = 3,
        seed: int = 123,
        sampling_method: str = "random",
    ) -> Dict[str, Any]:
        """Run time series forecasting task and return results.

        Args:
            df: Input DataFrame.
            date_column: Column containing the date/datetime/time component of the time
                series.
            prediction_length: Length of the prediction to forecast.
            predicted_columns: List of columns to forecast, if None all the columns will
                be selected.
            group_by: List of columns to use to separate different time series/groups.
                This list is used by the `groupby` function of the pandas library.
            feature_columns: List of columns containing extraneous features used to
                forecast. If one or more feature columns contain dynamic features
                (features that change over time) the dataset must contain
                `prediction_length` features data points in the future.
            RAY_GPU_PER_TRIAL: Number of CPU to use per trial.
            RAY_GPU_PER_TRIAL: Number of GPU to use per trial.
            RAY_MAX_CONCURRENT: Maximum number of concurrent ray task.
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
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from actableai.timeseries.models import params, AAITimeSeriesForecaster
        from actableai.data_validation.params import (
            TimeSeriesDataValidator,
            TimeSeriesPredictionDataValidator,
        )
        from actableai.data_validation.base import CheckLevels
        from actableai.timeseries.util import (
            handle_datetime_column,
            find_freq,
            interpolate,
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

        # To resolve any issues of access rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

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

        df_dict, group_dict, freq_dict = AAITimeSeriesForecaster.pre_process_data(
            df=df,
            date_column=date_column,
            group_by=group_by,
            inplace=True,
        )

        # Separate static from dynamic feature columns
        real_static_feature_dict = {group: [] for group in df_dict.keys()}
        cat_static_feature_dict = {group: [] for group in df_dict.keys()}
        real_dynamic_feature_columns = []
        cat_dynamic_feature_columns = []

        # Real columns
        for column in real_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in df_dict.items():
                    group_feature_value = df_group[column].loc[
                        df_group[column].first_valid_index()
                    ]
                    real_static_feature_dict[group].append(group_feature_value)
            else:
                real_dynamic_feature_columns.append(column)

        # Categorical columns
        for column in cat_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in df_dict.items():
                    group_feature_value = df_group[column].loc[
                        df_group[column].first_valid_index()
                    ]
                    cat_static_feature_dict[group].append(group_feature_value)
            else:
                cat_dynamic_feature_columns.append(column)

        has_dynamic_features = (
            len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) > 0
        )
        df_train_dict = {}
        df_valid_dict = {}
        df_predict_dict = {}
        for group in df_dict.keys():
            # Filter Dataframe
            df_dict[group] = df_dict[group][
                predicted_columns
                + real_dynamic_feature_columns
                + cat_dynamic_feature_columns
                + group_by
            ]

            last_valid_index = (
                -prediction_length if has_dynamic_features else df_dict[group].shape[0]
            )

            # Interpolate missing values
            df_dict[group] = pd.concat(
                [
                    interpolate(
                        df_dict[group].iloc[:last_valid_index], freq_dict[group]
                    ),
                    df_dict[group].iloc[last_valid_index:],
                ]
            )

            # Split train/validation/test
            df_train_dict[group] = df_dict[group].iloc[
                : last_valid_index - prediction_length
            ]
            df_valid_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_predict_dict[group] = df_dict[group]

        # Second Data Validation (for the prediction part of the data which needed pre-processing)
        data_prediction_validation_results = (
            TimeSeriesPredictionDataValidator().validate(
                df_train_dict,
                df_valid_dict,
                df_predict_dict,
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

        first_group = list(df_train_dict.keys())[0]
        freq = freq_dict[first_group]

        mx_ctx = mx.gpu() if RAY_GPU_PER_TRIAL > 0 else mx.cpu()

        if model_params is None:
            model_params = [
                params.ProphetParams(),
                params.RForecastParams(
                    method_name=("thetaf", "stlar", "arima", "ets"),
                ),
                params.TreePredictorParams(
                    use_feat_dynamic_cat=len(cat_dynamic_feature_columns) > 0,
                    use_feat_dynamic_real=len(real_dynamic_feature_columns) > 0
                    or len(predicted_columns) > 1,
                    method=("QRX", "QuantileRegression"),
                    context_length=(1, 2 * prediction_length),
                ),
                params.DeepVARParams(
                    epochs=(5, 20),
                    num_layers=(1, 3),
                    num_cells=(1, 20),
                    scaling=False,
                    context_length=(prediction_length, 2 * prediction_length),
                ),
            ]

            if len(df_train_dict[first_group]) >= 1000:
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
            df_dict=df_train_dict,
            freq=freq,
            group_dict=group_dict,
            loss="mean_wQuantileLoss",
            trials=trials,
            max_concurrent=RAY_MAX_CONCURRENT,
            use_ray=use_ray,
            tune_samples=tune_samples,
            sampling_method=sampling_method,
            random_state=seed,
            ray_tune_kwargs={
                "resources_per_trial": {
                    "cpu": RAY_CPU_PER_TRIAL,
                    "gpu": RAY_GPU_PER_TRIAL,
                },
                "raise_on_failed_trial": False,
            },
            verbose=verbose,
        )

        start = time.time()

        # Generate validation results
        (
            df_val_predictions,
            df_item_metrics,
            df_agg_metrics,
        ) = model.score(df_dict=df_valid_dict)

        # Refit with validation data
        if refit_full:
            model.refit(df_dict=df_valid_dict)

        # Generate predictions
        df_predictions = model.predict(df_dict=df_predict_dict)

        # TODO REMOVE LEGACY CODE
        # --------------------
        val_dates = [
            df_group_valid_dict.index[-prediction_length:]
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
            for df_group_valid_dict in df_valid_dict.values()
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
                                "date": df_valid_dict[group]
                                .index.strftime("%Y-%m-%d %H:%M:%S")[
                                    -4 * prediction_length :
                                ]
                                .tolist(),
                                "value": df_valid_dict[group][target][
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
                "agg_metrics": None,  # Not used in the frontend, and not compatible with multivariate
                "item_metrics": df_item_metrics.to_dict(),
            },
        }
        # --------------------

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
