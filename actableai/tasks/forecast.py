import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIForecastTask(AAITask):
    """Forecast (time series) Task"""

    @staticmethod
    def _split_train_valid_predict(
        dataset: Any,
        prediction_length: int,
    ) -> Tuple[Any, Any, Any]:
        """Split dataset into three sub datasets, train, validation, and prediction.

        Args:
            dataset: Dataset containing the time series.
            prediction_length: Length of the prediction to forecast.

        Returns:
            - Dataset containing the training time series.
            - Dataset containing the validation time series.
            - Dataset containing the prediction time series.
        """
        from actableai.timeseries.utils import interpolate
        from actableai.timeseries.dataset import AAITimeSeriesDataset

        df_train_dict = {}
        df_valid_dict = {}
        df_predict_dict = {}
        for group, df in dataset.dataframes.items():
            last_valid_index = (
                -prediction_length if dataset.has_dynamic_features else df.shape[0]
            )

            # Interpolate missing values
            df = pd.concat(
                [
                    interpolate(df.iloc[:last_valid_index], dataset.freq),
                    df.iloc[last_valid_index:],
                ]
            )

            if not dataset.has_dynamic_features:
                last_valid_index = df.shape[0]

            # Split train/validation/test
            df_train_dict[group] = df.iloc[: last_valid_index - prediction_length]
            df_valid_dict[group] = df.iloc[:last_valid_index]
            df_predict_dict[group] = df

        return (
            AAITimeSeriesDataset(
                dataframes=df_train_dict,
                target_columns=dataset.target_columns,
                freq=dataset.freq,
                prediction_length=dataset.prediction_length,
                feat_dynamic_real=dataset.feat_dynamic_real,
                feat_dynamic_cat=dataset.feat_dynamic_cat,
                feat_static_real=dataset.feat_static_real,
                feat_static_cat=dataset.feat_static_cat,
            ),
            AAITimeSeriesDataset(
                dataframes=df_valid_dict,
                target_columns=dataset.target_columns,
                freq=dataset.freq,
                prediction_length=dataset.prediction_length,
                feat_dynamic_real=dataset.feat_dynamic_real,
                feat_dynamic_cat=dataset.feat_dynamic_cat,
                feat_static_real=dataset.feat_static_real,
                feat_static_cat=dataset.feat_static_cat,
            ),
            AAITimeSeriesDataset(
                dataframes=df_predict_dict,
                target_columns=dataset.target_columns,
                freq=dataset.freq,
                prediction_length=dataset.prediction_length,
                feat_dynamic_real=dataset.feat_dynamic_real,
                feat_dynamic_cat=dataset.feat_dynamic_cat,
                feat_static_real=dataset.feat_static_real,
                feat_static_cat=dataset.feat_static_cat,
            ),
        )

    @staticmethod
    def _get_default_model_params(
        train_size: float,
        prediction_length: int,
        n_groups: int,
        predicted_columns: List[str],
        feat_dynamic_real: List[str],
        feat_dynamic_cat: List[str],
        feat_static_real: List[str],
        feat_static_cat: List[str],
    ) -> List[object]:
        """Get/generate default model parameters.

        Args:
            train_size: size of the training dataset.
            prediction_length: Length of the prediction to forecast.
            n_groups: Number of groups.
            predicted_columns: List of columns to forecast.
            feat_dynamic_real: List of columns containing real dynamic
                features.
            feat_dynamic_cat: List of columns containing categorical dynamic
                features.
            feat_static_real: List of columns containing real static features.
            feat_static_cat: List of columns containing categorical static features.

        Returns:
            List containing the default parameters.
        """
        from actableai.timeseries.models import params

        model_params = [
            params.ProphetParams(),
            params.RForecastParams(
                method_name=("arima", "ets"),
            ),
            params.TreePredictorParams(
                use_feat_dynamic_cat=len(feat_dynamic_cat) > 0,
                use_feat_static_real=len(feat_static_real) > 0,
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

        if train_size >= 1000:
            model_params.append(
                params.DeepARParams(
                    context_length=(1, 2 * prediction_length),
                    epochs=(1, 20),
                    num_layers=(1, 3),
                    num_cells=(1, 10),
                    use_feat_dynamic_real=len(feat_dynamic_real) > 0,
                    use_feat_static_real=len(feat_static_real) > 0,
                ),
            )
            if n_groups >= 10:
                model_params.append(
                    params.NBEATSParams(
                        context_length=(prediction_length, 2 * prediction_length),
                        epochs=(5, 20),
                    )
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
        valid_dataset: Any,
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
            valid_dataset: Dataset containing the validation time series.

        Returns:
            Legacy output.
        """
        # TODO REMOVE LEGACY CODE/FUNCTION
        val_dates = [
            df_group_valid_dict.index[-prediction_length:]
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
            for df_group_valid_dict in valid_dataset.dataframes.values()
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
                                "date": valid_dataset.dataframes[group]
                                .index.strftime("%Y-%m-%d %H:%M:%S")[
                                    -4 * prediction_length :
                                ]
                                .tolist(),
                                "value": valid_dataset.dataframes[group][target][
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
        tuning_metric: str = "mean_wQuantileLoss",
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
            tuning_metric: Metric to to minimize when tuning.

        Returns:
            Dict: Dictionary containing the results.
                - "model": AAIModel to redeploy the model
        """
        import time
        import mxnet as mx
        import numpy as np
        from actableai.timeseries.models import AAITimeSeriesForecaster
        from actableai.data_validation.params import (
            TimeSeriesDataValidator,
            TimeSeriesPredictionDataValidator,
        )
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.sanitize import sanitize_timezone
        from actableai.models.aai_predictor import AAITimeSeriesForecasterModel

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
            df, date_column, predicted_columns, feature_columns, group_by, tuning_metric
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

        dataset = AAITimeSeriesForecaster.pre_process_data(
            df=df,
            date_column=date_column,
            target_columns=predicted_columns,
            prediction_length=prediction_length,
            feature_columns=feature_columns,
            group_by=group_by,
            inplace=True,
        )

        # Split dataset
        train_dataset, valid_dataset, predict_dataset = self._split_train_valid_predict(
            dataset, prediction_length
        )

        # Second Data Validation (for the prediction part of the data which needed pre-processing)
        data_prediction_validation_results = (
            TimeSeriesPredictionDataValidator().validate(
                train_dataset,
                valid_dataset,
                predict_dataset,
                dataset.feat_dynamic_real + dataset.feat_dynamic_cat,
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

        ray_gpu_per_trial = 0
        if "resources_per_trial" in ray_tune_kwargs:
            ray_gpu_per_trial = ray_tune_kwargs["resources_per_trial"].get("gpu", 0)
        mx_ctx = mx.gpu() if ray_gpu_per_trial > 0 else mx.cpu()

        mean_train_size = 0.0
        for group, df in train_dataset.dataframes.items():
            mean_train_size += df.shape[0]
        mean_train_size /= len(train_dataset)

        if model_params is None:
            model_params = self._get_default_model_params(
                mean_train_size,
                prediction_length,
                len(dataset),
                predicted_columns,
                dataset.feat_dynamic_real,
                dataset.feat_dynamic_cat,
                dataset.feat_static_real,
                dataset.feat_static_cat,
            )

        model = AAITimeSeriesForecaster(
            date_column=date_column,
            target_columns=predicted_columns,
            prediction_length=prediction_length,
            group_by=group_by,
            feature_columns=feature_columns,
        )
        total_trials_times, df_leaderboard = model.fit(
            model_params=model_params,
            mx_ctx=mx_ctx,
            dataset=train_dataset,
            loss=tuning_metric,
            trials=trials,
            max_concurrent=max_concurrent,
            use_ray=use_ray,
            tune_samples=tune_samples,
            sampling_method=sampling_method,
            random_state=seed,
            ray_tune_kwargs=ray_tune_kwargs,
            verbose=verbose,
        )

        start = time.time()

        # Generate validation results
        (
            df_val_predictions,
            df_item_metrics,
            df_agg_metrics,
        ) = model.score(dataset=valid_dataset)

        # Refit with validation data
        if refit_full:
            model.refit(
                dataset=valid_dataset,
                mx_ctx=mx_ctx,
            )

        # Generate predictions
        df_predictions = model.predict(dataset=predict_dataset)

        # TODO REMOVE LEGACY CODE
        data = self._convert_to_legacy_output(
            df_item_metrics,
            df_val_predictions,
            df_predictions,
            date_column,
            prediction_length,
            group_by,
            valid_dataset,
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
                "leaderboard": df_leaderboard,
            },
            "data": data,  # TODO remove legacy code
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "runtime": runtime,
            "model": AAITimeSeriesForecasterModel(model),
        }
