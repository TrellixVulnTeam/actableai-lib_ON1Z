from typing import Dict, List
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask

import pandas as pd

class AAIForecastTask(AAITask):
    """
    Forecast (time series) Task
    """

    @AAITask.run_with_ray_remote(TaskType.FORECAST)
    def run(self,
            df,
            date_column,
            predicted_columns=None,
            prediction_length=None,
            group_by=None,
            feature_columns=None,
            RAY_CPU_PER_TRIAL=3,
            RAY_GPU_PER_TRIAL=0,
            RAY_MAX_CONCURRENT=3,
            epochs="auto",
            num_cells="auto",
            num_layers="auto",
            dropout_rate="auto",
            learning_rate="auto",
            trials=1,
            model_params=None,
            use_ray=True,
            tune_samples=20,
            refit_full=True,
            verbose=3,
            seed=123):
        """
        TODO write documentation
        """
        import time
        import torch
        import mxnet as mx
        import numpy as np
        import pandas as pd
        from copy import copy
        from sklearn.preprocessing import LabelEncoder
        from actableai.timeseries import params
        from actableai.timeseries.forecaster import AAITimeSeriesForecaster
        from actableai.data_validation.params import TimeSeriesDataValidator, TimeSeriesPredictionDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.timeseries.util import handle_datetime_column, find_freq, interpolate
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

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        # First parameters validation
        data_validation_results = TimeSeriesDataValidator().validate(
            df,
            date_column,
            predicted_columns,
            feature_columns,
            group_by
        )
        failed_checks = [x for x in data_validation_results if x is not None]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [{"name": x.name, "level": x.level, "message": x.message} for x in failed_checks],
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
            df[cat_feature_columns] = df[cat_feature_columns].apply(LabelEncoder().fit_transform)

        # Create grouped df
        df_dict = {}
        group_dict = {}
        df_unique = None
        if len(group_by) > 0:
            df_group_by = df.groupby(group_by)
            df_unique = df_group_by.nunique().max()

            for group_index, (group, grouped_df) in enumerate(df_group_by):
                if len(group_by) == 1:
                    group = (group,)

                group_dict[group] = group_index
                df_dict[group] = grouped_df.reset_index(drop=True)
        else:
            df_dict["data"] = df
            df_unique = df.nunique()

        # Separate static from dynamic feature columns
        real_static_feature_dict = {group: [] for group in df_dict.keys()}
        cat_static_feature_dict = {group: [] for group in df_dict.keys()}
        real_dynamic_feature_columns = []
        cat_dynamic_feature_columns = []

        # Real columns
        for column in real_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in df_dict.items():
                    group_feature_value = df_group[column].loc[df_group[column].first_valid_index()]
                    real_static_feature_dict[group].append(group_feature_value)
            else:
                real_dynamic_feature_columns.append(column)

        # Categorical columns
        for column in cat_feature_columns:
            if df_unique[column] == 1:
                for group, df_group in df_dict.items():
                    group_feature_value = df_group[column].loc[df_group[column].first_valid_index()]
                    cat_static_feature_dict[group].append(group_feature_value)
            else:
                cat_dynamic_feature_columns.append(column)

        has_dynamic_features = len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) > 0
        df_train_dict = {}
        df_valid_dict = {}
        df_predict_dict = {}
        freq_dict = {}
        pd_date_dict = {}
        for group in df_dict.keys():
            # Handle datetime
            pd_date, _ = handle_datetime_column(df_dict[group][date_column])

            # Find Frequencies
            freq = find_freq(pd_date)
            freq_dict[group] = freq


            # Sort Dataframe
            df_dict[group] = df_dict[group][
                predicted_columns + real_dynamic_feature_columns + cat_dynamic_feature_columns + group_by
            ]
            df_dict[group].index = pd_date
            df_dict[group].name = date_column
            df_dict[group].sort_index(inplace=True)

            pd_date_dict[group] = pd.Series(df_dict[group].index)

            last_valid_index = -prediction_length if has_dynamic_features else df_dict[group].shape[0]

            df_dict[group].iloc[:last_valid_index] = interpolate(df_dict[group].iloc[:last_valid_index], freq)

            df_train_dict[group] = df_dict[group].iloc[:last_valid_index - prediction_length]
            df_valid_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_predict_dict[group] = df_dict[group]

        # Second Data Validation (for the prediction part of the data which needed pre-processing)
        data_prediction_validation_results = TimeSeriesPredictionDataValidator().validate(
            df_train_dict,
            df_valid_dict,
            df_predict_dict,
            freq_dict,
            real_dynamic_feature_columns + cat_dynamic_feature_columns,
            predicted_columns,
            prediction_length
        )
        failed_checks = [x for x in data_prediction_validation_results if x is not None]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [{"name": x.name, "level": x.level, "message": x.message} for x in failed_checks],
                "runtime": time.time() - start_time,
                "data": {},
            }

        first_group = list(df_train_dict.keys())[0]
        freq = freq_dict[first_group]

        mx_ctx = mx.gpu() if RAY_GPU_PER_TRIAL > 0 else mx.cpu()
        torch_device = torch.device("cuda" if RAY_GPU_PER_TRIAL > 0 else "cpu")

        if model_params is None:
            model_params=[
                params.ProphetParams(),

                params.RForecastParams(
                    method_name=("thetaf", "stlar", "arima", "ets"),
                ),

                params.TreePredictorParams(
                    use_feat_dynamic_cat=len(cat_dynamic_feature_columns) > 0,
                    use_feat_dynamic_real=len(real_dynamic_feature_columns) > 0 or len(predicted_columns) > 1,
                    method=("QRX", "QuantileRegression"),
                    context_length=(1, 2 * prediction_length)
                ),

                params.DeepARParams(
                    context_length=(1, 2 * prediction_length),
                    epochs=(1, 20),
                    num_layers=(1, 3),
                    num_cells=(1, 10),
                    use_feat_dynamic_real=len(real_dynamic_feature_columns) > 0 or len(predicted_columns) > 1
                ),
            ]

        m = AAITimeSeriesForecaster(
            prediction_length,
            mx_ctx,
            torch_device,
            model_params=model_params
        )

        m.fit(
            df_train_dict,
            freq,
            predicted_columns,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
            group_dict=group_dict,
            trials=trials,
            loss="mean_wQuantileLoss",
            tune_params={
                "resources_per_trial": {
                    "cpu": RAY_CPU_PER_TRIAL,
                    "gpu": RAY_GPU_PER_TRIAL,
                },
                "raise_on_failed_trial": False
            },
            max_concurrent=RAY_MAX_CONCURRENT,
            tune_samples=tune_samples,
            use_ray=use_ray,
            verbose=verbose,
            seed=seed,
        )

        total_trials_times = m.total_trial_time
        start = time.time()

        # Generate validation results
        validations = m.score(df_valid_dict)

        # Refit with validation data
        if refit_full:
            m.refit(df_valid_dict)

        # Generate predictions
        predictions = m.predict(df_predict_dict)

        # Post process data
        df_val_predictions = pd.DataFrame()
        for group, df_group in validations["predictions"].items():
            df_group["_group"] = [group] * len(df_group)
            df_val_predictions = pd.concat([
                df_val_predictions,
                df_group
            ], ignore_index=True)

        df_val_item_metrics = pd.DataFrame()
        for group, df_group in validations["item_metrics"].items():
            df_group["_group"] = [group] * len(df_group)
            df_val_item_metrics = pd.concat([
                df_val_item_metrics,
                df_group
            ], ignore_index=True)

        df_predictions = pd.DataFrame()
        for group, df_group in predictions["predictions"].items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([
                df_predictions,
                df_group
            ], ignore_index=True)

        for group_index, group in enumerate(group_by):
            f_group_values = (lambda group_values: group_values[group_index])
            df_val_predictions[group] = df_val_predictions["_group"].apply(f_group_values)
            df_val_item_metrics[group] = df_val_item_metrics["_group"].apply(f_group_values)
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)

        df_val_predictions = df_val_predictions.rename(columns={"date": date_column}).drop(columns="_group")
        df_predictions = df_predictions.rename(columns={"date": date_column}).drop(columns="_group")

        # FIXME REMOVE LEGACY CODE
        # --------------------
        val_dates = [
            df_group_valid_dict.index[-prediction_length:].strftime("%Y-%m-%d %H:%M:%S").tolist()
            for df_group_valid_dict in df_valid_dict.values()
        ]
        if len(group_by) <= 0:
            val_dates = val_dates[0]

        df_val_item_metrics["item_id"] = df_val_item_metrics["target"]
        df_val_item_metrics.index = df_val_item_metrics["item_id"]

        data = {
            "predict": [
                [
                    {
                        "name": target,
                        "group": group,
                        "value": {
                            "data": {
                                "date": df_valid_dict[group].index.strftime("%Y-%m-%d %H:%M:%S")[-4 * prediction_length:].tolist(),
                                "value": df_valid_dict[group][target][-4 * prediction_length:].tolist()
                            },
                            "prediction": {
                                "date": df_group_target_predictions.sort_values(by="date")["date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                                "min": df_group_target_predictions.sort_values(by="date")["q5"].tolist(),
                                "median": df_group_target_predictions.sort_values(by="date")["q50"].tolist(),
                                "max": df_group_target_predictions.sort_values(by="date")["q95"].tolist()
                            }
                        }
                    }
                    for target, df_group_target_predictions in df_group_predictions.groupby("target")
                ]
                for group, df_group_predictions in predictions["predictions"].items()
            ],
            "evaluate": {
                "dates": val_dates,
                "values": [
                    [
                        {
                            "q5": df_group_target_predictions.sort_values(by="date")["q5"].tolist(),
                            "q50": df_group_target_predictions.sort_values(by="date")["q50"].tolist(),
                            "q95": df_group_target_predictions.sort_values(by="date")["q95"].tolist(),
                        }
                        for _, df_group_target_predictions in df_group_predictions.groupby("target")
                    ]
                    for df_group_predictions in validations["predictions"].values()
                ],
                "agg_metrics": None, # Not used in the frontend, and not compatible with multivariate
                "item_metrics": df_val_item_metrics.to_dict()
            }
        }
        # --------------------

        runtime = time.time() - start + total_trials_times

        resultPredict = {
            "status": "SUCCESS",
            "messenger": "",
            "data_2.0": {
                "predict": df_predictions,
                "validation": {
                    "predict": df_val_predictions,
                    "agg_metrics": validations["agg_metrics"],
                    "item_metrics": df_val_item_metrics
                },
            },
            "data": data, # FIXME Legacy
            "validations": [{"name": x.name, "level": x.level, "message": x.message} for x in failed_checks],
            "runtime": runtime,
        }

        return resultPredict

