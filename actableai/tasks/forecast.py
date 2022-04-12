from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIForecastTask(AAITask):
    """
    Forecast (time series) Task
    """

    @AAITask.run_with_ray_remote(TaskType.FORECAST)
    def run(
        self,
        df,
        date_column,
        predicted_columns,
        prediction_length,
        RAY_CPU_PER_TRIAL=3,
        RAY_GPU_PER_TRIAL=0,
        RAY_MAX_CONCURRENT=3,
        epochs="auto",
        num_cells="auto",
        num_layers="auto",
        dropout_rate="auto",
        learning_rate="auto",
        trials=1,
        univariate_model_params=None,
        multivariate_model_params=None,
        use_ray=True,
        seed=123,
    ):
        import time
        import torch
        import mxnet as mx
        import numpy as np
        import pandas as pd
        from actableai.timeseries import params
        from actableai.timeseries.models import AAITimeseriesForecaster
        from actableai.data_validation.params import TimeSeriesDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.timeseries.util import handle_datetime_column
        from actableai.utils.sanitize import sanitize_timezone

        np.random.seed(seed)

        pd.set_option("chained_assignment", "warn")
        start_time = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        data_validation_results = TimeSeriesDataValidator().validate(
            date_column, predicted_columns, df, prediction_length
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

        pd_date, _ = handle_datetime_column(df[date_column])
        data = df[predicted_columns]
        data.dropna(how="all", axis=1, inplace=True)
        data.index = pd_date
        data.sort_index(inplace=True)

        mx_ctx = mx.gpu() if RAY_GPU_PER_TRIAL > 0 else mx.cpu()
        torch_device = torch.device("cuda" if RAY_GPU_PER_TRIAL > 0 else "cpu")

        if univariate_model_params is None:
            univariate_model_params = [
                params.FeedForwardParams(
                    hidden_layer_size=(1, 20),
                    epochs=(5, 20),
                    mean_scaling=True,
                    context_length=(prediction_length, 2 * prediction_length),
                ),
                params.ProphetParams(),
                params.RForecastParams(),
            ]
            if df.shape[0] >= 1000:
                univariate_model_params.append(
                    params.DeepARParams(
                        num_cells=(1, 20),
                        num_layers=(1, 3),
                        epochs=(5, 20),
                        scaling=True,
                        context_length=(prediction_length, 2 * prediction_length),
                    )
                )

        if multivariate_model_params is None:
            multivariate_model_params = [
                params.FeedForwardParams(
                    hidden_layer_size=(1, 20),
                    epochs=(5, 20),
                    mean_scaling=False,
                    context_length=(prediction_length, 2 * prediction_length),
                ),
                params.DeepVARParams(
                    epochs=(5, 20),
                    num_layers=(1, 3),
                    num_cells=(1, 20),
                    scaling=False,
                    context_length=(prediction_length, 2 * prediction_length),
                ),
                params.TransformerTempFlowParams(
                    context_length=(prediction_length, 2 * prediction_length),
                    epochs=(5, 20),
                    scaling=False,
                ),
            ]

        m = AAITimeseriesForecaster(
            prediction_length,
            mx_ctx,
            torch_device,
            univariate_model_params=univariate_model_params,
            multivariate_model_params=multivariate_model_params,
        )

        m.fit(
            data,
            trials=trials,
            loss="mean_wQuantileLoss",
            tune_params={
                "resources_per_trial": {
                    "cpu": RAY_CPU_PER_TRIAL,
                    "gpu": RAY_GPU_PER_TRIAL,
                },
                "raise_on_failed_trial": False,
            },
            max_concurrent=RAY_MAX_CONCURRENT,
            eval_samples=3,
            use_ray=use_ray,
            seed=seed,
        )

        total_trials_times = m.total_trial_time
        start = time.time()

        predictions = m.predict(data)
        previous_dates = pd_date.dt.strftime("%Y-%m-%d %H:%M:%S")[
            -4 * prediction_length :
        ].tolist()
        ret_data = [
            [
                {
                    "name": col,
                    "value": {
                        "data": {
                            "date": previous_dates,
                            "value": df[col][-4 * prediction_length :].tolist(),
                        },
                        "prediction": {
                            "date": predictions["date"],
                            "min": val["q5"].tolist(),
                            "median": val["q50"].tolist(),
                            "max": val["q95"].tolist(),
                        },
                    },
                }
                for col, val in zip(predicted_columns, vals)
            ]
            for vals in predictions["values"]
        ]
        validations = m.score(data)

        val_values = [
            [
                {
                    "q5": p["q5"].tolist(),
                    "q50": p["q50"].tolist(),
                    "q95": p["q95"].tolist(),
                }
                for p in predictions
            ]
            for predictions in validations["values"]
        ]

        runtime = time.time() - start + total_trials_times

        resultPredict = {
            "status": "SUCCESS",
            "messenger": "",
            "data": {
                "predict": ret_data,
                "evaluate": {
                    "dates": validations["dates"],
                    "values": val_values,
                    "agg_metrics": validations["agg_metrics"],
                    "item_metrics": validations["item_metrics"].to_dict(),
                },
            },
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "runtime": runtime,
        }

        return resultPredict
