import pytest
import ray
import torch
import numpy as np
import pandas as pd
import mxnet as mx
from actableai.timeseries import params
from actableai.timeseries.models import AAITimeseriesForecaster
from actableai.utils.testing import init_ray

rng = pd.date_range("2015-02-24", periods=30, freq="T")
np.random.seed(12)

mx_ctx = mx.cpu()
torch_device = torch.device("cpu")


def predict_univariate(
    df, target_col, feature_col, prediction_length, param, use_ray=True
):
    pd_date = pd.to_datetime(df[target_col])
    data = df[feature_col]
    data.dropna(how="all", axis=1, inplace=True)
    data.index = pd_date
    data.sort_index(inplace=True)

    m = AAITimeseriesForecaster(
        prediction_length,
        mx_ctx,
        torch_device,
        univariate_model_params=param,
    )

    ray_shutdown = False
    if use_ray and not ray.is_initialized():
        init_ray()
        ray_shutdown = True

    m.fit(
        data,
        trials=1,
        loss="mean_wQuantileLoss",
        tune_params={
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0,
            },
            "raise_on_failed_trial": False,
        },
        max_concurrent=4,
        eval_samples=3,
        use_ray=use_ray,
    )

    if ray_shutdown:
        ray.shutdown()

    predictions = m.predict(data)
    validations = m.score(data)

    return predictions, validations


def predict_multivariate(
    df, target_col, feature_col, prediction_length, param=None, use_ray=True
):
    pd_date = pd.to_datetime(df[target_col])
    data = df[feature_col]
    data.dropna(how="all", axis=1, inplace=True)
    data.index = pd_date
    data.sort_index(inplace=True)

    m = AAITimeseriesForecaster(
        prediction_length,
        mx_ctx,
        torch_device,
        multivariate_model_params=param,
    )

    ray_shutdown = False
    if use_ray and not ray.is_initialized():
        init_ray()
        ray_shutdown = True

    m.fit(
        data,
        trials=1,
        loss="mean_wQuantileLoss",
        tune_params={
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0,
            },
            "raise_on_failed_trial": False,
        },
        max_concurrent=4,
        eval_samples=3,
        use_ray=use_ray,
    )

    if ray_shutdown:
        ray.shutdown()

    predictions = m.predict(data)
    validations = m.score(data)

    return predictions, validations


@pytest.fixture
def simple_dataframe():
    return pd.DataFrame(
        {
            "Date": rng,
            "Val": np.random.randn(len(rng)),
            "Val2": np.random.randn(len(rng)),
        }
    )


@pytest.mark.parametrize("use_ray", [True, False])
class TestAAITimeseriesForecaster:
    def test_univariate_prophet(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [params.ProphetParams()]

        predictions, validations = predict_univariate(
            df, "Date", ["Val"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_univariate_feedforward(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [
            params.FeedForwardParams(hidden_layer_size=1, epochs=2, context_length=None)
        ]

        predictions, validations = predict_univariate(
            df, "Date", ["Val"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_univariate_rforecast(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [params.RForecastParams()]

        predictions, validations = predict_univariate(
            df, "Date", ["Val"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_univariate_deepar(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [
            params.DeepARParams(
                num_cells=1, num_layers=1, epochs=2, context_length=None
            )
        ]

        predictions, validations = predict_univariate(
            df, "Date", ["Val"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_multivariate_feedforward(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [
            params.FeedForwardParams(hidden_layer_size=1, epochs=2, context_length=None)
        ]

        predictions, validations = predict_multivariate(
            df, "Date", ["Val", "Val2"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_multivariate_deepvar(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [
            params.DeepVARParams(
                epochs=2, num_layers=1, num_cells=1, context_length=None
            )
        ]

        predictions, validations = predict_multivariate(
            df, "Date", ["Val", "Val2"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None

    def test_multivariate_transformer(self, use_ray, simple_dataframe):
        df = simple_dataframe

        param = [params.TransformerTempFlowParams(context_length=None, epochs=2)]

        predictions, validations = predict_multivariate(
            df, "Date", ["Val", "Val2"], 1, param, use_ray=use_ray
        )

        assert predictions.get("date") is not None
        assert predictions.get("values") is not None
        assert validations.get("dates") is not None
        assert validations.get("values") is not None
