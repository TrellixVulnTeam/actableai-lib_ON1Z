import pytest
import numpy as np

from actableai.tasks.forecast import AAIForecastTask
from actableai.data_validation.base import *
from actableai.timeseries import params

np.random.seed(42)


@pytest.fixture(scope="function")
def forecast_task():
    yield AAIForecastTask(use_ray=False)


@pytest.fixture(scope="function")
def date_range():
    yield pd.date_range("2015-02-24", periods=30, freq="T")


def run_forecast_task(forecast_task, prediction_length, *args, **kwargs):
    univariate_model_params = [
        params.FeedForwardParams(
            hidden_layer_size=(1, 2),
            epochs=(5, 6),
            mean_scaling=True,
            context_length=(prediction_length, 2 * prediction_length),
        )
    ]
    multivariate_model_params = [
        params.FeedForwardParams(
            hidden_layer_size=(1, 2),
            epochs=(5, 6),
            mean_scaling=False,
            context_length=(prediction_length, 2 * prediction_length),
        )
    ]

    results = forecast_task.run(
        use_ray=False,
        prediction_length=prediction_length,
        univariate_model_params=univariate_model_params,
        multivariate_model_params=multivariate_model_params,
        trials=1,
        *args,
        **kwargs
    )

    return results


class TestTimeSeries:
    def test_univariate(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "SUCCESS"

    def test_multivariate(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val", "Val2"])

        assert r["status"] == "SUCCESS"

    def test_mix_target_column(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": ["a", "a", "a", 0.1, "b", "b", 1, "b", "b", "b"] * 3,
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_date_column(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": ["a", "a", "a", 0.1, "b", "b", 1, "b", "b", "b"] * 3,
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Val", ["Val2"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsDatetimeChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_insufficent_data(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range[:10],
                "Val": ["a", "a", "a", 0.1, "b", "b", 1, "b", "b", "b"],
                "Val2": np.random.randn(10),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Val", ["Val2"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_unsorted_datetime(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )
        df = df.sample(frac=1)

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "SUCCESS"

    def test_invalid_prediction_length(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 10, df, "Date", ["Val"])

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsValidPredictionLengthChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_cat_feature(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": ["a", "a", "a", "c", "b", "b", "c", "b", "b", "b"] * 3,
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "CategoryChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_column(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val3"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "ColumnsExistChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_empty_columns(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": [None] * len(date_range),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val", "Val2"])

        assert r["status"] == "SUCCESS"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainEmptyColumnsChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING

    def test_int_feature(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randint(1, len(date_range), len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val", "Val2"])

        assert r["status"] == "SUCCESS"

    def test_invalid_frequency(self, forecast_task, date_range):
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )
        df = df.append(df).sort_index()

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsValidFrequencyChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_ydm_series(self, forecast_task, date_range):
        ymd_series = pd.Series(
            pd.date_range("2015-01-01", periods=30, freq="MS").astype(str).values
        )
        ydm_series = pd.to_datetime(ymd_series, format="%Y-%d-%m").astype(str)
        df = pd.DataFrame(
            {
                "Date": ydm_series,
                "Val": np.random.randn(len(date_range)),
                "Val2": np.random.randn(len(date_range)),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        print(r)
        assert r["status"] == "SUCCESS"

    def test_mixed_time_format_series(self, forecast_task, date_range):
        mix_dt_series = pd.Series(
            pd.date_range("2015-03-24", periods=30, freq="55T").astype(str).values
        )
        drop_indices = np.random.randint(1, 30, 5)
        mix_dt_series.iloc[drop_indices,] = pd.to_datetime(
            mix_dt_series.iloc[
                drop_indices,
            ]
        ).dt.strftime("%Y-%m-%d %H:%M")
        mix_dt_series = mix_dt_series.astype(str)
        df = pd.DataFrame(
            {
                "Date": mix_dt_series,
                "Val": np.random.randn(30),
                "Val2": np.random.randn(30),
            }
        )

        r = run_forecast_task(forecast_task, 1, df, "Date", ["Val"])

        assert r["status"] == "SUCCESS"
