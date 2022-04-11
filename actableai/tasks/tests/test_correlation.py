import pytest
import numpy as np
from actableai.tasks.correlation import AAICorrelationTask
from actableai.data_validation.base import *


@pytest.fixture(scope="function")
def correlation_task():
    yield AAICorrelationTask(use_ray=False)


class TestRemoteCorrelation:
    def test_continuous_vs_continuous(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]
        assert "type" in r["data"]["charts"][0]
        assert "data" in r["data"]["charts"][0]
        assert "coef" in r["data"]["charts"][0]["data"]
        assert "intercept" in r["data"]["charts"][0]["data"]
        assert "x" in r["data"]["charts"][0]["data"]
        assert "y" in r["data"]["charts"][0]["data"]
        assert "x_pred" in r["data"]["charts"][0]["data"]
        assert "r2" in r["data"]["charts"][0]["data"]
        assert "y_mean" in r["data"]["charts"][0]["data"]
        assert "y_std" in r["data"]["charts"][0]["data"]
        assert "x_label" in r["data"]["charts"][0]["data"]
        assert "y_label" in r["data"]["charts"][0]["data"]
        assert "corr" in r["data"]["charts"][0]


    def test_continuous_vs_continuous_nan(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_continuous_vs_continuous_nan_target(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_continuous_vs_categorical(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_continuous_vs_categorical_nan(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": ["a", "a", None, "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_categorical_vs_continuous(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_categorical_vs_continuous_nan(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", None, "b", "b", "b", "b", "b"] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_categorical_vs_categorical(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_categorical_vs_categorical_nan(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            "y": ["a", "a", "a", "a", None, "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_categorical_vs_categorical_nan_target(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", None, "b", "b", "b", "b", "b"] * 5,
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]


    def test_continuous_vs_mix(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_categorical_vs_mix(self, correlation_task):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "x", target_value="a")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_chart_sorted_by_corr(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "charts" in r["data"]
        assert all(abs(r["data"]["charts"][i]["corr"]) >= abs(r["data"]["charts"][i + 1]["corr"]) \
                   for i in range(0, len(r["data"]["charts"]) - 1))

    def test_target_col_is_datetime(self, correlation_task):
        rng = pd.date_range('2015-02-24', periods=20, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Date': rng,
            'Val': np.random.randn(len(rng)),
            'Val2': np.random.randn(len(rng)),
        })

        r = correlation_task.run(df, "Date")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_feat_col_is_datetime(self, correlation_task):
        rng = pd.date_range('2015-02-24', periods=20, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Date': rng,
            'Val': np.random.randn(len(rng)),
            'Val2': np.random.randn(len(rng)),
        })

        r = correlation_task.run(df, "Val")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_empty_columns(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            'z': [None] * 50
        })

        r = correlation_task.run(df, "x")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]
        assert len(r["validations"]) > 0
        assert r["validations"][0]["level"] == CheckLevels.WARNING

    def test_empty_target(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 5,
            'z': [None] * 50
        })

        r = correlation_task.run(df, "z")

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert r["validations"][1]["level"] == CheckLevels.CRITICAL

    def test_control(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 3, 5, 4, 9] * 10,
            "y": [1, 1, 1, 1, 2, 2, 3, 3, 3, 3] * 10,
            "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 10
        })

        r = correlation_task.run(df, "x", control_columns=["y"], control_values=[None])

        print(r)
        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_use_bonferroni(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "z": [5, 2, 6, 4, 1, 3, 7, 8, 9, 10],
        })

        r = correlation_task.run(df, target_column="x", use_bonferroni=False)
        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert len(r["data"]["corr"]) == 2

        r = correlation_task.run(df, target_column="x", use_bonferroni=True)
        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert len(r["data"]["corr"]) == 1

    def test_target_value_in_row_contains_nan(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, None, 8, 9, 10] * 5,
            "y": [1, 2, 3, 4, 5, 6, None, 8, 9, 10] * 5,
            "z": ["a", "a", "a", "a", "a", "b", "c", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "z", "c")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_nan_in_categorical_column(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, None, 8, 9, 10] * 5,
            "y": ["a", "a", "a", None, "a", "b", "c", "b", "b", "b"] * 5,
            "z": ["a", "a", "a", 'a', "a", "b", "c", "b", "b", "b"] * 5
        })

        r = correlation_task.run(df, "z", "c")

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert "charts" in r["data"]

    def test_top_k(self, correlation_task):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2
        })

        r = correlation_task.run(df, "x", top_k=1)

        assert r["status"] == "SUCCESS"
        assert "corr" in r["data"]
        assert len(r["data"]["corr"])==1
        assert len(r["data"]["charts"])==1
