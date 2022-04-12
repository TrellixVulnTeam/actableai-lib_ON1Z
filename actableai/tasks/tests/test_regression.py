import numpy as np
import pandas as pd
import pytest

from actableai.tasks.regression import AAIRegressionTask
from actableai.data_validation.base import *
from actableai.utils.testing import unittest_hyperparameters
from actableai.utils.dataset_generator import DatasetGenerator
from actableai.regression.quantile import ag_quantile_hyperparameters


@pytest.fixture(scope="function")
def regression_task():
    yield AAIRegressionTask(use_ray=False)


@pytest.fixture(scope="function")
def data():
    yield pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
        }
    )


def run_regression_task(regression_task: AAIRegressionTask, tmp_path, *args, **kwargs):
    if "hyperparameters" not in kwargs:
        kwargs["hyperparameters"] = unittest_hyperparameters()

    if "drop_duplicates" not in kwargs:
        kwargs["drop_duplicates"] = False

    return regression_task.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        residuals_hyperparameters=unittest_hyperparameters()
    )


class TestRemoteRegression:
    def test_num_vs_num(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_datetime(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {"x": [now()] * 20, "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2}
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", features=["x"], validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_mixed_datetime(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {
                "x": [now()] * 20,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "z",
            features=["x", "y"],
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_datetime_target(self, regression_task, tmp_path):
        from datetime import datetime

        now = datetime.now
        df = pd.DataFrame(
            {"x": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2, "y": [now()] * 20}
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", features=["x"], validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["data"]) == 0
        assert r["validations"][0]["name"] == "IsNumericalChecker"

    def test_num_vs_categorical(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_num_vs_mix(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_complex_text(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, None, 3, 4, 4] * 5,
                "z": ["good", "good", "bad", "bad", "freaking bad", "freaking bad"] * 5,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_feature_missing_value(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, None, 3, 4, 4] * 5,
                "z": ["good", "good", np.nan, np.nan, "freaking bad", "freaking bad"]
                * 5,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_hyperparam_multimodal(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 2, 2, 3, 3, 3, 7, 7, 7] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_autogluon_multiclass_case(self, regression_task, tmp_path):
        """
        AutoGluon infers your prediction problem is: 'multiclass' (because dtype of label-column == int, but few unique label-values observed).
        """
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3] * 100,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "x")

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_mixed_feature_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, "abc xyz", 7.1, 8, 9, 10] * 2,
                "x2": [1, 2, 3, 4, 5, "6", 7, 8, 9, 10] * 2,
                "y": [1, 2, None, None, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert "leaderboard" in r["data"]

    def test_mixed_target_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, "5", None, None, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsNumericalChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_insufficient_data(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [1, 2, 3, 4, "5", None, None, 8, 9, 10],
            }
        )

        r = run_regression_task(regression_task, tmp_path, df, "y")

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_eval_metric(self, regression_task, tmp_path, data):
        r = run_regression_task(regression_task, tmp_path, data, "y", eval_metric="abc")
        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "RegressionEvalMetricChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_validation_has_prediction(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 2, 2, 3, 3, 3, 7, 7, 7] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            prediction_quantile_low=5,
            prediction_quantile_high=95,
            validation_ratio=0.2,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "x_low" in r["data"]["validation_table"].columns
        assert "x_high" in r["data"]["validation_table"].columns
        assert "leaderboard" in r["data"]

    def test_invalid_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "z", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "ColumnsExistChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_empty_feature_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [None, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "y", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "prediction_table" in r["data"]
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainEmptyColumnsChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert "leaderboard" in r["data"]

    def test_empty_target_column(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [None, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainEmptyColumnsChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_suggest_analytic(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"] * 2,
            }
        )

        r = run_regression_task(
            regression_task, tmp_path, df, "x", validation_ratio=0.2
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "prediction_table" in r["data"]
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "CorrectAnalyticChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING
        assert "leaderboard" in r["data"]

    def test_explain_samples(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["validation_shaps"]) > 0
        assert "leaderboard" in r["data"]

    def test_explain_samples_with_datetime(self, regression_task, tmp_path):
        rng = pd.date_range("2015-02-24", periods=20, freq="T")
        drop_indices = np.random.randint(0, 20, 5)
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": rng,
            }
        )
        df.iloc[drop_indices, :] = None
        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            validation_ratio=0.2,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["validation_shaps"]) > 0
        assert "leaderboard" in r["data"]

    def test_drop_duplicates(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ]
                * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 4,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "leaderboard" in r["data"]

    def test_drop_duplicates_insufficient(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "y",
            ["x"],
            validation_ratio=0.2,
            drop_duplicates=True,
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL


class TestRemoteRegressionCrossValidation:
    def test_cross_val(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4,
        )

        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "RMSE" in evaluate
        assert "RMSE_std_err" in evaluate
        assert "R2" in evaluate
        assert "R2_std_err" in evaluate
        assert "MAE" in evaluate
        assert "MAE_std_err" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
        assert "leaderboard" in r["data"]

    def test_cross_val_with_explain(self, regression_task, tmp_path):
        rng = pd.date_range("2015-02-24", periods=20, freq="T")
        drop_indices = np.random.randint(0, 20, 5)
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": rng,
            }
        )
        df.iloc[drop_indices, :] = None
        r = run_regression_task(
            regression_task, tmp_path, df, "x", kfolds=4, explain_samples=True
        )

        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["prediction_table"]) > 0
        assert "RMSE" in evaluate
        assert "RMSE_std_err" in evaluate
        assert "R2" in evaluate
        assert "R2_std_err" in evaluate
        assert "MAE" in evaluate
        assert "MAE_std_err" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
        assert "leaderboard" in r["data"]

    def test_with_quantile(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=1,
            prediction_quantile_high=95,
            prediction_quantile_low=5,
            hyperparameters=ag_quantile_hyperparameters(),
        )

        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "x_predicted" in r["data"]["prediction_table"].columns
        assert "x_low" in r["data"]["prediction_table"].columns
        assert "x_high" in r["data"]["prediction_table"].columns
        assert "x_predicted" in r["data"]["validation_table"].columns
        assert "x_low" in r["data"]["validation_table"].columns
        assert "x_high" in r["data"]["validation_table"].columns
        assert "RMSE" in evaluate
        assert "R2" in evaluate
        assert "MAE" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
        assert "leaderboard" in r["data"]

    def test_cross_val_with_quantiles(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            prediction_quantile_high=95,
            prediction_quantile_low=5,
            hyperparameters=ag_quantile_hyperparameters(),
        )

        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert r["status"] == "SUCCESS"
        assert len(r["data"]["prediction_table"]) > 0
        assert "x_predicted" in r["data"]["prediction_table"].columns
        assert "x_low" in r["data"]["prediction_table"].columns
        assert "x_high" in r["data"]["prediction_table"].columns
        assert "RMSE" in evaluate
        assert "RMSE_std_err" in evaluate
        assert "R2" in evaluate
        assert "R2_std_err" in evaluate
        assert "MAE" in evaluate
        assert "MAE_std_err" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat
        assert "leaderboard" in r["data"]

    def test_debiasing_feature(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 2,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            validation_ratio=0.2,
            kfolds=2,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list


class TestIntervention:
    @pytest.mark.parametrize("cate_alpha", [None, 0.05])
    def test_intervention_numeric(self, regression_task, tmp_path, cate_alpha):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [2, 2, 2, 2, None, 3, None, 3, 4, 4] * 2,
            }
        )
        df["new_intervention"] = df["current_intervention"] * 2

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            ["y", "current_intervention"],
            current_intervention_column="current_intervention",
            new_intervention_column="new_intervention",
            causal_hyperparameters=unittest_hyperparameters(),
            cate_alpha=cate_alpha,
        )

        assert r["status"] == "SUCCESS"
        assert "intervention_table" in r["data"]
        assert "leaderboard" in r["data"]

    @pytest.mark.parametrize("cate_alpha", [None, 0.05])
    def test_intervention_categorical(self, regression_task, tmp_path, cate_alpha):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "current_intervention": [
                    "a",
                    None,
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                ]
                * 2,
            }
        )
        df["new_intervention"] = [
            "b",
            "b",
            "b",
            "b",
            "b",
            None,
            None,
            "a",
            "a",
            "a",
        ] * 2

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            ["y", "current_intervention"],
            current_intervention_column="current_intervention",
            new_intervention_column="new_intervention",
            causal_hyperparameters=unittest_hyperparameters(),
            cate_alpha=cate_alpha,
        )

        assert r["status"] == "SUCCESS"
        assert "intervention_table" in r["data"]
        assert "leaderboard" in r["data"]

    @pytest.mark.parametrize("cate_alpha", [None, 0.05])
    def test_intervention_with_common_causes(
        self, regression_task, tmp_path, cate_alpha
    ):
        df = pd.DataFrame(
            {
                "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": ["+", "-", "+", "-", "+", "-", "+", "-", "+", "-"] * 2,
                "current_intervention": [
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    None,
                    "b",
                    "b",
                ]
                * 2,
            }
        )
        df["new_intervention"] = [
            "b",
            "b",
            "b",
            "b",
            None,
            None,
            "a",
            "a",
            "a",
            "a",
        ] * 2

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            "x",
            ["y", "z", "current_intervention"],
            current_intervention_column="current_intervention",
            new_intervention_column="new_intervention",
            causal_hyperparameters=unittest_hyperparameters(),
            common_causes=["y", "z"],
            cate_alpha=cate_alpha,
        )

        assert r["status"] == "SUCCESS"
        assert "intervention_table" in r["data"]
        assert "leaderboard" in r["data"]


class TestDebiasing:
    def test_simple_debiasing_feature(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_debiasing_text_column(self, regression_task, tmp_path):
        df = DatasetGenerator.generate(
            columns_parameters=[
                {"name": "x", "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2},
                {
                    "name": "y",
                    "values": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                },
                {"name": "z", "type": "text", "word_range": (5, 10)},
                {"name": "t", "values": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 2},
            ],
            rows=20,
            random_state=0,
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][-1]["name"] == "DoNotContainTextChecker"
        assert r["validations"][-1]["level"] == CheckLevels.CRITICAL

    def test_mixed_debiasing_feature_cat(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "a", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_mixed_debiasing_feature_num(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": ["a", "a", "a", "a", "a", None, "b", "b", "b", "b"] * 3,
                "z": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list

    def test_simple_debiasing_feature_explain_samples(self, regression_task, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_regression_task(
            regression_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            prediction_quantile_low=None,
            prediction_quantile_high=None,
            explain_samples=True,
        )

        assert r["status"] == "SUCCESS"
        assert "validation_table" in r["data"]
        assert "prediction_table" in r["data"]
        assert "predict_shaps" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_shaps" in r["data"]
        assert "importantFeatures" in r["data"]
        assert "debiasing_charts" in r["data"]
        assert "leaderboard" in r["data"]

        assert len(r["data"]["predict_shaps"]) > 0
        assert len(r["data"]["validation_shaps"]) > 0

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["kde", "scatter"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "x" in chart
                assert type(chart["x"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "kde":
                    assert "lines" in chart
                    assert type(chart["lines"]) is list

                    for line in chart["lines"]:
                        assert "y" in line
                        assert type(line["y"]) is list

                        assert "name" in line
                else:
                    assert "y" in chart
                    assert type(chart["y"]) is list
