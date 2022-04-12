import pytest
import pandas as pd

from actableai.tasks.bayesian_regression import AAIBayesianRegressionTask


@pytest.fixture(scope="function")
def task():
    yield AAIBayesianRegressionTask(use_ray=False)


@pytest.fixture(scope="function")
def df():
    yield pd.DataFrame(
        {
            "x": [1, 1, 2, 2, None, 3, 4, None, 5, 5] * 2,
            "y": [2, 1, 3, 5, None, 6, None, 7, 8, 9] * 2,
            "z": [1, 2, 3, 4, 5, 6, 7, 8, None, None] * 2,
            "w": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"] * 2,
            "v": ["c", "d", "c", "c", "c", "d", "c", "d", "c", "d"] * 2,
        }
    )


class TestBayesianRegression:
    def test_full_data(self, df, task):
        re = task.run(df, features=["x", "y"], target="z", trials=5)
        assert re["status"] == "SUCCESS"
        assert "validation_table" in re["data"]
        assert "prediction_table" in re["data"]
        assert "evaluation" in re["data"]

        assert "coeffs" in re["data"]

        assert "univariate" in re["data"]["coeffs"]
        assert "multivariate" in re["data"]["coeffs"]

        # Check the multivariate
        for c in re["data"]["coeffs"]["multivariate"]:
            assert "name" in c
            assert "mean" in c
            assert "stds" in c
            assert "pdfs" in c

        # Check the unvariates analysis
        for c in re["data"]["coeffs"]["univariate"]:
            assert "name" in c
            assert "x" in c
            assert "coeffs" in c
            assert "stds" in c
            assert "pdfs" in c
            assert "y_mean" in c
            assert "y_std" in c
            assert "r2" in c
            assert "rmse" in c

    def test_full_data_multivariate_result(self, df, task):
        re = task.run(
            df, features=["x", "y", "w"], target="z", trials=5, polynomial_degree=2
        )
        # print(re)
        assert re["status"] == "SUCCESS"
        assert "validation_table" in re["data"]
        assert "prediction_table" in re["data"]
        assert "evaluation" in re["data"]

        assert [x["name"] for x in re["data"]["coeffs"]["univariate"]] == [
            "x",
            "y",
            "w_a",
            "w_b",
        ]
        assert [x["name"] for x in re["data"]["coeffs"]["multivariate"]] == [
            "x",
            "y",
            "w_a",
            "w_b",
            "x^2",
            "x y",
            "x w_a",
            "x w_b",
            "y^2",
            "y w_a",
            "y w_b",
        ]

        # Checking for univariate values and their fields
        for c in [x for x in re["data"]["coeffs"]["univariate"]]:
            assert "name" in c
            assert "x" in c
            assert "coeffs" in c
            assert "stds" in c
            assert "pdfs" in c
            assert "y_mean" in c
            assert "y_std" in c
            assert "r2" in c
            assert "rmse" in c

        # Checking for multivariate values and their fields
        for c in [x for x in re["data"]["coeffs"]["multivariate"]]:
            assert "name" in c
            assert "mean" in c
            assert "stds" in c
            assert "pdfs" in c

    def test_categorical_features(self, df, task):
        re = task.run(df, features=["w", "v"], target="z", trials=5)
        assert re["status"] == "SUCCESS"

    def test_mix_features(self, df, task):
        re = task.run(df, features=["x", "w"], target="z", trials=5)
        assert re["status"] == "SUCCESS"

    def test_polynomial_degree(self, df, task):
        re = task.run(
            df, features=["x", "y"], target="z", polynomial_degree=3, trials=5
        )
        assert re["status"] == "SUCCESS"
        assert "validation_table" in re["data"]
        assert "prediction_table" in re["data"]
        assert "evaluation" in re["data"]

    def test_no_prediction(self, df, task):
        df["z"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
        re = task.run(
            df, features=["x", "y"], target="z", polynomial_degree=3, trials=5
        )
        assert re["status"] == "SUCCESS"
        assert "prediction_table" not in re["data"]

    def test_no_validation(self, df, task):
        re = task.run(
            df,
            features=["x", "y"],
            target="z",
            polynomial_degree=3,
            trials=5,
            validation_split=0,
        )
        assert re["status"] == "SUCCESS"
        assert "validation_table" not in re["data"]

    def test_priors(self, df, task):
        priors = [
            {"column": "x", "control": None, "degree": 2, "value": 0.2},
            {"column": "w", "control": "a", "degree": 1, "value": 0.1},
        ]
        re = task.run(
            df,
            features=["x", "y", "w"],
            target="z",
            polynomial_degree=3,
            trials=5,
            priors=priors,
        )
        assert re["status"] == "SUCCESS"

    def test_invalid_priors(self, df, task):
        priors = [
            {"column": "x", "control": None, "degree": 2, "value": 0.2},
            {"column": "w", "control": "a", "degree": 2, "value": 0.1},
        ]
        with pytest.raises(ValueError):
            re = task.run(
                df,
                features=["x", "y", "w"],
                target="z",
                polynomial_degree=3,
                trials=5,
                priors=priors,
            )
