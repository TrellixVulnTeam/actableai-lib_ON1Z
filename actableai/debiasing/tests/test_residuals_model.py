import shutil
import pytest

import numpy as np
import pandas as pd

from actableai.debiasing.residuals_model import ResidualsModel
from actableai.utils.testing import unittest_hyperparameters


class TestLogLoss:
    def test_simple_sample_log_loss(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

        y_true = ["spam", "ham", "ham", "spam"]
        y_pred = [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]]

        log_loss = ResidualsModel._per_sample_log_loss(y_true, y_pred)

        assert log_loss is not None  # Checking returned Value
        assert log_loss.dtype == "float64"  # Type of returned Value
        assert np.isclose(
            log_loss,
            np.array(
                [
                    0.10536051565782628,
                    0.10536051565782628,
                    0.2231435513142097,
                    0.4307829160924542,
                ]
            ),
        ).all()  # Checking returned Value


class TestResidualsModel:
    def test_numeric_make_residuals(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )
        biased_groups = ["x", "y"]
        debiased_features = ["z", "t"]
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )
        df_residuals, residuals_features, _ = model.predict(df)

        residuals_features_list = list(residuals_features.keys())

        assert df_residuals is not None  # Value return check
        assert residuals_features is not None
        assert np.all(
            [x == "float64" for x in df_residuals[residuals_features_list].dtypes]
        )  # Type Checking of columns

    def test_categorical_none_make_residuals(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": ["a", "b", "c", None, "b", "c", "a", "b", "c", "a"] * 2,
                "y": ["b", "a", "b", "a", None, "a", "b", "a", "b", "a"] * 2,
                "z": ["a", "b", "a", "b", "a", None, "a", "b", "a", "b"] * 2,
                "t": ["a", "b", "c", "b", "c", "b", None, "b", "c", "b"] * 2,
            },
            dtype="category",
        )
        biased_groups = ["x", "y"]
        debiased_features = ["z", "t"]
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )
        df_residuals, residuals_features, _ = model.predict(df)

        residuals_features_list = list(residuals_features.keys())

        assert df_residuals is not None  # Value return check
        assert residuals_features is not None
        assert np.all(
            [x == "float64" for x in df_residuals[residuals_features_list].dtypes]
        )  # Type Checking of columns

    def test_datetime_make_residuals(self, tmp_path):
        from datetime import datetime

        now = datetime.now()
        df = pd.DataFrame(
            {
                "x": [now, now, now, now, now, now, now, now, now, now] * 2,
                "y": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 2,
                "z": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 2,
                "t": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 2,
            }
        )

        biased_groups = ["x", "y"]
        debiased_features = [
            "z",
            "t",
        ]  # Note z and t can't be datetime because it uses TabularPredictor
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )
        df_residuals, residuals_features, _ = model.predict(df)

        residuals_features_list = list(residuals_features.keys())

        assert df_residuals is not None  # Value return check
        assert residuals_features is not None
        assert np.all(
            [x == "float64" for x in df_residuals[residuals_features_list].dtypes]
        )  # Type Checking of columns

    def test_make_residuals_persist(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )
        biased_groups = ["x", "y"]
        debiased_features = ["z", "t"]
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )
        model.persist_models()

        shutil.rmtree(tmp_path)

        df_residuals, residuals_features, _ = model.predict(df)
        residuals_features_list = list(residuals_features.keys())

        assert df_residuals is not None  # Value return check
        assert residuals_features is not None
        assert np.all(
            [x == "float64" for x in df_residuals[residuals_features_list].dtypes]
        )  # Type Checking of columns

    def test_make_residuals_unpersist(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )
        biased_groups = ["x", "y"]
        debiased_features = ["z", "t"]
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )
        model.unpersist_models()

        shutil.rmtree(tmp_path)

        with pytest.raises(Exception):
            df_residuals, residuals_features, _ = model.predict(df)

    def test_make_residuals_save_load(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            }
        )
        biased_groups = ["x", "y"]
        debiased_features = ["z", "t"]
        model_directory = tmp_path

        model = ResidualsModel(model_directory, biased_groups, debiased_features)
        model.fit(
            df,
            hyperparameters=unittest_hyperparameters(),
            presets="medium_quality_faster_train",
        )

        path = model.save()
        loaded_model = ResidualsModel.load(path=path)

        df_residuals, residuals_features, _ = loaded_model.predict(df)
        residuals_features_list = list(residuals_features.keys())

        assert df_residuals is not None  # Value return check
        assert residuals_features is not None
        assert np.all(
            [x == "float64" for x in df_residuals[residuals_features_list].dtypes]
        )  # Type Checking of columns
