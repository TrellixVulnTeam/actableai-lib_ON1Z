import shutil
import pytest

import pandas as pd

from autogluon.tabular import TabularPredictor

from actableai.debiasing.debiasing_model import DebiasingModel
from actableai.utils.testing import unittest_hyperparameters


class TestDebiasingModel:
    def test_simple_numeric_regression(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_simple_numeric_classification(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(label=target, path=tmp_path, problem_type="binary")
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_mixed_num_regression(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_mixed_num_classification(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(label=target, path=tmp_path, problem_type="binary")
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_mixed_cat_regression(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_mixed_cat_classification(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(label=target, path=tmp_path, problem_type="binary")
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        assert predictor._learner.is_fit

    def test_persist(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )
        predictor.persist_models()

        shutil.rmtree(tmp_path)

        results = predictor.predict(df)
        assert results is not None

    def test_unpersist(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )
        predictor.unpersist_models()

        shutil.rmtree(tmp_path)

        with pytest.raises(Exception):
            predictor.predict(df)

    def test_save_load(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 10,
            }
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        ag_args_fit = {
            "drop_duplicates": True,
            "label": target,
            "biased_groups": biased_groups,
            "debiased_features": debiased_features,
            "hyperparameters_residuals": unittest_hyperparameters(),
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": unittest_hyperparameters(),
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }
        hyperparameters = {DebiasingModel: {}}

        predictor = TabularPredictor(
            label=target, path=tmp_path, problem_type="regression"
        )
        predictor.fit(
            train_data=df,
            hyperparameters=hyperparameters,
            presets="medium_quality_faster_train",
            ag_args_fit=ag_args_fit,
        )

        predictor.save()
        new_predictor = TabularPredictor.load(path=tmp_path)

        results = new_predictor.predict(df)
        assert results is not None
