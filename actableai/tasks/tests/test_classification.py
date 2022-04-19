import pytest
import pandas as pd

from actableai.tasks.classification import AAIClassificationTask
from actableai.data_validation.checkers import CheckLevels
from actableai.utils.testing import unittest_hyperparameters
from actableai.utils.dataset_generator import DatasetGenerator


@pytest.fixture(scope="function")
def classification_task():
    yield AAIClassificationTask(use_ray=False)


def run_classification_task(classification_task: AAIClassificationTask,
                            tmp_path,
                            *args,
                            drop_duplicates=False,
                            **kwargs):
    return classification_task.run(
        *args,
        **kwargs,
        presets="medium_quality_faster_train",
        model_directory=tmp_path,
        hyperparameters=unittest_hyperparameters(),
        residuals_hyperparameters=unittest_hyperparameters(),
        drop_duplicates=drop_duplicates
    )


class TestRemoteClassification:
    def test_numeric(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ['x'], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0

    def test_ray(self, tmp_path, init_ray):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        task = AAIClassificationTask(use_ray=True)
        r = run_classification_task(
            task, tmp_path, df, "y", ['x'], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0

    def test_categorical(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 2,
            "y": [1, 2, 1, 2, 1, 2, 1, None, 1, 2] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_datetime(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now()
        df = pd.DataFrame({
            "x": [now, now, now, now, now, now, now, now, now, now] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_extra_columns(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
            "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
            "y": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x1"])
        assert r["status"] == "SUCCESS"
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["importantFeatures"]) == 1


    def test_numeric_and_categorical_and_datetime(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now()
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
            "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
            "x3": [now, now, None, now, now, now, now, now, now, now] * 2,
            "y": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1", "x2", "x3"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_multiclass_num(self, classification_task, tmp_path):
        """
        AutoGluon infers your prediction problem is: 'regression' (because dtype of label-column == float and many unique label-values observed).
        """
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 3, 2, 1, None, 1, 2, 1, 3] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_multiclass_cat(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8] * 5,
            "y": ['a', 'a', 'a', 'b', 'b', None, 'c', 'c'] * 5
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_mix_target_column(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now()
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
            "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2] * 2,
            "x3": [now, now, None, now, now, now, now, now, now, now] * 2,
            "y": [1, 2, 1, 2, 1, 2, None, 2, 1, "a"] * 2
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1", "x2", "x3"], validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsCategoricalChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_mix_feature_column(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now()
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", 1, "b", "b", "b"] * 2,
            "x2": [1, 2, 1, 2, 1, 2, 2.1, 2, 1, 2] * 2,
            "x3": [now, now, None, now, now, now, now, now, now, now] * 2,
            "y": ["1", "2", "1", "2", "1", "2", None, "2", "1", "2"] * 2
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_datetime_target(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now
        df = pd.DataFrame({
            "x": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            "y": [now()] * 20
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", features=["x"], validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r['data']) == 0
        assert r['validations'][0]['name'] == 'IsCategoricalChecker'

    def test_insufficient_data(self, classification_task, tmp_path):
        from datetime import datetime
        now = datetime.now()
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"],
            "x2": [1, 2, 1, 2, 1, 2, None, 2, 1, 2],
            "x3": [now, now, None, now, now, now, now, now, now, now],
            "y": [1, 2, 1, 2, 1, 2, None, 2, 1, "a"]
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_column(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "z", ["x"], validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "ColumnsExistChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_empty_column(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": [None, None, None, None, None, None, None, None, None, None] * 2,
            "x2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", ["x1", "x2"], validation_ratio=0.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainEmptyColumnsChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING

    def test_insufficient_valid_data(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 10,
            "y": [1, 2, 1, 3, 1, 2, 3, 2, 1, 2] * 10
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", validation_ratio=.01)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientValidationSampleChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_insufficient_cls_sample(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": ["a", "a", "c"] + ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 2,
            "y": [4, 4, 5] + [1, 2, 1, 2, 1, 2, None, 2, 1, 3] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x1"], validation_ratio=0.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientClassSampleChecker"

    def test_insufficient_class(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": ["a", "a", "a", "a", "b", "b", None, "b", "b", "b"] * 2,
            "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x1"], validation_ratio=.2)

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "IsSufficientNumberOfClassChecker"

    def test_unencoded_binary(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": ['1', '2', '1', '2', '1', None, '1', '2', '1', '2'] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", ["x"], validation_ratio=0.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_suggest_analytic(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 10,
            "y": [1, 2, 3, 2, 1, None, 1, 2, 1, 3, 1] * 10
        })

        r = run_classification_task(classification_task, tmp_path, df, "x", ["y"], validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "CorrectAnalyticChecker"
        assert r["validations"][0]["level"] == CheckLevels.WARNING

    def test_explain_bool_sample_with_nan(self, classification_task, tmp_path, init_ray):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 3, 1, None, 3, 2, None, 2] * 2,
            "z": [True, False, None, True, False] * 4
        })

        r = run_classification_task(
            classification_task, tmp_path, df, "y", validation_ratio=.2, explain_samples=True)

        assert r["status"] == "SUCCESS"
        assert len(r["data"]["validation_explanations"]) > 0
        assert len(r["data"]["predict_explanations"]) > 0

    def test_boolean_str_target_column(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": ['True', 'False', 'True', 'False', 'True', None, 'True', 'False', 'True', 'False'] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_boolean_target_column(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [True, False, True, False, True, None, True, False, True, False] * 2
        })

        r = run_classification_task(classification_task, tmp_path, df, "y", validation_ratio=.2)

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]

    def test_drop_duplicates(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 4
        })

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            'y',
            ['x'],
            validation_ratio=0.2,
            drop_duplicates=True
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0

    def test_drop_duplicates_insufficient(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            'y',
            ['x'],
            validation_ratio=0.2,
            drop_duplicates=True
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r['validations'][0]['name'] == "IsSufficientDataChecker"
        assert r['validations'][0]['level'] == CheckLevels.CRITICAL

class TestRemoteClassificationCrossValidation:
    def test_cross_val(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "z": [2, 2, 2, 2, 2, 3, 3, 3, 4, 4] * 3
        })

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "x",
            kfolds=4,
            cross_validation_max_concurrency=4
        )

        assert r["status"] == "SUCCESS"
        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert len(r["data"]["predictData"]) > 0
        assert "accuracy" in evaluate
        assert "accuracy_std_err" in evaluate
        assert "confusion_matrix" in evaluate
        assert "confusion_matrix_std_err" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat

    # def test_cross_val_with_explain(self):
    # import ray
    # if not ray.is_initialized():
    # ray.init()

    # df = pd.DataFrame({
    # "x": [2,2,2,2,2,None,3,3,4,4] * 3,
    # "y": [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10] * 3,
    # "z": [2,2,2,2,2,3,3,3,4,4] * 3,
    # })
    # r = run_classification_task(classification_task, tmp_path, df, "x", [], use_cross_validation=True, kfolds=5, explain_samples=True)
    # assert r["status"] == "SUCCESS"
    # assert len(r["data"]["predict_explanations"]) > 0

    def test_cross_val_with_text(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 3,
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "z": ["2", "2", "2", "2", "2", "3", "3", "3", "4", "4"] * 3
        })

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "x",
            ["y", "z"],
            kfolds=4,
            cross_validation_max_concurrency=4
        )

        assert r["status"] == "SUCCESS"
        evaluate = r["data"]["evaluate"]
        important_features = r["data"]["importantFeatures"]
        assert len(r["data"]["predictData"]) > 0
        assert "accuracy" in evaluate
        assert "accuracy_std_err" in evaluate
        assert "confusion_matrix" in evaluate
        assert "confusion_matrix_std_err" in evaluate
        for feat in important_features:
            assert "feature" in feat
            assert "importance" in feat
            assert "importance_std_err" in feat

    def test_cross_val_with_text_fail(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x1": ["a", "a", "c"] + ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"] * 3,
            "y": [4, 4, 5] + [1, 2, 1, 2, 1, 2, None, 2, 1, 3] * 3
        })

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            "y",
            ["x1"],
            kfolds=4,
            cross_validation_max_concurrency=4
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) > 0
        assert r["validations"][1]["name"] == "IsSufficientClassSampleForCrossValidationChecker"

    def test_debiasing_feature(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "t": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2
        })
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            validation_ratio=.2,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            kfolds=2
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list


class TestDebiasing:
    def test_simple_debiasing_feature(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3
        })
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list


    def test_debiasing_text_column(self, classification_task, tmp_path):
        df = DatasetGenerator.generate(
            columns_parameters=[
                {
                    "name": "x",
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
                },
                {
                    "name": "y",
                    "values": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 2,
                },
                {
                    "name": "z",
                    "type": "text",
                    "word_range": (5, 10)
                },
                {
                    "name": "t",
                    "values": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 2
                }

            ],
            rows=20,
            random_state=0
        )
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features
        )

        assert r["status"] == "FAILURE"
        assert len(r["validations"]) >= 1
        assert r["validations"][0]["name"] == "DoNotContainTextChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL


    def test_mixed_debiasing_feature_num(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "y": ["a", "a", "a", "a", "b", None, "b", "b", "b", "b"] * 3,
            "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
            "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3
        })
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["x"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list


    def test_mixed_debiasing_feature_cat(self, classification_task, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "y": ["a", "a", "a", "a", "b", None, "b", "b", "b", "b"] * 3,
            "z": [1, 2, 3, None, 5, 6, 7, 8, 9, 10] * 3,
            "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3
        })
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) == 0
        assert len(r["data"]["predict_explanations"]) == 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

    def test_simple_debiasing_feature_explain_samples(self, classification_task, tmp_path, init_ray):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "y": [1, 2, 1, 2, 1, None, 1, 2, 1, 2] * 3,
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
            "t": [1, 2, 1, 2, 1, None, None, 2, 1, 2] * 3
        })
        target = "t"
        features = ["x", "y"]
        biased_groups = ["z"]
        debiased_features = ["y"]

        r = run_classification_task(
            classification_task,
            tmp_path,
            df,
            target,
            features,
            biased_groups=biased_groups,
            debiased_features=debiased_features,
            explain_samples=True
        )

        assert r["status"] == "SUCCESS"
        assert "fields" in r["data"]
        assert "exdata" in r["data"]
        assert "predictData" in r["data"]
        assert "predict_explanations" in r["data"]
        assert "evaluate" in r["data"]
        assert "validation_explanations" in r["data"]
        assert "importantFeatures" in r["data"]
        assert len(r["data"]["validation_explanations"]) > 0
        assert len(r["data"]["predict_explanations"]) > 0
        assert "debiasing_charts" in r["data"]

        debiasing_charts = r["data"]["debiasing_charts"]
        assert len(debiasing_charts) == len(biased_groups)

        for debiasing_chart in debiasing_charts:
            assert "type" in debiasing_chart
            assert debiasing_chart["type"] in ["scatter", "bar"]

            assert "group" in debiasing_chart
            assert debiasing_chart["group"] in biased_groups

            assert "target" in debiasing_chart
            assert debiasing_chart["target"] == target

            charts = debiasing_chart["charts"]
            assert len(charts) == 2

            for chart in charts:
                assert "x_label" in chart
                assert type(chart["x_label"]) is str

                assert "y" in chart
                assert type(chart["y"]) is list

                assert "corr" in chart
                assert "pvalue" in chart

                if debiasing_chart["type"] == "bar":
                    assert "bars" in chart
                    assert type(chart["bars"]) is list

                    for bar in chart["bars"]:
                        assert "x" in bar
                        assert type(bar["x"]) is list

                        assert "name" in bar
                else:
                    assert "x" in chart
                    assert type(chart["x"]) is list

