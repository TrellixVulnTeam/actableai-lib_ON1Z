from tempfile import mkdtemp
import pandas as pd

from actableai.regression.cross_validation import run_cross_validation
from actableai.tasks.regression import _AAIRegressionTrainTask
from actableai.utils.testing import unittest_hyperparameters


def test_cross_validation():
    task = _AAIRegressionTrainTask()
    df_train = pd.DataFrame(
        {
            "x": [x for x in range(0, 100)],
            "y": [y for y in range(100, 200)],
        }
    )
    df_test = pd.DataFrame(
        {
            "x": [x for x in range(0, 100)],
            "y": [y for y in range(100, 200)],
        }
    )
    (
        important_features,
        evaluate,
        predictions,
        prediction_low,
        prediction_high,
        predict_shap_values,
        df_val,
        leaderboard,
    ) = run_cross_validation(
        regression_train_task=task,
        kfolds=5,
        cross_validation_max_concurrency=1,
        explain_samples=False,
        presets="medium_quality_faster_train",
        hyperparameters=unittest_hyperparameters(),
        model_directory=mkdtemp(prefix="autogluon_model"),
        target="y",
        features=["x"],
        run_model=False,
        df_train=df_train,
        df_test=df_test,
        prediction_quantile_low=None,
        prediction_quantile_high=None,
        drop_duplicates=True,
        run_debiasing=False,
        biased_groups=[],
        debiased_features=[],
        residuals_hyperparameters=None,
        num_gpus=0,
        eval_metric="r2",
    )

    assert important_features is not None
    assert evaluate is not None
    assert predictions is not None
    assert predict_shap_values is not None
    assert df_val is not None
    assert leaderboard is not None

    assert prediction_low is None
    assert prediction_high is None
