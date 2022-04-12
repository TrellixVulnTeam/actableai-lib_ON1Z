from tempfile import mkdtemp
from numpy import positive
import pandas as pd

from actableai.classification.cross_validation import run_cross_validation
from actableai.tasks.classification import _AAIClassificationTrainTask
from actableai.utils.testing import unittest_hyperparameters


def test_run_cross_validation():
    task = _AAIClassificationTrainTask()
    df_train = pd.DataFrame(
        {
            "x": [x for x in range(0, 100)],
            "y": ["a" for y in range(100, 150)] + ["b" for y in range(150, 200)],
        }
    )
    (
        ensemble_model,
        important_features,
        evaluate,
        df_val_cross_val_pred_prob,
        df_val,
        leaderboard,
    ) = run_cross_validation(
        classification_train_task=task,
        problem_type="multiclass",
        positive_label=None,
        kfolds=5,
        cross_validation_max_concurrency=1,
        presets="medium_quality_faster_train",
        hyperparameters=unittest_hyperparameters(),
        model_directory=mkdtemp(prefix="autogluon_model"),
        target="y",
        features=["x"],
        df_train=df_train,
        drop_duplicates=True,
        run_debiasing=False,
        biased_groups=[],
        debiased_features=[],
        residuals_hyperparameters=None,
        num_gpus=0,
    )

    assert important_features is not None
    assert evaluate is not None
    assert ensemble_model is not None
    assert df_val_cross_val_pred_prob is not None
    assert df_val is not None
    assert leaderboard is not None
