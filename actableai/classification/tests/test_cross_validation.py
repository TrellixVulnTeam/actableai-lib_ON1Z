import pandas as pd
from tempfile import mkdtemp

from actableai.classification.cross_validation import run_cross_validation
from actableai.tasks.classification import _AAIClassificationTrainTask
from actableai.utils.testing import unittest_hyperparameters


def test_run_cross_validation():
    task = _AAIClassificationTrainTask()
    df_train = pd.DataFrame(
        {
            "x": [x for x in range(0, 100)],
            "z": [x for x in range(100, 200)],
            "y": ["a" for y in range(100, 150)] + ["b" for y in range(150, 200)],
        }
    )
    (
        ensemble_model,
        important_features,
        evaluate,
        df_val_cross_val_pred_prob,
        predict_shap_values,
        df_val,
        leaderboard,
    ) = run_cross_validation(
        classification_train_task=task,
        problem_type="binary",
        explain_samples=False,
        positive_label=None,
        kfolds=5,
        cross_validation_max_concurrency=1,
        presets="medium_quality_faster_train",
        hyperparameters=unittest_hyperparameters(),
        model_directory=mkdtemp(prefix="autogluon_model"),
        target="y",
        features=["x", "z"],
        run_model=False,
        df_train=df_train,
        df_test=None,
        drop_duplicates=True,
        run_debiasing=False,
        biased_groups=[],
        debiased_features=[],
        residuals_hyperparameters=None,
        num_gpus=0,
        eval_metric="accuracy",
        time_limit=None,
        drop_unique=False,
        drop_useless_features=False,
    )

    assert important_features is not None
    assert isinstance(important_features, list)
    assert len(important_features) == 2
    assert set([x["feature"] for x in important_features]) == set(["x", "z"])
    for i in range(2):
        assert "feature" in important_features[i]
        assert "importance" in important_features[i]
        assert "importance_std_err" in important_features[i]
        assert "p_value" in important_features[i]
        assert "p_value_std_err" in important_features[i]
    assert evaluate is not None
    assert ensemble_model is not None
    assert df_val_cross_val_pred_prob is not None
    assert df_val is not None
    assert leaderboard is not None
