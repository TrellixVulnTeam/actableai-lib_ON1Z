import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from actableai.classification.roc_curve_cross_validation import (
    cross_validation_curve,
)
from actableai.classification.utils import leaderboard_cross_val
from actableai.tasks.classification import _AAIClassificationTrainTask
from sklearn.metrics import auc


class AverageEnsembleClassifier:
    def __init__(self, predictors):
        """Constructor for AverageEnsembleClassifier.

        Args:
            predictors: List of predictors.
        """
        self.predictors = predictors
        self.class_labels = predictors[0].class_labels

    def _predict_proba(self, X: pd.DataFrame, *args, **kwargs) -> List[np.ndarray]:
        """Predict probabilities for each predictor for each class for each sample.

        Args:
            X: DataFrame with features.

        Returns:
            List[np.ndarray]: List of probabilities for each predictor for each class
                for each sample.
        """
        predictors_results = []

        for predictor in self.predictors:
            predictors_results.append(predictor.predict_proba(X, *args, **kwargs))

        return predictors_results

    def predict(self, X) -> pd.Series:
        """Predicts the class for each sample in X.
        Args:
            X: DataFrame with features.

        Returns:
            pd.Series: Predicted class for each sample.
        """
        predictors_results = self._predict_proba(X)
        pred_probas = np.mean(predictors_results, axis=0).tolist()

        if len(self.class_labels) == 2:
            pred_labels = [
                self.class_labels[0] if x < 0.5 else self.class_labels[1]
                for x in pred_probas
            ]
        else:
            pred_labels = [self.class_labels[np.argmax(x)] for x in pred_probas]

        return pd.Series(pred_labels)

    def predict_proba(self, X, *args, **kwargs):
        """Predict probabilities for each predictor for each class for each sample.

        Args:
            X: DataFrame with features.

        Returns:
            List[np.ndarray]: List of probabilities for each predictor for each class
        """
        predictors_results = self._predict_proba(X, *args, **kwargs)
        return sum(predictors_results) / len(predictors_results)

    def unpersist_models(self):
        """Unpersists all models in the ensemble."""
        for predictor in self.predictors:
            predictor.unpersist_models()


def run_cross_validation(
    classification_train_task: _AAIClassificationTrainTask,
    problem_type: str,
    explain_samples: bool,
    positive_label: Optional[str],
    presets: str,
    hyperparameters: dict,
    model_directory: str,
    target: str,
    features: list,
    run_model: bool,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    kfolds: int,
    cross_validation_max_concurrency: int,
    drop_duplicates: bool,
    run_debiasing: bool,
    biased_groups: list,
    debiased_features: list,
    residuals_hyperparameters: Optional[dict],
    num_gpus: int,
    eval_metric: str,
    time_limit: Optional[int],
) -> Tuple[AverageEnsembleClassifier, list, dict, list, pd.DataFrame, pd.DataFrame]:
    """Runs a cross validation for a classification task.

    Args:
        classification_train_task: The classification task to run cross validation on.
        problem_type (str): The problem type. Can be either 'binary' or 'multiclass'.
        explain_samples (bool): Explaining the samples.
        positive_label (str): The positive label. Only used if problem_type is 'binary'.
        presets (dict): The presets to use for AutoGluon.
            See
            https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            for more information.
        hyperparameters: The hyperparameters to use for AutoGluon.
            See
            https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            for more information.
        model_directory: The directory to store the models.
        target: The target column.
        features: The features columns used for training/prediction.
        run_model: If True, regressions models run predictions on unseen values.
        df_train: The input dataframe.
        df_test: Testing data.
        kfolds: The number of folds to use for cross validation.
        cross_validation_max_concurrency: The maximum number of concurrent
            processes to use for cross validation.
        drop_duplicates: Whether to drop duplicates.
        run_debiasing: Whether to run debiasing.
        biased_groups: The groups introducing bias.
        debiased_features: The features to debias.
        residuals_hyperparameters: The hyperparameters to use for the debiasing model.
        num_gpus (int): The number of GPUs to use.
        eval_metric: Metric to be optimized for.

    Returns:
        Tuple: Result of the cross validation.
            - AverageEnsembleClassifier: The average ensemble classifier.
            - list: The feature importances.
            - dict: The evaluation metrics.
            - list: Probabilities of the predicted classes.
            - pd.DataFrame: The training dataframe.
            - pd.DataFrame: The test dataframe.
    """
    import os
    import math
    import json
    import numpy as np
    from multiprocessing.pool import ThreadPool
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=kfolds, shuffle=True)
    kfolds_index_list = list(kf.split(df_train, df_train[target]))

    kfold_pool = ThreadPool(processes=cross_validation_max_concurrency)

    # Run trainers
    cross_val_async_results = [
        kfold_pool.apply_async(
            classification_train_task.run,
            kwds={
                "problem_type": problem_type,
                "explain_samples": explain_samples,
                "positive_label": positive_label,
                "presets": presets,
                "hyperparameters": hyperparameters,
                "model_directory": os.path.join(
                    model_directory, f"trainer_{kfold_index}"
                ),
                "target": target,
                "features": features,
                "run_model": run_model,
                "df_train": df_train.iloc[train_index],
                "df_val": df_train.iloc[val_index],
                "df_test": df_test,
                "drop_duplicates": drop_duplicates,
                "run_debiasing": run_debiasing,
                "biased_groups": biased_groups,
                "debiased_features": debiased_features,
                "residuals_hyperparameters": residuals_hyperparameters,
                "num_gpus": num_gpus,
                "eval_metric": eval_metric,
                "time_limit": time_limit,
            },
        )
        for kfold_index, (train_index, val_index) in enumerate(kfolds_index_list)
    ]

    cross_val_results = [results.get() for results in cross_val_async_results]

    kfold_pool.close()

    # Combine results
    cross_val_predictors = []
    cross_val_leaderboard = []
    cross_val_important_features = {}
    cross_val_evaluates = {}
    cross_val_auc_curves = {}
    cross_val_precision_recall_curves = {}
    cross_val_predict_shap_values = []
    df_val_cross_val_pred_prob = []

    df_val = pd.DataFrame()

    for kfold_index, (
        predictor,
        explainer,
        important_features,
        evaluate,
        df_val_pred_prob,
        predict_shap_values,
        leaderboard,
    ) in enumerate(cross_val_results):
        _, val_index = kfolds_index_list[kfold_index]

        df_k_val = df_train.iloc[val_index].copy()
        df_k_val[f"{target}_predicted"] = df_val_pred_prob.idxmax(axis=1)
        df_val = df_val.append(df_k_val, ignore_index=True)

        cross_val_predictors.append(predictor)
        cross_val_leaderboard.append(predictor.leaderboard())

        for feature in important_features:
            if feature["feature"] not in cross_val_important_features:
                cross_val_important_features[feature["feature"]] = []

            cross_val_important_features[feature["feature"]].append(
                feature["importance"]
            )

        for metric in evaluate:
            if metric not in cross_val_evaluates:
                cross_val_evaluates[metric] = []

            cross_val_evaluates[metric].append(evaluate[metric])

            auc_curve = evaluate["auc_curve"]
            for metric in auc_curve:
                if metric not in cross_val_auc_curves:
                    cross_val_auc_curves[metric] = []
                cross_val_auc_curves[metric].append(auc_curve[metric])

            precision_recall_curve = evaluate["precision_recall_curve"]
            for metric in precision_recall_curve:
                if metric not in cross_val_precision_recall_curves:
                    cross_val_precision_recall_curves[metric] = []
                cross_val_precision_recall_curves[metric].append(
                    precision_recall_curve[metric]
                )

        df_val_cross_val_pred_prob.append(
            json.loads(df_val_pred_prob.to_json(orient="table"))["data"]
        )

        if run_model and explain_samples:
            cross_val_predict_shap_values.append(predict_shap_values)

    # Evaluate results
    sqrt_k = math.sqrt(kfolds)
    important_features = []
    for k in cross_val_important_features.keys():
        important_features.append(
            {
                "feature": k,
                "importance": np.mean(cross_val_important_features[k]),
                "importance_std_err": np.std(cross_val_important_features[k]) / sqrt_k,
            }
        )
    important_features = sorted(
        important_features, key=lambda k: k["importance"], reverse=True
    )

    metric_groups = pd.concat(cross_val_evaluates["metrics"]).groupby("metric")
    metric_df = pd.DataFrame(
        {
            "metrics": metric_groups.mean()["value"].index,
            "value": metric_groups.mean()["value"].values.flatten(),
            "stderr": metric_groups.agg(np.std)["value"].values / sqrt_k,
        }
    )
    evaluate = {
        "problem_type": cross_val_evaluates["problem_type"][0],
        "labels": [str(x) for x in cross_val_evaluates["labels"][0]],
        "accuracy": np.mean(cross_val_evaluates["accuracy"], axis=0),
        "accuracy_std_err": np.std(cross_val_evaluates["accuracy"], axis=0) / sqrt_k,
        "confusion_matrix": np.mean(
            cross_val_evaluates["confusion_matrix"], axis=0
        ).tolist(),
        "confusion_matrix_std_err": (
            np.std(cross_val_evaluates["confusion_matrix"], axis=0) / sqrt_k
        ).tolist(),
        "metrics": metric_df,
    }

    evaluate["auc_curve"] = cross_validation_curve(
        cross_val_auc_curves, x="False Positive Rate", y="True Positive Rate"
    )
    evaluate["auc_score"] = auc(
        evaluate["auc_curve"]["False Positive Rate"],
        evaluate["auc_curve"]["True Positive Rate"],
    )
    evaluate["precision_recall_curve"] = cross_validation_curve(
        cross_val_precision_recall_curves, x="Recall", y="Precision"
    )
    evaluate["precision_score"] = np.mean(
        cross_val_evaluates["precision_score"], axis=0
    )
    evaluate["recall_score"] = np.mean(cross_val_evaluates["recall_score"], axis=0)
    evaluate["f1_score"] = np.mean(cross_val_evaluates["f1_score"], axis=0)
    evaluate["positive_count"] = np.mean(cross_val_evaluates["positive_count"])
    evaluate["negative_count"] = np.mean(cross_val_evaluates["negative_count"])

    # Create ensemble model
    ensemble_model = AverageEnsembleClassifier(cross_val_predictors)

    predict_shap_values = []
    if run_model and explain_samples:
        predict_shap_values = np.mean(cross_val_predict_shap_values, axis=0)

    leaderboard = leaderboard_cross_val(cross_val_leaderboard)

    return (
        ensemble_model,
        important_features,
        evaluate,
        df_val_cross_val_pred_prob,
        predict_shap_values,
        df_val,
        leaderboard,
    )
