import pandas as pd
from typing import Dict, List, Optional, Tuple

from actableai.classification.utils import leaderboard_cross_val
from actableai.tasks.regression import _AAIRegressionTrainTask


def run_cross_validation(
    regression_train_task: _AAIRegressionTrainTask,
    kfolds: int,
    cross_validation_max_concurrency: int,
    explain_samples: bool,
    presets: str,
    hyperparameters: Dict[str, Dict],
    model_directory: str,
    target: str,
    features: List[str],
    run_model: bool,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prediction_quantile_low: Optional[float],
    prediction_quantile_high: Optional[float],
    drop_duplicates: bool,
    run_debiasing: bool,
    biased_groups: List[str],
    debiased_features: List[str],
    residuals_hyperparameters: Dict[str, Dict],
    num_gpus: int,
    eval_metric: str,
    time_limit: Optional[int],
    drop_unique: bool,
    drop_useless_features: bool,
) -> Tuple[Dict, Dict, List, float, float, pd.DataFrame, List, List]:
    """Run cross validation on Regression Task. Data is divided in kfold groups and each
    run a regression. The returned values are means or lists of values from
    each sub regression task.

    Args:
        regression_train_task (_AAIRegressionTrainTask): Sub Regression Task Object
        kfolds (int): Number of folds
        cross_validation_max_concurrency (int): Max concurrency of sub regression task
        explain_samples (bool): Explaining the samples
        presets (str): Presets of the regressions
        hyperparameters (Dict[str, Dict]): Hyperparameters of the regressions
        model_directory (str): Model directory for the regression
        target (str): Target for regressions
        features (List[str]): Features for the regressions
        run_model (bool): If True, regressions models run predictions on unseen values
        df_train (pd.DataFrame): Training data
        df_test (pd.DataFrame): Testing data
        prediction_quantile_low (Optional[float]): Prediction quantile low for the regressions
        prediction_quantile_high (Optional[float]): Prediction quantile high for the regressions
        drop_duplicates (bool): If True, only the unique values are kept
        run_debiasing (bool): If True, features are debiased w.r.t to biased_groups and debiased_features
        biased_groups (List[str]): Biased features in the data
        debiased_features (List[str]): Features debiased of the biased groups
        residuals_hyperparameters (Dict[str, Dict]): HyperParameters for debiasing models
        num_gpus (int): Number of GPUs for the model
        time_limit: Time limit of training (in seconds)

    Returns:
        Tuple[Dict, Dict, List, float, float, List, List, List]:
            important_features
            evaluate
            predictions
            prediction_low
            prediction_high
            predict_shap_values
            df_val
            leaderboard
    """
    import os
    from math import sqrt
    import numpy as np
    import pandas as pd
    from multiprocessing.pool import ThreadPool
    from sklearn.model_selection import KFold

    # Run trainers
    kf = KFold(n_splits=kfolds, shuffle=True)
    kfolds_index_list = list(kf.split(df_train))

    kfold_pool = ThreadPool(processes=cross_validation_max_concurrency)

    cross_val_async_results = [
        kfold_pool.apply_async(
            regression_train_task.run,
            kwds={
                "explain_samples": explain_samples,
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
                "prediction_quantile_low": prediction_quantile_low,
                "prediction_quantile_high": prediction_quantile_high,
                "drop_duplicates": drop_duplicates,
                "run_debiasing": run_debiasing,
                "biased_groups": biased_groups,
                "debiased_features": debiased_features,
                "residuals_hyperparameters": residuals_hyperparameters,
                "num_gpus": num_gpus,
                "eval_metric": eval_metric,
                "time_limit": time_limit,
                "drop_unique": drop_unique,
                "drop_useless_features": drop_useless_features,
            },
        )
        for kfold_index, (train_index, val_index) in enumerate(kfolds_index_list)
    ]

    cross_val_results = [results.get() for results in cross_val_async_results]

    kfold_pool.close()

    # Combine results
    cross_val_important_features = {}
    cross_val_important_p_value_features = {}
    cross_val_evaluates = {}
    cross_val_predict_shap_values = []
    cross_val_predictions = []
    cross_val_prediction_low = None
    cross_val_prediction_high = None
    cross_val_leaderboard = []

    df_val = pd.DataFrame()

    for kfold_index, cv_results in enumerate(cross_val_results):
        (
            predictor,
            important_features,
            evaluate,
            y_pred,
            _,
            predictions,
            prediction_low,
            prediction_high,
            df_predict_shap_values,
            leaderboard,
        ) = cv_results

        _, val_index = kfolds_index_list[kfold_index]

        predictor.unpersist_models()
        cross_val_leaderboard.append(leaderboard)

        df_k_val = df_train.iloc[val_index].copy()
        if prediction_quantile_low is not None and prediction_quantile_high is not None:
            y_pred = y_pred[0.5]
        df_k_val[f"{target}_predicted"] = y_pred
        df_val = df_val.append(df_k_val, ignore_index=True)

        for feature in important_features:
            if feature["feature"] not in cross_val_important_features:
                cross_val_important_features[feature["feature"]] = []
            if feature["feature"] not in cross_val_important_p_value_features:
                cross_val_important_p_value_features[feature["feature"]] = []
            cross_val_important_features[feature["feature"]].append(
                feature["importance"],
            )
            cross_val_important_p_value_features[feature["feature"]].append(
                feature["p_value"],
            )

        for metric in evaluate:
            if metric not in cross_val_evaluates:
                cross_val_evaluates[metric] = []
            cross_val_evaluates[metric].append(evaluate[metric])

        if run_model:
            if explain_samples:
                cross_val_predict_shap_values.append(df_predict_shap_values)
            cross_val_predictions.append(predictions)
            if prediction_quantile_low is not None:
                if cross_val_prediction_low is None:
                    cross_val_prediction_low = prediction_low
                else:
                    cross_val_prediction_low = pd.concat(
                        [cross_val_prediction_low, prediction_low], axis=1
                    )
            if prediction_quantile_high is not None:
                if cross_val_prediction_high is None:
                    cross_val_prediction_high = prediction_high
                else:
                    cross_val_prediction_high = pd.concat(
                        [cross_val_prediction_high, prediction_high], axis=1
                    )

    # Evaluate
    sqrt_k = sqrt(kfolds)
    important_features = []
    for k in cross_val_important_features.keys():
        important_features.append(
            {
                "feature": k,
                "importance": np.mean(cross_val_important_features[k]),
                "importance_std_err": np.std(cross_val_important_features[k]) / sqrt_k,
                "p_value": np.mean(cross_val_important_p_value_features[k]),
                "p_value_std_err": np.std(cross_val_important_p_value_features[k])
                / sqrt_k,
            }
        )
    important_features = sorted(
        important_features, key=lambda k: k["importance"], reverse=True
    )

    if prediction_quantile_low is None or prediction_quantile_high is None:
        # Legacy (TODO: to be removed)
        evaluate = {
            "RMSE": np.mean(cross_val_evaluates["RMSE"]),
            "RMSE_std_err": np.std(cross_val_evaluates["RMSE"]) / sqrt_k,
            "R2": np.mean(cross_val_evaluates["R2"]),
            "R2_std_err": np.std(cross_val_evaluates["R2"]) / sqrt_k,
            "MAE": np.mean(cross_val_evaluates["MAE"]),
            "MAE_std_err": np.std(cross_val_evaluates["MAE"]) / sqrt_k,
            "MEDIAN_ABSOLUTE_ERROR": np.mean(
                cross_val_evaluates["MEDIAN_ABSOLUTE_ERROR"]
            ),
            "MEDIAN_ABSOLUTE_ERROR_std_err": np.std(
                cross_val_evaluates["MEDIAN_ABSOLUTE_ERROR"]
            )
            / sqrt_k,
        }

        evaluate["metrics"] = pd.DataFrame(
            {
                "metric": [
                    "Root Mean Squared Error",
                    "R2",
                    "Mean Absolute Error",
                    "Median Absolute Error",
                ],
                "value": [
                    np.mean(cross_val_evaluates["RMSE"]),
                    np.mean(cross_val_evaluates["R2"]),
                    np.mean(cross_val_evaluates["MAE"]),
                    np.mean(cross_val_evaluates["MEDIAN_ABSOLUTE_ERROR"]),
                ],
                "stderr": [
                    np.std(cross_val_evaluates["RMSE"]) / sqrt_k,
                    np.std(cross_val_evaluates["R2"]) / sqrt_k,
                    np.std(cross_val_evaluates["MAE"]) / sqrt_k,
                    np.std(cross_val_evaluates["MEDIAN_ABSOLUTE_ERROR"]) / sqrt_k,
                ],
            }
        )
    else:
        # Legacy (TODO: to be removed)
        evaluate = {
            "PINBALL_LOSS": np.mean(cross_val_evaluates["PINBALL_LOSS"]) / sqrt_k,
        }

        evaluate["metrics"] = pd.DataFrame(
            {
                "metric": [
                    "Pinball Loss",
                ],
                "value": [
                    np.mean(cross_val_evaluates["PINBALL_LOSS"]),
                ],
                "stderr": [
                    np.std(cross_val_evaluates["PINBALL_LOSS"]) / sqrt_k,
                ],
            }
        )

    predictions = []
    df_predict_shap_values = None
    prediction_low = None
    prediction_high = None
    if run_model:
        if explain_samples:
            df_predict_shap_values = (
                pd.concat(cross_val_predict_shap_values, axis=1)
                .groupby(level=0, axis=1)
                .mean()
            )

        predictions = np.mean(cross_val_predictions, axis=0).tolist()
        if prediction_quantile_low is not None:
            prediction_low = np.mean(cross_val_prediction_low, axis=1)
        if prediction_quantile_high is not None:
            prediction_high = np.mean(cross_val_prediction_high, axis=1)

    leaderboard = leaderboard_cross_val(cross_val_leaderboard)

    return (
        important_features,
        evaluate,
        predictions,
        prediction_low,
        prediction_high,
        df_predict_shap_values,
        df_val,
        leaderboard,
    )
