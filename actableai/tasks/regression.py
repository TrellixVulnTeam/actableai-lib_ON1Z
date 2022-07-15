import pandas as pd
from typing import Dict, List, Optional

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class _AAIRegressionTrainTask(AAITask):
    """
    TODO write documentation
    """

    @AAITask.run_with_ray_remote(TaskType.REGRESSION_TRAIN)
    def run(
        self,
        explain_samples: bool,
        presets: str,
        hyperparameters: Dict,
        model_directory: str,
        target: str,
        features: List[str],
        run_model: bool,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        prediction_quantile_low: int,
        prediction_quantile_high: int,
        drop_duplicates: bool,
        run_debiasing: bool,
        biased_groups: List[str],
        debiased_features: List[str],
        residuals_hyperparameters: Dict,
        num_gpus: int,
        eval_metric: Dict,
        time_limit: Optional[int],
        drop_unique: bool,
        drop_useless_features: bool,
    ):
        """Sub class for running a regression without cross validation

        Args:
            explain_samples: Whether we explain the samples
            presets: Presets for AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            hyperparameters: Hyperparameters for AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            model_directory: Directory for model
            target: Target for regression
            features: Features used in DataFrame for regression
            run_model: Whether the model should be run on df_test
            df_train: DataFrame for training
            df_val: DataFrame for validation
            df_test: DataFrame for testing
            prediction_quantile_low: Low quantile for prediction and validation
            prediction_quantile_high: High quantile for prediction and validation
            drop_duplicates: Whether we should drop duplicated rows
            run_debiasing: Whether we should debias the data
            biased_groups: Features that creates a bias in prediction
            debiased_features: Features debiased w.r.t to biased_groups
            residuals_hyperparameters: Hyperparameters for debiasing with AutoGluon.
                See https://auto.gluon.ai/stable/api/autogluon.task.html?highlight=tabularpredictor#autogluon.tabular.TabularPredictor
            num_gpus: Number of GPUs used by AutoGluon
            eval_metric: Evaluation metric for validation
            time_limit: Time limit of training

        Returns:
            Tuple:
                - AutoGluon's predictor
                - List of feature importance
                - Dictionary of evaluation metrics
                - Predicted values for df_val
                - Explainer for explaining the prediction and validation
                - Predicted values for df_test if run_model is true
                - Lowest prediction for df_test if prediction_quantile_low is not None
                - Highest prediction for df_test if prediction_quantile_high is not None
                - Predicted shap values if explain_samples is true
                - Leaderboard of the best model ran by AutoGluon
        """
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from actableai.utils import (
            debiasing_feature_generator_args,
        )
        from actableai.debiasing.debiasing_model import DebiasingModel
        from actableai.explanation.autogluon_explainer import AutoGluonShapTreeExplainer

        ag_args_fit = {"drop_unique": drop_unique}
        feature_generator_args = {}

        if not drop_useless_features:
            feature_generator_args["pre_drop_useless"] = False
            feature_generator_args["post_generators"] = []

        if run_debiasing:
            ag_args_fit["drop_duplicates"] = drop_duplicates
            ag_args_fit["label"] = target
            ag_args_fit["features"] = features
            ag_args_fit["biased_groups"] = biased_groups
            ag_args_fit["debiased_features"] = debiased_features
            ag_args_fit["hyperparameters_residuals"] = residuals_hyperparameters
            ag_args_fit["presets_residuals"] = presets
            ag_args_fit["hyperparameters_non_residuals"] = hyperparameters
            ag_args_fit["presets_non_residuals"] = presets
            ag_args_fit["drop_useless_features"] = drop_useless_features

            feature_generator_args = {
                **feature_generator_args,
                **debiasing_feature_generator_args(),
            }

            hyperparameters = {DebiasingModel: {}}

        ag_args_fit["num_cpus"] = 1
        ag_args_fit["num_gpus"] = num_gpus

        df_train = df_train[features + biased_groups + [target]]
        df_val = df_val[features + biased_groups + [target]]
        df_test = df_test[features + biased_groups]

        quantile_levels = None
        if prediction_quantile_low is not None and prediction_quantile_high is not None:
            quantile_levels = [
                prediction_quantile_low / 100,
                0.5,
                prediction_quantile_high / 100,
            ]

        # Train
        predictor = TabularPredictor(
            label=target,
            path=model_directory,
            problem_type="regression" if quantile_levels is None else "quantile",
            eval_metric=eval_metric,
            quantile_levels=quantile_levels,
        )

        predictor = predictor.fit(
            train_data=df_train,
            presets=presets,
            hyperparameters=hyperparameters,
            ag_args_fit=ag_args_fit,
            feature_generator=AutoMLPipelineFeatureGenerator(**feature_generator_args),
            time_limit=time_limit,
        )

        explainer = None
        if explain_samples:
            # Filter out models which are not compatible with explanations
            while not AutoGluonShapTreeExplainer.is_predictor_compatible(predictor):
                predictor.delete_models(
                    models_to_delete=predictor.get_model_best(),
                    dry_run=False,
                    allow_delete_cascade=True,
                )

            explainer = AutoGluonShapTreeExplainer(predictor)

        predictor.persist_models()
        leaderboard = predictor.leaderboard(extra_info=True)
        pd.set_option("chained_assignment", "warn")

        important_features = []
        if quantile_levels is None:
            feature_importance = predictor.feature_importance(df_train)
            for i in range(len(feature_importance)):
                if feature_importance.index[i] in biased_groups:
                    continue
                important_features.append(
                    {
                        "feature": feature_importance.index[i],
                        "importance": feature_importance["importance"][i],
                        "p_value": feature_importance["p_value"][i],
                    }
                )
        y_pred = predictor.predict(df_val)
        metrics = predictor.evaluate_predictions(
            y_true=df_val[target], y_pred=y_pred, auxiliary_metrics=True
        )

        if quantile_levels is None:
            # Legacy (TODO: to be removed)
            evaluate = {
                "RMSE": abs(metrics["root_mean_squared_error"]),
                "R2": metrics["r2"],
                "MAE": abs(metrics["mean_absolute_error"]),
                "MSE": abs(metrics["mean_squared_error"]),
                "MEDIAN_ABSOLUTE_ERROR": abs(metrics["median_absolute_error"]),
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
                        abs(metrics["root_mean_squared_error"]),
                        metrics["r2"],
                        abs(metrics["mean_absolute_error"]),
                        abs(metrics["median_absolute_error"]),
                    ],
                }
            )
        else:
            # Legacy (TODO: to be removed)
            evaluate = {
                "PINBALL_LOSS": metrics["pinball_loss"],
            }

            evaluate["metrics"] = pd.DataFrame(
                {
                    "metric": [
                        "Pinball Loss",
                    ],
                    "value": [
                        metrics["pinball_loss"],
                    ],
                }
            )

        predictions = None
        predictions_low = None
        predictions_high = None
        df_predict_shap_values = None
        if run_model:
            if explain_samples:
                df_predict_shap_values = explainer.shap_values(df_test)

            full_predictions = predictor.predict(df_test)

            if quantile_levels is None:
                predictions = full_predictions
            else:
                predictions = full_predictions[0.5]
                predictions_low = full_predictions[prediction_quantile_low / 100]
                predictions_high = full_predictions[prediction_quantile_high / 100]

        return (
            predictor,
            important_features,
            evaluate,
            y_pred,
            explainer,
            predictions,
            predictions_low,
            predictions_high,
            df_predict_shap_values,
            leaderboard,
        )


class AAIRegressionTask(AAITask):
    """Regression task."""

    @AAITask.run_with_ray_remote(TaskType.REGRESSION)
    def run(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        biased_groups: Optional[List[str]] = None,
        debiased_features: Optional[List[str]] = None,
        eval_metric: str = "r2",
        validation_ratio: float = 0.2,
        prediction_quantile_low: int = 5,
        prediction_quantile_high: int = 95,
        explain_samples: bool = False,
        model_directory: Optional[str] = None,
        presets: str = "medium_quality_faster_train",
        hyperparameters: Optional[dict] = None,
        train_task_params: Optional[dict] = None,
        kfolds: int = 1,
        cross_validation_max_concurrency: int = 1,
        residuals_hyperparameters: Optional[dict] = None,
        drop_duplicates: bool = True,
        return_residuals: bool = False,
        kde_steps: int = 10,
        num_gpus: int = 0,
        time_limit: Optional[int] = None,
        drop_unique: bool = True,
        drop_useless_features: bool = True,
    ):
        """Run this regression task and return results.

        Args:
            df: Input data frame
            target: Target columns in df. If there are emtpy values in this columns,
                predictions will be generated for these rows.
            features: A list of features to be used for prediction. If None, all columns
                except target are used as features. Defaults to None.
            biased_groups: A list of columns of groups that should be protected from
                biases (e.g. gender, race, age). Defaults to None.
            debiased_features: A list of proxy features that need to be debiased for
                protection of sensitive groups. Defaults to None.
            eval_metric: Metric to be optimized during training. Possible values include
                'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error',
                'median_absolute_error', 'r2'. Defaults to "r2".
            validation_ratio: The ratio to randomly split data for training and
                validation. Defaults to .2.
            prediction_quantile_low: Lower quantile for quantile regression (in
                percentage). Defaults to 5.
            prediction_quantile_high: Higher quantile for quantile regression
                (in percentage). Defaults to 95.
            explain_samples: If true, explanations for predictions in test and
                validation will be generated. It takes significantly longer time to run.
                Defaults to False.
            model_directory: Destination to store trained model. If not set, a temporary
                folder will be created. Defaults to None.
            presets: Autogluon's presets for training model.
                See
                https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit.
                Defaults to "medium_quality_faster_train".
            hyperparameters: Autogluon's hyperparameters for training model.
                See
                https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit.
                Defaults to "medium_quality_faster_train".
            train_task_params: Parameters for _AAITrainTask constructor.
                Defaults to None.
            kfolds: Number of folds for cross validation. If 1, train test split is used
                instead.. Defaults to 1.
            cross_validation_max_concurrency: Maximum number of Ray actors used for
                cross validation (each actor execute for one split). Defaults to 1.
            residuals_hyperparameters: Autogluon's hyperparameteres used in final model
                of counterfactual predictions. Defaults to None.
            drop_duplicates: Whether duplicate values should be dropped before training.
                Defaults to True.
            return_residuals: Whether residual values should be returned in
                counterfactual prediction. Defaults to False.
            kde_steps: Steps used to generate KDE plots with debiasing. Defaults to 10.
            num_gpus: Number of GPUs used in nuisnace models in counterfactual
                prediction. Defaults to 0.
            time_limit: time limit (in seconds) of training. Defaults to None, which means
                there is no time limit.

        Examples:
            >>> import pandas as pd
            >>> from actableai.tasks.regression import AAIRegressionTask
            >>> df = pd.read_csv("path/to/csv")
            >>> result = AAIRegressionTask().run(
            ...     df,
            ...     'target_column',
            ... )

        Returns:
            Dict: Dictionnary containing results.
        """
        import time
        from tempfile import mkdtemp
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KernelDensity
        from actableai.utils import (
            memory_efficient_hyperparameters,
            explanation_hyperparameters,
        )
        from actableai.data_validation.params import RegressionDataValidator
        from actableai.data_validation.base import (
            CheckLevels,
            UNIQUE_CATEGORY_THRESHOLD,
        )
        from actableai.regression.cross_validation import run_cross_validation
        from actableai.utils.sanitize import sanitize_timezone

        start = time.time()
        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        # Handle default parameters
        if features is None:
            features = list(df.columns.drop(target, errors="ignore"))
        if biased_groups is None:
            biased_groups = []
        if debiased_features is None:
            debiased_features = []
        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")
        if train_task_params is None:
            train_task_params = {}

        run_debiasing = len(biased_groups) > 0 and len(debiased_features) > 0

        # Pre process data
        df = df.fillna(np.nan)

        # Validate parameters
        data_validation_results = RegressionDataValidator().validate(
            target,
            features,
            df,
            biased_groups,
            debiased_features,
            eval_metric,
            prediction_quantile_low,
            prediction_quantile_high,
            presets,
            explain_samples,
            drop_duplicates,
        )
        failed_checks = [
            check for check in data_validation_results if check is not None
        ]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": check.name, "level": check.level, "message": check.message}
                    for check in failed_checks
                ],
                "runtime": time.time() - start,
            }

        if prediction_quantile_low is not None and prediction_quantile_high is not None:
            eval_metric = "pinball_loss"

        if hyperparameters is None:
            if explain_samples:
                hyperparameters = explanation_hyperparameters()
            else:
                hyperparameters = memory_efficient_hyperparameters()

        # Split data
        df_train = df[pd.notnull(df[target])]
        if drop_duplicates:
            df_train = df_train.drop_duplicates(subset=features + [target])
        df_val = None
        if kfolds <= 1:
            df_train, df_val = train_test_split(df_train, test_size=validation_ratio)

        df_test = df[pd.isnull(df[target])].drop(columns=[target])

        # If true it means that the regression needs to be run on test data
        run_model = df_test.shape[0] > 0

        df_predict = df_test.copy()

        # Train
        regression_train_task = _AAIRegressionTrainTask(**train_task_params)

        y_pred = None
        predictor = None
        explainer = None
        leaderboard = None
        if kfolds > 1:
            (
                important_features,
                evaluate,
                predictions,
                prediction_low,
                prediction_high,
                df_predict_shap_values,
                df_val,
                leaderboard,
            ) = run_cross_validation(
                regression_train_task=regression_train_task,
                kfolds=kfolds,
                cross_validation_max_concurrency=cross_validation_max_concurrency,
                explain_samples=explain_samples,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_test=df_test,
                prediction_quantile_low=prediction_quantile_low,
                prediction_quantile_high=prediction_quantile_high,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
                eval_metric=eval_metric,
                time_limit=time_limit,
                drop_unique=drop_unique,
                drop_useless_features=drop_useless_features,
            )
        else:
            (
                predictor,
                important_features,
                evaluate,
                y_pred,
                explainer,
                predictions,
                prediction_low,
                prediction_high,
                df_predict_shap_values,
                leaderboard,
            ) = regression_train_task.run(
                explain_samples=explain_samples,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                prediction_quantile_low=prediction_quantile_low,
                prediction_quantile_high=prediction_quantile_high,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
                eval_metric=eval_metric,
                time_limit=time_limit,
                drop_unique=drop_unique,
                drop_useless_features=drop_useless_features,
            )

        # Validation
        df_eval_shap_values = None
        if kfolds <= 1:
            if prediction_quantile_low is None or prediction_quantile_high is None:
                df_val[target + "_predicted"] = y_pred
            else:
                df_val[target + "_predicted"] = y_pred[0.5]
                df_val[target + "_low"] = y_pred[prediction_quantile_low / 100]
                df_val[target + "_high"] = y_pred[prediction_quantile_high / 100]

            # FIXME this should not be done, but we need this for now so we can return the models
            predictor.persist_models()
            # predictor.unpersist_models()

            if explain_samples:
                df_eval_shap_values = explainer.shap_values(
                    df_val[features + biased_groups]
                )

        # Prediction
        if run_model:
            df_predict[target + "_predicted"] = predictions
            if prediction_quantile_low is not None:
                df_predict[target + "_low"] = prediction_low
            if prediction_quantile_high is not None:
                df_predict[target + "_high"] = prediction_high

        debiasing_charts = []
        # Generate debiasing charts
        if run_debiasing:

            def kde_bandwidth(x):
                return max(0.5 * x.std() * (x.size ** (-0.2)), 1e-2)

            kde_x_axis_gt = np.linspace(
                df_val[target].min(), df_val[target].max(), kde_steps
            )
            kde_x_axis_predicted = np.linspace(
                df_val[f"{target}_predicted"].min(),
                df_val[f"{target}_predicted"].max(),
                kde_steps,
            )

            plot_targets = [target, f"{target}_predicted"]

            for biased_group in biased_groups:
                group_charts = []
                group_chart_type = None

                df_val_biased_group = df_val[biased_group].fillna("NaN").astype(str)
                biased_classes = sorted(df_val_biased_group.unique())

                if len(biased_classes) <= UNIQUE_CATEGORY_THRESHOLD:
                    # Categorical Biased Group => KDE plot
                    group_chart_type = "kde"

                    for plot_target, x_axis in zip(
                        plot_targets, [kde_x_axis_gt, kde_x_axis_predicted]
                    ):
                        y_prob = {}

                        for biased_class in biased_classes:
                            values = df_val[df_val_biased_group == biased_class][
                                plot_target
                            ]
                            values = values[values.notna()]

                            kde = KernelDensity(bandwidth=kde_bandwidth(values))
                            kde = kde.fit(values.values.reshape(-1, 1))

                            y_prob[biased_class] = kde.score_samples(
                                x_axis.reshape(-1, 1)
                            )

                        corr, pvalue = spearmanr(
                            df_val[biased_group], df_val[plot_target]
                        )
                        group_charts.append(
                            {
                                "x_label": plot_target,
                                "x": x_axis.tolist(),
                                "lines": [
                                    {
                                        "y": np.exp(y).tolist(),
                                        "name": biased_class,
                                    }
                                    for biased_class, y in y_prob.items()
                                ],
                                "corr": corr,
                                "pvalue": pvalue,
                            }
                        )
                else:
                    # Non-Categorical Biased Group => Scatter plot
                    group_chart_type = "scatter"

                    for plot_target in plot_targets:
                        X = df_val[plot_target]
                        y = df_val[biased_group]

                        notna_mask = X.notna() & y.notna()
                        X = X[notna_mask]
                        y = y[notna_mask]

                        corr, pvalue = spearmanr(
                            df_val[biased_group], df_val[plot_target]
                        )
                        group_charts.append(
                            {
                                "x_label": plot_target,
                                "x": X.tolist(),
                                "y": y.tolist(),
                                "corr": corr,
                                "pvalue": pvalue,
                            }
                        )

                debiasing_charts.append(
                    {
                        "type": group_chart_type,
                        "group": biased_group,
                        "target": target,
                        "charts": group_charts,
                    }
                )

        leaderboard_obj_cols = leaderboard.select_dtypes(include=["object"]).columns
        leaderboard[leaderboard_obj_cols] = leaderboard[leaderboard_obj_cols].astype(
            str
        )

        data = {
            "validation_table": df_val if kfolds <= 1 else None,
            "prediction_table": df_predict,
            "predict_shaps": df_predict_shap_values.to_numpy()
            if df_predict_shap_values is not None
            else [],
            "predict_shaps_v2": df_predict_shap_values,
            "evaluate": evaluate,
            "validation_shaps": df_eval_shap_values.to_numpy()
            if df_eval_shap_values is not None
            else [],
            "validation_shaps_v2": df_eval_shap_values,
            "importantFeatures": important_features,
            "debiasing_charts": debiasing_charts,
            "leaderboard": leaderboard,
        }

        runtime = time.time() - start
        return {
            "status": "SUCCESS",
            "messenger": "",
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "runtime": runtime,
            "data": data,
            "model": predictor if kfolds <= 1 else None,
            # FIXME this predictor is not really usable as is for now
        }
