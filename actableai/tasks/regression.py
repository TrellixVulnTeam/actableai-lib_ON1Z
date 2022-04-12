from typing import List, Optional
import pandas as pd

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class _AAIRegressionTrainTask(AAITask):
    """
    TODO write documentation
    """

    @AAITask.run_with_ray_remote(TaskType.REGRESSION_TRAIN)
    def run(
        self,
        explain_samples,
        presets,
        hyperparameters,
        model_directory,
        target,
        features,
        run_model,
        df_train,
        df_val,
        df_test,
        prediction_quantile_low,
        prediction_quantile_high,
        drop_duplicates,
        run_debiasing,
        biased_groups,
        debiased_features,
        residuals_hyperparameters,
        num_gpus,
        eval_metric,
    ):
        """
        TODO write documentation
        """
        import os
        import shap
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from actableai.utils import (
            preprocess_data_for_shap,
            AutogluonShapWrapper,
            debiasing_feature_generator_args,
        )
        from actableai.debiasing.debiasing_model import DebiasingModel

        ag_args_fit = {}
        feature_generator_args = {}
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

            feature_generator_args = debiasing_feature_generator_args()

            hyperparameters = {DebiasingModel: {}}

        ag_args_fit["num_gpus"] = num_gpus

        df_train = df_train[features + biased_groups + [target]]
        df_val = df_val[features + biased_groups + [target]]
        df_test = df_test[features + biased_groups]

        # Train
        predictor = TabularPredictor(
            label=target,
            path=model_directory,
            problem_type="regression",
            eval_metric=eval_metric,
        )

        predictor = predictor.fit(
            train_data=df_train,
            presets=presets,
            hyperparameters=hyperparameters,
            ag_args_fit=ag_args_fit,
            feature_generator=AutoMLPipelineFeatureGenerator(**feature_generator_args),
        )
        predictor.persist_models()
        leaderboard = predictor.leaderboard()
        pd.set_option("chained_assignment", "warn")

        important_features = []
        for feature, importance in predictor.feature_importance(df_train)[
            "importance"
        ].iteritems():
            if feature in biased_groups:
                continue

            important_features.append({"feature": feature, "importance": importance})

        y_pred = predictor.predict(df_val)
        metrics = predictor.evaluate_predictions(
            y_true=df_val[target], y_pred=y_pred, auxiliary_metrics=True
        )
        evaluate = {
            "RMSE": abs(metrics["root_mean_squared_error"]),
            "R2": metrics["r2"],
            "MAE": abs(metrics["mean_absolute_error"]),
        }

        explainer = None
        if explain_samples:
            df_full = df_train.append(df_val).append(df_test).drop(columns=[target])

            shap_data = preprocess_data_for_shap(df_full)
            ag_wrapper = AutogluonShapWrapper(predictor, shap_data.columns)
            explainer = shap.KernelExplainer(
                ag_wrapper.predict, shap_data, feature_names=shap_data.columns
            )

        predictions = []
        prediction_low = []
        prediction_high = []
        predict_shap_values = []
        if run_model:
            if explain_samples:
                predict_shap_values = explainer.shap_values(
                    preprocess_data_for_shap(df_test)
                ).tolist()

            predictions = predictor.predict(df_test).tolist()
            if prediction_quantile_low is not None:
                prediction_low = predictor.predict(
                    df_test, quantile=prediction_quantile_low
                )
            if prediction_quantile_high is not None:
                prediction_high = predictor.predict(
                    df_test, quantile=prediction_quantile_high
                )

        return (
            predictor,
            important_features,
            evaluate,
            y_pred,
            explainer,
            predictions,
            prediction_low,
            prediction_high,
            predict_shap_values,
            leaderboard,
        )


class _AAIInterventionTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.CAUSAL_INFERENCE)
    def run(
        self,
        df,
        df_predict,
        target,
        run_model,
        current_intervention_column,
        new_intervention_column,
        common_causes,
        causal_cv,
        causal_hyperparameters,
        cate_alpha,
        presets,
        model_directory,
        num_gpus,
    ):
        import numpy as np
        import pandas as pd
        from tempfile import mkdtemp
        from econml.dml import LinearDML, NonParamDML
        from sklearn.impute import SimpleImputer
        from actableai.causal.predictors import SKLearnWrapper
        from actableai.causal import OneHotEncodingTransformer
        from autogluon.tabular import TabularPredictor

        df_ = df.copy()
        if run_model:
            df_.loc[df_predict.index, target] = df_predict[target + "_predicted"]
        df_ = df_[pd.notnull(df_[[current_intervention_column, target]]).all(axis=1)]

        columns = [current_intervention_column, new_intervention_column] + common_causes
        df_imputed = df_.copy()
        num_cols = df_imputed[columns]._get_numeric_data().columns
        cat_cols = list(set(df_imputed.columns) - set(num_cols))
        if len(num_cols) > 0:
            df_imputed[num_cols] = SimpleImputer(strategy="median").fit_transform(
                df_imputed[num_cols]
            )
        if len(cat_cols) > 0:
            df_imputed[cat_cols] = SimpleImputer(
                strategy="most_frequent"
            ).fit_transform(df_imputed[cat_cols])

        X = df_imputed[common_causes] if len(common_causes) > 0 else None
        model_t = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="t",
            problem_type="regression"
            if current_intervention_column in num_cols
            else "multiclass",
        )
        model_t = SKLearnWrapper(
            model_t,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )

        model_y = TabularPredictor(
            path=mkdtemp(prefix=str(model_directory)),
            label="y",
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y,
            hyperparameters=causal_hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )

        if (
            (X is None)
            or (cate_alpha is not None)
            or (
                current_intervention_column not in num_cols
                and len(df[current_intervention_column].unique()) > 2
            )
        ):
            # Multiclass treatment
            causal_model = LinearDML(
                model_t=model_t,
                model_y=model_y,
                featurizer=None if X is None else OneHotEncodingTransformer(X),
                cv=causal_cv,
                linear_first_stages=False,
                discrete_treatment=current_intervention_column not in num_cols,
            )
        else:
            model_final = TabularPredictor(
                path=mkdtemp(prefix=str(model_directory)),
                label="y_res",
                problem_type="regression",
            )
            model_final = SKLearnWrapper(
                df_[[target]],
                model_final,
                hyperparameters=causal_hyperparameters,
                presets=presets,
                ag_args_fit={
                    "num_gpus": num_gpus,
                },
            )
            causal_model = NonParamDML(
                model_t=model_t,
                model_y=model_y,
                model_final=model_final,
                featurizer=None if X is None else OneHotEncodingTransformer(X),
                cv=causal_cv,
                discrete_treatment=current_intervention_column not in num_cols,
            )

        causal_model.fit(
            df_[[target]].values,
            df_[[current_intervention_column]].values,
            X=X,
        )

        df_intervene = df_[
            pd.notnull(df_[[current_intervention_column, new_intervention_column]]).all(
                axis=1
            )
        ]
        df_imputed = df_imputed.loc[df_intervene.index]
        X = df_imputed[common_causes] if len(common_causes) > 0 else None
        effects = causal_model.effect(
            X,
            T0=df_imputed[[current_intervention_column]],
            T1=df_imputed[[new_intervention_column]],
        )

        targets = df_intervene[target].copy()
        if run_model:
            df_predict_index = df_predict.index.intersection(df_intervene.index)
            df_intervene.loc[df_predict_index, target] = np.NaN
            df_intervene.loc[df_predict_index, target + "_predicted"] = df_predict.loc[
                df_predict_index, target + "_predicted"
            ]
        df_intervene[target + "_intervened"] = targets + effects.flatten()
        if cate_alpha is not None:
            lb, ub = causal_model.effect_interval(
                X,
                T0=df_intervene[[current_intervention_column]],
                T1=df_intervene[[new_intervention_column]],
                alpha=cate_alpha,
            )
            df_intervene[target + "_intervened_low"] = targets + lb.flatten()
            df_intervene[target + "_intervened_high"] = targets + ub.flatten()

        df_intervene["intervention_effect"] = effects.flatten()
        if cate_alpha is not None:
            df_intervene["intervention_effect_low"] = lb.flatten()
            df_intervene["intervention_effect_high"] = ub.flatten()
        df_intervene = df_intervene[
            df[current_intervention_column] != df[new_intervention_column]
        ]

        return df_intervene


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
        intervention_task_params: Optional[dict] = None,
        kfolds: int = 1,
        cross_validation_max_concurrency: int = 1,
        current_intervention_column: Optional[str] = None,
        new_intervention_column: Optional[str] = None,
        cate_alpha: Optional[float] = None,
        common_causes: List[str] = [],
        causal_cv: int = 5,
        causal_hyperparameters: Optional[dict] = None,
        residuals_hyperparameters: Optional[dict] = None,
        drop_duplicates: bool = True,
        return_residuals: bool = False,
        kde_steps: int = 10,
        num_gpus: int = 0,
    ):
        """
        Run this regression task and return results.

        Parameters
        ----------
        df
            Input data frame
        target
            Target columns in df. If there are emtpy values in this columns, predictions will be generated for
            these rows.
        features
            A list of features to be used for prediction. If None, all columns except target are used as features
        biased_groups
            A list of columns of groups that should be protected from biases (e.g. gender, race, age)
        debiased_features
            A list of proxy features that need to be debiased for protection of sensitive groups
        eval_metric
            Metric to be optimized during training. Possible values include 'root_mean_squared_error',
            'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'
        validation_ratio
            The ratio to randomly split data for training and validation
        prediction_quantile_low
            Lower quantile for quantile regression (in percentage)
        prediction_quantile_high
            Higher quantile for quantile regression (in percentage)
        explain_samples
            If true, explanations for predictions in test and validation will be generated. It takes significantly
            longer time to run.
        model_directory
            Destination to store trained model. If not set, a temporary folder will be created
        presets
            Autogluon's presets for training model. More details at
            https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit
        intervention_task_params
            Task parameters to be passed to build intervention model and predictions
        kfolds
            Number of folds for cross validation. If 1, train test split is used instead.
        cross_validation_max_concurrency
            Maximum number of Ray actors used for cross validation (each actor execute for one split)
        current_intervention_column
            Current values of intervention for counterfactual prediction
        new_intervention_column
            New values of intervention for counterfactual prediction
        cate_alpha
            Alpha value for CATE (conditional average treatment effect) quantiles
        common_causes
            List of columns to be used as common causes in causal graph of intervention model
        causal_cv
            Number of folds used in nuisance models in intervention model
        causal_hyperparameters
            Autogluon's hyperparameters used in nuisance models of counterfactual prediction
        residuals_hyperparameters
            Autogluon's hyperparameteres used in final model of counterfactual predictions
        drop_duplicates
            Whether duplicate values should be dropped before training
        return_residuals
            Whether residual values should be returned in counterfactual prediction
        kde_steps
            Steps used to generate KDE plots with debiasing
        num_gpus
            Number of GPUs used in nuisnace models in counterfactual prediction

        Returns
        -------


        """
        import os
        import ray
        import psutil
        import time
        import json
        from tempfile import mkdtemp
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KernelDensity
        from autogluon.tabular import TabularPredictor
        from actableai.regression.quantile import ag_quantile_hyperparameters
        from actableai.utils import (
            memory_efficient_hyperparameters,
            preprocess_data_for_shap,
        )
        from actableai.data_validation.params import RegressionDataValidator
        from actableai.data_validation.base import (
            CheckLevels,
            UNIQUE_CATEGORY_THRESHOLD,
        )
        from actableai.regression.cross_validation import run_cross_validation
        from actableai.tasks.regression import (
            _AAIInterventionTask,
            _AAIRegressionTrainTask,
        )
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
        if intervention_task_params is None:
            intervention_task_params = {}

        run_debiasing = len(biased_groups) > 0 and len(debiased_features) > 0

        if prediction_quantile_low is not None or prediction_quantile_high is not None:
            hyperparameters = ag_quantile_hyperparameters(
                prediction_quantile_low, prediction_quantile_high
            )

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

        if hyperparameters is None:
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
                predict_shap_values,
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
                predict_shap_values,
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
            )

        # Validation
        eval_shap_values = []
        if kfolds <= 1:
            df_val[target + "_predicted"] = y_pred
            if prediction_quantile_low is not None:
                df_val[target + "_low"] = predictor.predict(
                    df_val, quantile=prediction_quantile_low
                )
            if prediction_quantile_high is not None:
                df_val[target + "_high"] = predictor.predict(
                    df_val, quantile=prediction_quantile_high
                )

            predictor.unpersist_models()

            if explain_samples:
                eval_shap_values = explainer.shap_values(
                    preprocess_data_for_shap(df_val[features + biased_groups])
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
            kde_bandwidth = lambda x: max(0.5 * x.std() * (x.size ** (-0.2)), 1e-2)

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

        data = {
            "validation_table": df_val if kfolds <= 1 else None,
            "prediction_table": df_predict,
            "predict_shaps": predict_shap_values,
            "evaluate": evaluate,
            "validation_shaps": eval_shap_values,
            "importantFeatures": important_features,
            "debiasing_charts": debiasing_charts,
            "leaderboard": leaderboard,
        }

        if (
            current_intervention_column is not None
            and new_intervention_column is not None
        ):
            # Counterfactual predictions
            intervention_task = _AAIInterventionTask(**intervention_task_params)
            df_intervene = intervention_task.run(
                df,
                df_predict,
                target,
                run_model,
                current_intervention_column,
                new_intervention_column,
                common_causes,
                causal_cv,
                causal_hyperparameters,
                cate_alpha,
                presets,
                model_directory,
                num_gpus,
            )
            data["intervention_table"] = df_intervene

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
        }
