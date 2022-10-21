import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from actableai.classification.config import MINIMUM_CLASSIFICATION_VALIDATION

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class _AAIClassificationTrainTask(AAITask):
    """Sub class for ClassificationTask. Runs a classification without crossvalidation

    Args:
        AAITask: Base Class for every tasks
    """

    @AAITask.run_with_ray_remote(TaskType.CLASSIFICATION_TRAIN)
    def run(
        self,
        problem_type: str,
        explain_samples: bool,
        positive_label: Optional[str],
        presets: str,
        hyperparameters: Dict,
        model_directory: str,
        target: str,
        features: List[str],
        run_model: bool,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame],
        df_test: Optional[pd.DataFrame],
        drop_duplicates: bool,
        run_debiasing: bool,
        biased_groups: List[str],
        debiased_features: List[str],
        residuals_hyperparameters: Optional[Dict],
        num_gpus: int,
        eval_metric: str,
        time_limit: Optional[int],
        drop_unique: bool,
        drop_useless_features: bool,
        feature_pruning: bool,
    ) -> Tuple[
        Any,
        Any,
        Optional[List],
        Optional[dict],
        Optional[np.ndarray],
        Union[np.ndarray, List],
        pd.DataFrame,
    ]:
        """Runs a sub Classification Task for cross-validation.

        Args:
            problem_type: Problem type of the classification
                - _binary_: Runs a binary classification
                - _multiclass_: Runs a multiclass classification
            explain_samples: Whether we explain the samples or not.
            positive_label: Positive label for binary classification.
                Should be None if problem_type is _multiclass_
            presets: Presets for AutoGluon predictor.
                See https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            hyperparameters: Hyperparameters for AutoGluon predictor
                See https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            model_directory: Directory to save AutoGluon model
                See https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor
            target: GroundTruth column in DataFrame for training
            features: Features in DataFrame used for training
            run_model: Whether the model should be run on df_test or not.
            df_train: DataFrame used for training.
            df_val: DataFrame used for Validation. Must contain the same
                columns as df_trains
            df_test: DataFrame for testing.
            drop_duplicates: If True drops the duplicated rows to avoid overfitting in
                validation
            run_debiasing: If True debias the debiased_features with biased_groups
            biased_groups: Groups removed from the DataFrame because they introduce a
                bias in the result.
            debiased_features: Debiased features from biased_groups
            residuals_hyperparameters: Hyperparameters for AutoGluon's debiasing
                TabularPredictor
                See https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor
            num_gpus: Number of gpus used by AutoGluon
            eval_metric: Metric to be optimized for.
            time_limit: Time limit for training (in seconds)
            drop_unique: Whether the classification algorithm drops columns that
                only have a unique value accross all rows at fit time
            drop_useless_features: Whether the classification algorithm drops columns that
                only have a unique value accross all rows as preprocessing
            feature_pruning: Wether the feature_pruning is enabled or not.
                This option improves results but extend the training time.
                If there is no time left to do feature_pruning after training
                this step is skipped.

        Returns:
            Tuple[
                Any,
                Any,
                Optional[List],
                Optional[dict],
                Optional[np.ndarray],
                Union[np.ndarray, List],
                pd.DataFrame
            ]: Return results for classification :
                - AutoGluon's predictor
                - Explainer for SHAP values
                - List of important features
                - Dictionnary of evaluated metrics
                - Class probabilities for predicted values
                - Shap values on test set
                - Leaderboard of the best trained models
        """
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
            roc_curve,
        )

        from actableai.utils import custom_precision_recall_curve
        from actableai.debiasing.debiasing_model import DebiasingModel
        from actableai.utils import debiasing_feature_generator_args
        from actableai.explanation.autogluon_explainer import AutoGluonShapTreeExplainer

        ag_args_fit = {"drop_unique": drop_unique}
        feature_generator_args = {}

        if "AG_AUTOMM" in hyperparameters:
            feature_generator_args["enable_raw_text_features"] = True

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
            ag_args_fit["drop_useless_features"] = False

            feature_generator_args = {
                **feature_generator_args,
                **debiasing_feature_generator_args(),
            }

            hyperparameters = {DebiasingModel: {}}

        ag_args_fit["num_cpus"] = 1
        ag_args_fit["num_gpus"] = num_gpus

        df_train = df_train[features + biased_groups + [target]]
        if df_val is not None:
            df_val = df_val[features + biased_groups + [target]]
        if df_test is not None:
            df_test = df_test[features + biased_groups + [target]]

        # Start training
        predictor = TabularPredictor(
            label=target,
            problem_type=problem_type,
            path=model_directory,
            eval_metric=eval_metric,
        )

        feature_prune_kwargs = None
        if feature_pruning:
            feature_prune_kwargs = {}
            if time_limit is not None:
                feature_prune_kwargs["feature_prune_time_limit"] = time_limit * 0.5

        holdout_frac = max(
            len(df_train[target].unique()) / len(df_train),
            MINIMUM_CLASSIFICATION_VALIDATION,
        )

        predictor = predictor.fit(
            train_data=df_train,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit=ag_args_fit,
            feature_generator=AutoMLPipelineFeatureGenerator(**feature_generator_args),
            time_limit=time_limit,
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            feature_prune_kwargs=feature_prune_kwargs,
            holdout_frac=holdout_frac,
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

        # Evaluate results
        important_features = None
        evaluate = None
        pred_prob_val = None
        if df_val is not None:
            important_features = []
            feature_importance = predictor.feature_importance(df_val)
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
            label_val = df_val[target]
            label_pred = predictor.predict(df_val)
            pred_prob_val = predictor.predict_proba(df_val, as_multiclass=True)
            perf = predictor.evaluate_predictions(
                y_true=label_val,
                y_pred=pred_prob_val,
                auxiliary_metrics=True,
                detailed_report=False,
            )

            evaluate = {
                # TODO: to be removed (legacy)
                "problem_type": predictor.problem_type,
                "accuracy": perf["accuracy"],
                "metrics": pd.DataFrame(
                    {"metric": perf.keys(), "value": perf.values()}
                ),
            }
            evaluate["labels"] = predictor.class_labels
            evaluate["confusion_matrix"] = confusion_matrix(
                label_val, label_pred, labels=evaluate["labels"], normalize="true"
            ).tolist()

            if evaluate["problem_type"] == "binary":
                if evaluate["labels"][0] in [
                    1,
                    1.0,
                    "1",
                    "1.0",
                    "true",
                    "yes",
                    positive_label,
                ]:
                    pos_label = evaluate["labels"][0]
                    neg_label = evaluate["labels"][1]
                else:
                    pos_label = evaluate["labels"][1]
                    neg_label = evaluate["labels"][0]

                evaluate["confusion_matrix"] = confusion_matrix(
                    label_val,
                    label_pred,
                    labels=[pos_label, neg_label],
                    normalize="true",
                )

                fpr, tpr, thresholds = roc_curve(
                    label_val,
                    pred_prob_val[pos_label],
                    pos_label=pos_label,
                    drop_intermediate=False,
                )
                evaluate["auc_curve"] = {
                    "False Positive Rate": fpr[1:].tolist()[::-1],
                    "True Positive Rate": tpr[1:].tolist()[::-1],
                    "thresholds": thresholds[1:].tolist()[::-1],
                    "positive_label": str(pos_label),
                    "negative_label": str(neg_label),
                    "threshold": 0.5,
                }
                evaluate["precision_score"] = precision_score(
                    label_val, label_pred, pos_label=pos_label
                )
                evaluate["recall_score"] = recall_score(
                    label_val, label_pred, pos_label=pos_label
                )
                precision, recall, thresholds = custom_precision_recall_curve(
                    label_val, pred_prob_val[pos_label], pos_label=pos_label
                )
                evaluate["precision_recall_curve"] = {
                    "Precision": precision.tolist(),
                    "Recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                    "positive_label": str(pos_label),
                    "negative_label": str(neg_label),
                }
                evaluate["f1_score"] = f1_score(
                    label_val, label_pred, pos_label=pos_label
                )
                evaluate["positive_count"] = len(label_val[label_val == pos_label])
                evaluate["negative_count"] = len(label_val) - evaluate["positive_count"]
            else:
                evaluate["auc_curve"] = []
                evaluate["precision_recall_curve"] = []
                for pos_label in evaluate["labels"]:
                    fpr, tpr, thresholds = roc_curve(
                        label_val,
                        pred_prob_val[pos_label],
                        pos_label=pos_label,
                        drop_intermediate=False,
                    )
                    evaluate["auc_curve"].append(
                        {
                            "False Positive Rate": fpr[1:].tolist()[::-1],
                            "True Positive Rate": tpr[1:].tolist()[::-1],
                            "thresholds": thresholds[1:].tolist()[::-1],
                            "positive_label": str(pos_label),
                            "threshold": 0.5,
                        }
                    )
                    precision, recall, thresholds = custom_precision_recall_curve(
                        label_val, pred_prob_val[pos_label], pos_label=pos_label
                    )
                    evaluate["precision_recall_curve"].append(
                        {
                            "Precision": precision.tolist(),
                            "Recall": recall.tolist(),
                            "thresholds": thresholds.tolist(),
                            "positive_label": str(pos_label),
                        }
                    )

        predict_shap_values = []

        if run_model and explainer is not None and df_test is not None:
            predict_shap_values = explainer.shap_values(df_test)

        return (
            predictor,
            explainer,
            important_features,
            evaluate,
            pred_prob_val,
            predict_shap_values,
            leaderboard,
        )


class AAIClassificationTask(AAITask):
    """AAIClassificationTask class for classification

    Args:
        AAITask: Base class for every tasks
    """

    @AAITask.run_with_ray_remote(TaskType.CLASSIFICATION)
    def run(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        biased_groups: Optional[List[str]] = None,
        debiased_features: Optional[List[str]] = None,
        validation_ratio: float = 0.2,
        positive_label: Optional[str] = None,
        explain_samples: bool = False,
        model_directory: Optional[str] = None,
        presets: str = "medium_quality_faster_train",
        hyperparameters: Optional[Dict] = None,
        train_task_params: Optional[Dict] = None,
        kfolds: int = 1,
        cross_validation_max_concurrency: int = 1,
        residuals_hyperparameters: Optional[Dict] = None,
        drop_duplicates: bool = True,
        num_gpus: int = 0,
        eval_metric: str = "accuracy",
        time_limit: Optional[int] = None,
        drop_unique: bool = True,
        drop_useless_features: bool = True,
        split_by_datetime: bool = False,
        datetime_column: Optional[str] = None,
        ag_automm_enabled=False,
        refit_full=False,
        feature_pruning=True,
        intervention_run_params: Optional[Dict] = None,
    ) -> Dict:
        """Run this classification task and return results.

        Args:
            df: Input DataFrame
            target: Target columns in df. If there are emtpy values in this columns,
                predictions will be generated for these rows.
            features: A list of features to be used for prediction. If None, all columns
                except target are used as features. Defaults to None.
            biased_groups: A list of columns of groups that should be protected from
                biases (e.g. gender, race, age). Defaults to None.
            debiased_features: A list of proxy features that need to be debiased for
                protection of sensitive groups. Defaults to None.
            validation_ratio: The ratio to randomly split data for training and
                validation. Defaults to 0.2.
            positive_label: If target contains only 2 different value, pick the positive
                label by setting postive_label to one of them. Defaults to None.
            explain_samples: If true, explanations for predictions in test and
                validation will be generated. It takes significantly longer time to
                run. Defaults to False.
            model_directory: Directory to output the model after training.
                Defaults to None.
            presets: Autogluon's presets for training model. More details at
                https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor.fit.
                Defaults to "medium_quality_faster_train".
            hyperparameters: Autogluon's hyperparameters. Defaults to None.
            train_task_params: ?. Defaults to None.
            kfolds: Number of fold for cross-validation. Defaults to 1.
            cross_validation_max_concurrency: Maximum number of Ray actors used for
                cross validation (each actor execute for one split). Defaults to 1.
            residuals_hyperparameters: Autogluon's hyperparameteres used in final model
                of counterfactual predictions. Defaults to None.
            drop_duplicates: Whether duplicate values should be dropped before training.
                Defaults to True.
            num_gpus: Number of gpus used for training. Defaults to 0.
            eval_metric: Metric to be optimized for. Possible values include ‘accuracy’, ‘balanced_accuracy’, ‘f1’,
                ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, ‘roc_auc’, ‘roc_auc_ovo_macro’, ‘average_precision’,
                ‘precision’, ‘precision_macro’, ‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’,
                ‘recall_micro’, ‘recall_weighted’, ‘log_loss’, ‘pac_score’.
                Defaults to "accuracy".
            time_limit: Time limit of training (in seconds)
            drop_unique: Wether to drop columns with only unique values as preprocessing step.
            drop_useless_features: Whether to drop columns with only unique values at fit time.
            split_by_datetime: Whether the training/validation has to be split based on a datetime column.
            datetime_column: If *split_by_datetime*, the column that will split training and validation,
                else, the parameter is ignored.
            ag_automm_enabled: Whether to use autogluon multimodal model on text
                columns. This features makes text classification way more accurate by using
                text models. This feature is heavy on resources and requires GPU.
            refit_full: Whether at the end of classification, a second task is launched to
                refit a new model on the whole dataset. This makes accuracy much better but divides
                the training time in half. (half for first task, other half for refitting)
            feature_pruning: Wether the feature_pruning is enabled or not.
                This option improves results but extend the training time.
                If there is no time left to do feature_pruning after training
                this step is skipped.

        Raises:
            Exception: If the target has less than 2 unique values.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> AAIClassificationTask(df, ["feature1", "feature2", "feature3"], "target")

        Returns:
            Dict: Dictionnary containing the results
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Message returned with the task
                - "validations": List of validations on the data.
                    non-empty if the data presents a problem for the task
                - "runtime": Execution time of the task
                - "data": Dictionnary containing the data for the task
                    - "validation_table": Validation table
                    - "prediction_table": Prediction table
                    - "fields": Column names of the prediction table
                    - "predictData": Prediction Table
                    - "predict_shaps": Shapley values for prediction table
                    - "validation_shaps": Shapley values for validation table
                    - "exdata": Validation Table
                    - "evaluate": Evaluation metrics on validation set
                    - "importantFeatures": Feature importance on validation set
                    - "debiasing_charts": If debiasing enabled, debiasing data to create charts
                    - "leaderboard": Leaderboard of the best model on validation
                - "model": AAIModel to redeploy the model
        """
        import json
        import time
        from tempfile import mkdtemp
        import pandas as pd
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from autogluon.common.features.infer_types import check_if_nlp_feature

        from actableai.utils import (
            memory_efficient_hyperparameters,
            handle_boolean_features,
            explanation_hyperparameters,
        )
        from actableai.data_validation.params import ClassificationDataValidator
        from actableai.data_validation.base import (
            CheckLevels,
            CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE,
            UNIQUE_CATEGORY_THRESHOLD,
        )
        from actableai import AAIInterventionTask
        from actableai.models.aai_predictor import (
            AAITabularModel,
            AAITabularModelInterventional,
        )
        from actableai.classification.cross_validation import run_cross_validation
        from actableai.utils.sanitize import sanitize_timezone
        from actableai.classification.utils import split_validation_by_datetime

        pd.set_option("chained_assignment", "warn")
        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        # Handle default parameters
        if features is None:
            features = df.columns.drop(target, errors="ignore").tolist()
        if biased_groups is None:
            biased_groups = []
        if debiased_features is None:
            debiased_features = []
        if model_directory is None:
            model_directory = mkdtemp(prefix="autogluon_model")
        if train_task_params is None:
            train_task_params = {}
        if refit_full and time_limit is not None:
            # Half the time limit for train and half the time for refit
            time_limit = time_limit // 2

        use_cross_validation = kfolds > 1
        run_debiasing = len(biased_groups) > 0 and len(debiased_features) > 0

        if run_debiasing and drop_useless_features:
            drop_useless_features = False
            logging.warning(
                "`drop_useless_features` is set to False: `run_debiasing` is True"
            )

        # Validate parameters
        data_validation_results = ClassificationDataValidator().validate(
            target,
            features,
            biased_groups,
            debiased_features,
            df,
            presets,
            validation_ratio=validation_ratio,
            kfolds=kfolds,
            drop_duplicates=drop_duplicates,
            explain_samples=explain_samples,
            eval_metric=eval_metric,
            drop_unique=drop_unique,
            datetime_column=datetime_column,
            split_by_datetime=split_by_datetime,
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

        # Pre process data
        df = handle_boolean_features(df)

        if hyperparameters is None:
            if explain_samples:
                hyperparameters = explanation_hyperparameters()
            else:
                any_text_cols = df.apply(check_if_nlp_feature).any(axis=None)
                hyperparameters = memory_efficient_hyperparameters(
                    ag_automm_enabled and any_text_cols
                )

        from actableai.utils import get_type_special

        # If the types are mixed, the train_test_split function with stratify crashes
        if get_type_special(df[target]) == "mixed":
            df[target] = df[target].astype(str)

        # Split data
        df_train = df[pd.notnull(df[target])]
        if drop_duplicates:
            df_train = df_train.drop_duplicates(subset=features + [target])
        df_train = df_train.groupby(target).filter(
            lambda x: len(x) >= CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE
        )
        df_val = None
        if not use_cross_validation:
            if split_by_datetime and datetime_column is not None:
                df_train, df_val = split_validation_by_datetime(
                    df_train=df_train,
                    datetime_column=datetime_column,
                    validation_ratio=validation_ratio,
                )
                sorted_df = df_train.sort_values(by=datetime_column, ascending=True)
                split_datetime_index = int((1 - validation_ratio) * len(sorted_df))
                df_train = sorted_df.iloc[:split_datetime_index].sample(frac=1)
                df_val = sorted_df.iloc[split_datetime_index:]
            else:
                df_train, df_val = train_test_split(
                    df_train, test_size=validation_ratio, stratify=df_train[target]
                )

        df_test = df[pd.isnull(df[target])]

        # If true it means that the classification needs to be run on test data
        run_model = df_test.shape[0] > 0

        # Check classification type
        if df[target].nunique() == 2:
            problem_type = "binary"
        elif df[target].nunique() > 2:
            problem_type = "multiclass"
        else:
            # TODO proper exception
            raise Exception()

        leaderboard = None
        explainer = None
        # Train
        classification_train_task = _AAIClassificationTrainTask(**train_task_params)
        if kfolds > 1:
            (
                predictor,
                important_features,
                evaluate,
                pred_prob_val,
                predict_shap_values,
                df_val,
                leaderboard,
            ) = run_cross_validation(
                classification_train_task=classification_train_task,
                problem_type=problem_type,
                explain_samples=explain_samples,
                positive_label=positive_label,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_test=df_test,
                kfolds=kfolds,
                cross_validation_max_concurrency=cross_validation_max_concurrency,
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
                feature_pruning=feature_pruning,
            )
        else:
            (
                predictor,
                explainer,
                important_features,
                evaluate,
                pred_prob_val,
                predict_shap_values,
                leaderboard,
            ) = classification_train_task.run(
                problem_type=problem_type,
                explain_samples=explain_samples,
                positive_label=positive_label,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=run_model,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
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
                feature_pruning=feature_pruning,
            )

        if not use_cross_validation:
            for c in predictor.class_labels:
                df_val[str(c) + " probability"] = pred_prob_val[c]
            df_val[target + "_predicted"] = pred_prob_val.idxmax(axis=1)

        # Run predictions on test if needed
        pred_prob = None
        df_predict = df_test.copy()
        if run_model:
            pred_prob = predictor.predict_proba(df_test, as_multiclass=True)
            for c in predictor.class_labels:
                df_predict[str(c) + " probability"] = pred_prob[c]
            df_predict[target] = pred_prob.idxmax(axis=1)

        # Validation
        eval_shap_values = []
        if kfolds <= 1 and explain_samples:
            eval_shap_values = explainer.shap_values(df_val[features + biased_groups])

        debiasing_charts = []
        # Generate debiasing charts
        if run_debiasing:
            plot_targets = [target, f"{target}_predicted"]

            for biased_group in biased_groups:
                group_charts = []
                group_chart_type = None

                df_val_biased_group = df_val[biased_group].fillna("NaN").astype(str)
                biased_classes = sorted(df_val_biased_group.unique())

                if len(biased_classes) <= UNIQUE_CATEGORY_THRESHOLD:
                    # Categorical Biased Group => Bar plot
                    group_chart_type = "bar"

                    for plot_target in plot_targets:
                        df_values = pd.DataFrame()

                        for biased_class in biased_classes:
                            values = df_val[df_val_biased_group == biased_class][
                                plot_target
                            ]
                            values = values.value_counts(normalize=True)

                            df_values[biased_class] = values

                        df_values.sort_index(inplace=True)
                        corr, pvalue = spearmanr(
                            df_val[biased_group], df_val[plot_target]
                        )
                        group_charts.append(
                            {
                                "x_label": plot_target,
                                "y": df_values.columns.tolist(),
                                "bars": [
                                    {"x": values.tolist(), "name": target_class}
                                    for target_class, values in df_values.iterrows()
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

        predict_data = json.loads(df_predict.to_json(orient="table"))
        predict_data["schema"]["fields"].pop(0)

        exdata = None
        if not use_cross_validation:
            exdata = json.loads(df_val.to_json(orient="table"))
            exdata["schema"]["fields"].pop(0)

        # FIXME we need better logic here, this is done so we can return models
        if kfolds <= 1:
            predictor.persist_models()
        else:
            predictor.unpersist_models()

        leaderboard_obj_cols = leaderboard.select_dtypes(include=["object"]).columns
        leaderboard[leaderboard_obj_cols] = leaderboard[leaderboard_obj_cols].astype(
            str
        )

        validations = [
            {"name": x.name, "level": x.level, "message": x.message}
            for x in failed_checks
        ]
        aai_intervention_model = None
        if intervention_run_params is not None:
            intervention_run_params["target_proba"] = predictor.predict_proba(df)
            intervention_task_result = AAIInterventionTask(
                return_model=True, upload_model=False
            ).run(**intervention_run_params)
            if intervention_task_result["status"] == "SUCCESS":
                aai_intervention_model = intervention_task_result["model"]
            else:
                validations.append(
                    {
                        "name": "Intervention Failed",
                        "level": CheckLevels.WARNING,
                        "message": "Counterfactual ran into an issue",
                    }
                )

        if refit_full:
            df_only_training = df.loc[df[target].notnull()]
            predictor, _, _, _, _, _, _ = _AAIClassificationTrainTask(
                **train_task_params
            ).run(
                explain_samples=False,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                run_model=False,
                df_train=df_only_training,
                df_val=None,
                df_test=None,
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
                problem_type=problem_type,
                positive_label=positive_label,
                feature_pruning=feature_pruning,
            )
            predictor.refit_full(model="best", set_best_to_refit_full=True)

        model = None
        if (kfolds <= 1 or refit_full) and predictor:
            model = AAITabularModel(
                predictor=predictor
            )
            if aai_intervention_model is not None:
                model = AAITabularModelInterventional(
                    predictor=predictor,
                    intervention_model=aai_intervention_model,
                )

        runtime = time.time() - start
        return {
            "messenger": "",
            "status": "SUCCESS",
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "runtime": runtime,
            "data": {
                "validation_table": df_val if not use_cross_validation else None,
                "prediction_table": df_predict,
                "fields": predict_data["schema"]["fields"],
                "predictData": predict_data["data"],
                "predict_shaps": predict_shap_values,
                "validation_shaps": eval_shap_values,
                "exdata": exdata["data"] if not use_cross_validation else [],
                "evaluate": evaluate,
                "importantFeatures": important_features,
                "debiasing_charts": debiasing_charts,
                "leaderboard": leaderboard,
            },
            "model": model,
            # FIXME this predictor is not really usable as is for now
        }
