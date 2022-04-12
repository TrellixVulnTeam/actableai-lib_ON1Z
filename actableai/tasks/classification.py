from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class _AAIClassificationTrainTask(AAITask):
    """
    TODO write documentation
    """

    @AAITask.run_with_ray_remote(TaskType.CLASSIFICATION_TRAIN)
    def run(
        self,
        problem_type,
        positive_label,
        presets,
        hyperparameters,
        model_directory,
        target,
        features,
        df_train,
        df_val,
        drop_duplicates,
        run_debiasing,
        biased_groups,
        debiased_features,
        residuals_hyperparameters,
        num_gpus,
    ):
        """
        TODO write documentation
        """
        import os
        import pandas as pd
        import numpy as np
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        from sklearn.model_selection import train_test_split
        from actableai.debiasing.debiasing_model import DebiasingModel
        from actableai.utils import debiasing_feature_generator_args

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

        # Start training
        predictor = TabularPredictor(
            label=target, problem_type=problem_type, path=model_directory
        )
        predictor = predictor.fit(
            train_data=df_train,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit=ag_args_fit,
            feature_generator=AutoMLPipelineFeatureGenerator(**feature_generator_args),
        )
        predictor.persist_models()
        leaderboard = predictor.leaderboard()
        pd.set_option("chained_assignment", "warn")

        # Evaluate results
        important_features = []
        for feature, importance in predictor.feature_importance(df_val)[
            "importance"
        ].iteritems():
            if feature in biased_groups:
                continue

            important_features.append({"feature": feature, "importance": importance})

        label_val = df_val[target]
        label_pred = predictor.predict(df_val)
        perf = predictor.evaluate_predictions(
            y_true=label_val, y_pred=label_pred, auxiliary_metrics=True
        )
        pred_prob_val = predictor.predict_proba(df_val, as_multiclass=True)

        evaluate = {
            "problem_type": predictor.problem_type,
            "accuracy": perf["accuracy"],
        }
        evaluate["labels"] = predictor.class_labels
        evaluate["confusion_matrix"] = confusion_matrix(
            label_val, label_pred, labels=evaluate["labels"], normalize="true"
        ).tolist()

        if evaluate["problem_type"] == "binary":
            pos_label = (
                positive_label if positive_label is not None else evaluate["labels"][1]
            )
            neg_label = (
                evaluate["labels"][0]
                if evaluate["labels"][0] != pos_label
                else evaluate["labels"][1]
            )
            fpr, tpr, thresholds = roc_curve(
                label_val, pred_prob_val[pos_label], pos_label=pos_label
            )
            evaluate["auc_score"] = auc(fpr, tpr)
            evaluate["auc_curve"] = {
                "False Positive Rate": fpr.tolist(),
                "True Positive Rate": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "positive_label": str(pos_label),
                "negative_label": str(neg_label),
                "threshold": 0.5,
            }

        return predictor, important_features, evaluate, pred_prob_val, leaderboard


class AAIClassificationTask(AAITask):
    """
    Classification Task
    """

    @AAITask.run_with_ray_remote(TaskType.CLASSIFICATION)
    def run(
        self,
        df,
        target,
        features=None,
        biased_groups=None,
        debiased_features=None,
        validation_ratio=0.2,
        positive_label=None,
        explain_samples=False,
        model_directory=None,
        presets="medium_quality_faster_train",
        hyperparameters=None,
        train_task_params=None,
        kfolds=1,
        cross_validation_max_concurrency=1,
        residuals_hyperparameters=None,
        drop_duplicates=True,
        num_gpus=0,
    ):
        """
        TODO write documentation
        """
        import os
        import json
        import time
        import ray
        import psutil
        from tempfile import mkdtemp
        import pandas as pd
        import numpy as np
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OrdinalEncoder
        from actableai.utils import (
            memory_efficient_hyperparameters,
            handle_boolean_features,
            preprocess_dataset,
            explain_predictions,
            create_explainer,
            gen_anchor_explanation,
            debiasing_hyperparameters,
        )
        from actableai.data_validation.params import ClassificationDataValidator
        from actableai.data_validation.base import (
            CheckLevels,
            CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE,
            UNIQUE_CATEGORY_THRESHOLD,
        )
        from actableai.classification.cross_validation import run_cross_validation
        from actableai.utils.sanitize import sanitize_timezone

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

        use_cross_validation = kfolds > 1
        run_debiasing = len(biased_groups) > 0 and len(debiased_features) > 0

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
            hyperparameters = memory_efficient_hyperparameters()

        # Split data
        df_train = df[pd.notnull(df[target])]
        if drop_duplicates:
            df_train = df_train.drop_duplicates(subset=features + [target])
        df_train = df_train.groupby(target).filter(
            lambda x: len(x) >= CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE
        )
        df_val = None
        if not use_cross_validation:
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
        # Train
        classification_train_task = _AAIClassificationTrainTask(**train_task_params)
        if kfolds > 1:
            (
                predictor,
                important_features,
                evaluate,
                pred_prob_val,
                df_val,
                leaderboard,
            ) = run_cross_validation(
                classification_train_task=classification_train_task,
                problem_type=problem_type,
                positive_label=positive_label,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                df_train=df_train,
                kfolds=kfolds,
                cross_validation_max_concurrency=cross_validation_max_concurrency,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
            )
        else:
            (
                predictor,
                important_features,
                evaluate,
                pred_prob_val,
                leaderboard,
            ) = classification_train_task.run(
                problem_type=problem_type,
                positive_label=positive_label,
                presets=presets,
                hyperparameters=hyperparameters,
                model_directory=model_directory,
                target=target,
                features=features,
                df_train=df_train,
                df_val=df_val,
                drop_duplicates=drop_duplicates,
                run_debiasing=run_debiasing,
                biased_groups=biased_groups,
                debiased_features=debiased_features,
                residuals_hyperparameters=residuals_hyperparameters,
                num_gpus=num_gpus,
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

        # Construct explanations
        val_anchors = []
        predict_anchors = []
        if explain_samples and (not use_cross_validation or run_model):
            from alibi.utils.data import gen_category_map

            ordinal_encoder = None

            # Preprocess dataset
            df_full = df_train[features + biased_groups]
            if not use_cross_validation:
                df_full = df_full.append(df_val[features + biased_groups])
            df_full = df_full.append(df_test[features + biased_groups])
            df_full = preprocess_dataset(df_full)
            cat_map = gen_category_map(df_full)
            cat_cols = df_full.columns[list(cat_map.keys())]

            if len(cat_cols) > 0:
                # fit ordinal encoder
                ordinal_encoder = OrdinalEncoder(dtype=int)
                ordinal_encoder.fit(df_full[cat_cols])

            # Create explainer
            explainer = create_explainer(
                df_full,
                predictor,
                cat_map,
                encoder=ordinal_encoder,
                ncpu=int(self.ray_params.get("num_cpus", 1)),
            )
            pd.set_option("chained_assignment", "warn")

            if not use_cross_validation:
                df_val_processed = preprocess_dataset(df_val[features + biased_groups])
                val_anchors = explain_predictions(
                    df_val_processed, cat_cols, explainer, encoder=ordinal_encoder
                )

                df_val["explanation"] = [
                    gen_anchor_explanation(anchor, df_full.shape[0])
                    for anchor in val_anchors
                ]

            if run_model:
                df_predict_processed = preprocess_dataset(
                    df_test[features + biased_groups]
                )
                predict_anchors = explain_predictions(
                    df_predict_processed, cat_cols, explainer, encoder=ordinal_encoder
                )
                df_predict["explanation"] = [
                    gen_anchor_explanation(anchor, df_full.shape[0])
                    for anchor in predict_anchors
                ]

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

        predictor.unpersist_models()

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
                "predict_explanations": predict_anchors,
                "validation_explanations": val_anchors,
                "exdata": exdata["data"] if not use_cross_validation else [],
                "evaluate": evaluate,
                "importantFeatures": important_features,
                "debiasing_charts": debiasing_charts,
                "leaderboard": leaderboard,
            },
        }
