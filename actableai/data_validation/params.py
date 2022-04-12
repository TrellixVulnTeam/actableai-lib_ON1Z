from logging import CRITICAL
from typing import Union
from actableai.data_validation.checkers import *
from actableai.causal import has_categorical_column, prepare_sanitize_data
from actableai.data_validation.base import CheckLevels, CheckResult, IChecker


class RegressionDataValidator:
    def validate(
        self,
        target,
        features,
        df,
        debiasing_features,
        debiased_features,
        eval_metric="r2",
        prediction_quantile_low=None,
        prediction_quantile_high=None,
        presets="medium_quality_faster_train",
        explain_samples=False,
        drop_duplicates=True,
    ):
        if drop_duplicates:
            df = df.drop_duplicates(subset=features + [target])

        validation_results = [
            RegressionEvalMetricChecker(level=CheckLevels.CRITICAL).check(eval_metric),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, [target]),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, features
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, [target]
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, features),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                [target], debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                features, debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                debiased_features, debiasing_features
            ),
            ColumnsInList(level=CheckLevels.CRITICAL).check(
                features, debiased_features
            ),
        ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsNumericalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=REGRESSION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

        # Check debiasing
        if len(debiasing_features) > 0 and len(debiased_features) <= 0:
            validation_results.append(
                CheckResult(
                    name="DebiasingChecker",
                    level=CheckLevels.CRITICAL,
                    message="At least one debiasing features must be selected",
                )
            )

        run_debiasing = len(debiasing_features) > 0 and len(debiased_features) > 0
        prediction_intervals = (
            prediction_quantile_low is not None or prediction_quantile_high is not None
        )
        # Check prediction intervals
        if run_debiasing and prediction_intervals:
            validation_results.append(
                CheckResult(
                    name="PredictionIntervalChecker",
                    level=CheckLevels.CRITICAL,
                    message="Debiasing is incompatible with prediction intervals",
                )
            )

        # Check presets
        # Note: best_quality is incompatible with debiasing because it activates bagging in AutoGluonj
        if run_debiasing and presets == "best_quality":
            validation_results.append(
                CheckResult(
                    name="PresetsChecker",
                    level=CheckLevels.CRITICAL,
                    message="Optimize for performance is incompatible with debiasing",
                )
            )

        if run_debiasing:
            validation_results.append(
                DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(
                    df, debiasing_features + debiased_features
                )
            )

        return validation_results


class BayesianRegressionDataValidator:
    def validate(
        self, target: str, features: List[str], df: pd.DataFrame, polynomial_degree: int
    ) -> List:

        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, [target]),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, features
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, [target]
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, features),
            # We do not run for more than n_unique_level unique categorical values
            CheckNUnique(level=CheckLevels.CRITICAL).check(
                df=df,
                n_unique_level=EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT,
                analytics="Bayesian Regression",
            ),
            CheckColumnInflateLimit(level=CheckLevels.CRITICAL).check(
                df, features, polynomial_degree, POLYNOMIAL_INFLATE_COLUMN_LIMIT
            ),
        ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsNumericalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=REGRESSION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

        return validation_results


class ClassificationDataValidator:
    def validate(
        self,
        target,
        features,
        debiasing_features,
        debiased_features,
        df,
        presets,
        validation_ratio=None,
        kfolds=None,
        drop_duplicates=True,
        explain_samples=False,
    ):
        if drop_duplicates:
            df = df.drop_duplicates(subset=features + [target])

        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df, [target, *features, *debiasing_features, *debiased_features]
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, df.columns
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, features),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                [target], debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                features, debiasing_features
            ),
            ColumnsNotInList(level=CheckLevels.CRITICAL).check(
                debiased_features, debiasing_features
            ),
            ColumnsInList(level=CheckLevels.CRITICAL).check(
                features, debiased_features
            ),
        ]

        if explain_samples:
            # We do not explain samples for more than n_unique_level categorical values
            validation_results += [
                CheckNUnique(level=CheckLevels.CRITICAL).check(
                    df=df, n_unique_level=EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT
                )
            ]

        if target in df.columns and not pd.isnull(df[target]).all():
            validation_results += [
                IsCategoricalChecker(level=CheckLevels.CRITICAL).check(df[target]),
                IsSufficientNumberOfClassChecker(level=CheckLevels.CRITICAL).check(
                    df[target]
                ),
                CorrectAnalyticChecker(level=CheckLevels.WARNING).check(
                    df[target],
                    problem_type=CLASSIFICATION_ANALYTIC,
                    unique_threshold=UNIQUE_CATEGORY_THRESHOLD,
                ),
            ]

            if validation_ratio is not None:
                validation_results.append(
                    IsSufficientValidationSampleChecker(
                        level=CheckLevels.CRITICAL
                    ).check(df[target], validation_ratio),
                )

            critical_validation_count = 0
            for result in validation_results:
                if result is not None and result.level == CheckLevels.CRITICAL:
                    critical_validation_count += 1

            if critical_validation_count == 0:
                if validation_ratio is not None:
                    validation_results.append(
                        IsSufficientClassSampleChecker(level=CheckLevels.WARNING).check(
                            df, target, validation_ratio
                        )
                    )
                if kfolds > 1:
                    validation_results.append(
                        IsSufficientClassSampleForCrossValidationChecker(
                            level=CheckLevels.CRITICAL
                        ).check(df, target, kfolds)
                    )

        # Check debiasing
        if len(debiasing_features) > 0 and len(debiased_features) <= 0:
            validation_results.append(
                CheckResult(
                    name="DebiasingChecker",
                    level=CheckLevels.CRITICAL,
                    message="At least one debiasing features must be selected",
                )
            )

        run_debiasing = len(debiasing_features) > 0 and len(debiased_features) > 0
        # Check presets
        # Note: best_quality is incompatible with debiasing because it activates bagging in AutoGluonj
        if run_debiasing and presets == "best_quality":
            validation_results.append(
                CheckResult(
                    name="PresetsChecker",
                    level=CheckLevels.CRITICAL,
                    message="Optimize for performance is incompatible with debiasing",
                )
            )

        if run_debiasing:
            validation_results.append(
                DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(
                    df, debiasing_features + debiased_features
                )
            )

        return validation_results


class TimeSeriesDataValidator:
    def __init__(self):
        pass

    def validate(self, feature, target, df, prediction_length):
        return [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(
                df, [feature] + target
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, [feature] + target
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            IsDatetimeChecker(level=CheckLevels.CRITICAL).check(df[feature])
            if ((feature in df.columns) and not pd.isnull(df[feature]).all())
            else None,
            DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, target),
            IsValidPredictionLengthChecker(level=CheckLevels.CRITICAL).check(
                df, prediction_length=prediction_length
            ),
            CategoryChecker(level=CheckLevels.CRITICAL).check(df, target),
            IsValidFrequencyChecker(level=CheckLevels.CRITICAL).check(df[feature])
            if ((feature in df.columns) and not pd.isnull(df[feature]).all())
            else None,
            UniqueDateTimeChecker(level=CheckLevels.CRITICAL).check(df[feature]),
        ]


class ClusteringDataValidator:
    def __init__(self):
        pass

    def validate(self, target, df, n_cluster, explain_samples=False):
        return [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, target),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, target
            ),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
            DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, target),
            IsValidNumberOfClusterChecker(level=CheckLevels.CRITICAL).check(
                df, n_cluster=n_cluster
            ),
            IsValidTypeNumberOfClusterChecker(level=CheckLevels.CRITICAL).check(
                n_cluster
            ),
            DoNotContainDatetimeChecker(level=CheckLevels.CRITICAL).check(df[target]),
            CheckNUnique(level=CheckLevels.CRITICAL).check(
                df=df, n_unique_level=EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT
            )
            if explain_samples
            else None,
            DoNotContainTextChecker(level=CheckLevels.CRITICAL).check(df, target),
        ]


class CausalDataValidator:
    def __init__(self):
        pass

    def validate(
        self,
        treatments: List[str],
        outcomes: List[str],
        df: pd.DataFrame,
        effect_modifiers: List[str],
        common_causes: List[str],
    ) -> List[Union[CheckResult, None]]:
        columns = effect_modifiers + common_causes
        validation_results = [
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, treatments),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, outcomes),
            ColumnsExistChecker(level=CheckLevels.CRITICAL).check(df, columns),
        ]
        if len([x for x in validation_results if x is not None]) > 0:
            return validation_results
        # Columns are sane now we treat
        df = prepare_sanitize_data(
            df, treatments, outcomes, effect_modifiers, common_causes
        )
        validation_results += [
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, columns
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, treatments
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, outcomes
            ),
            DoNotContainMixedChecker(level=CheckLevels.WARNING).check(df, columns),
            IsSufficientDataChecker(level=CheckLevels.CRITICAL).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
        ]
        for t in set(treatments):
            validation_results.append(
                DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, [t])
                if t in df.columns and not pd.isnull(df[t]).all()
                else None
            )
        for y in set(outcomes):
            validation_results.append(
                DoNotContainMixedChecker(level=CheckLevels.CRITICAL).check(df, [y])
                if y in df.columns and not pd.isnull(df[y]).all()
                else None
            )

        if not CheckLevels.CRITICAL in [
            check.level for check in validation_results if check is not None
        ]:
            for treatment in treatments:
                if has_categorical_column(df, [treatment]):
                    validation_results.append(
                        InsufficientCategoricalRows(level=CheckLevels.CRITICAL).check(
                            df,
                            treatment=treatment,
                            n_rows=CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT,
                        )
                    )
        return validation_results


class CorrelationDataValidator:
    def __init__(self):
        pass

    def validate(self, df, target):
        return [
            DoNotContainEmptyColumnsChecker(level=CheckLevels.WARNING).check(
                df, df.columns
            ),
            DoNotContainEmptyColumnsChecker(level=CheckLevels.CRITICAL).check(
                df, target
            ),
        ]


class DataImputationDataValidator:
    def __init__(self):
        pass

    def validate(self, df):
        return [
            IsSufficientDataChecker(level=CheckLevels.WARNING).check(
                df, n_sample=MINIMUM_NUMBER_OF_SAMPLE
            ),
        ]
