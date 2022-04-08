from sklearn.impute import SimpleImputer
from actableai.tasks import TaskType
from actableai.tasks.base import AAITask

import pandas as pd

class AAICorrelationTask(AAITask):
    """
    Correlation Task
    """
    @AAITask.run_with_ray_remote(TaskType.CORRELATION)
    def run(self,
            df,
            target_column,
            target_value=None,
            kde_steps=100,
            lr_steps=100,
            control_columns=None,
            control_values=None,
            correlation_threshold=0.05,
            p_value=0.05,
            use_bonferroni=False,
            top_k=20):
        """
        TODO write documentation
        """
        import logging
        import time
        import pandas as pd
        import numpy as np
        from sklearn.neighbors import KernelDensity
        from sklearn.linear_model import BayesianRidge
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.feature_extraction.text import CountVectorizer
        from autogluon.features import TextNgramFeatureGenerator, DatetimeFeatureGenerator
        import nltk
        from nltk.corpus import stopwords

        from actableai.stats import Stats
        from actableai.data_validation.params import CorrelationDataValidator
        from actableai.utils import is_fitted
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.preprocessing import SKLearnAGFeatureWrapperBase
        from actableai.utils import get_type_special_no_ag

        if use_bonferroni:
            p_value /= len(df.columns) - 1

        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()

        data_validation_results = CorrelationDataValidator().validate(df, target_column)
        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return ({
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": check.name, "level": check.level, "message": check.message}
                    for check in failed_checks
                ],
                "runtime": time.time() - start,
            })

        # Type reader
        type_specials = df.apply(get_type_special_no_ag)
        cat_cols = list((type_specials == 'category') | (type_specials == 'boolean'))
        text_cols = list(type_specials == 'text')
        date_cols = list(type_specials == 'datetime')
        og_df_col = df.columns
        og_target_col = df.loc[:, cat_cols]

        # Data Transformation
        ct = ColumnTransformer([
            (OneHotEncoder.__name__, OneHotEncoder(), cat_cols),
            (
                TextNgramFeatureGenerator.__name__,
                SKLearnAGFeatureWrapperBase(
                    TextNgramFeatureGenerator(
                        vectorizer=CountVectorizer(stop_words=stopwords.words()),
                        vectorizer_strategy='separate'
                    )
                ),
                text_cols
            ),
            (DatetimeFeatureGenerator.__name__, SKLearnAGFeatureWrapperBase(DatetimeFeatureGenerator()), date_cols)
        ],
        remainder='passthrough', sparse_threshold=0, verbose_feature_names_out=False, verbose=True)
        df = pd.DataFrame(ct.fit_transform(df).tolist(), columns=ct.get_feature_names_out())

        if control_columns is not None or control_values is not None:
            if len(control_columns) != len(control_values):
                return {
                    "status": "FAILURE",
                    "messenger": "control_columns and control_values must have the same length"
                }

            for control_column, control_value in zip(control_columns, control_values):
                try:
                    idx = Stats().decorrelate(
                        df, target_column, control_column,
                        target_value=target_value, control_value=control_value)
                except:
                    logging.exception("Fail to de-correlate.")
                    return {
                        "status": "FAILURE",
                        "messenger": "Can't de-correlate {} from {}".format(control_column, target_column)
                    }
                if idx.shape[0] == 0:
                    return {
                        "status": "FAILURE",
                        "messenger": "De-correlation returns empty data"
                    }
                df = df.loc[idx]

        if df.shape[0] < 3:
            return {
                "status": "FAILURE",
                "messenger": "Not enough data to calculate correlation",
                "validations": [],
            }

        cat_cols = []
        gen_cat_cols = []
        if is_fitted(ct.named_transformers_['OneHotEncoder']):
            cat_cols = og_df_col[ct.transformers[0][2]]
            gen_cat_cols = ct.named_transformers_['OneHotEncoder'].categories_
        corrs = Stats().corr(
            df,
            target_column,
            target_value,
            p_value=p_value,
            categorical_columns=cat_cols,
            gen_categorical_columns=gen_cat_cols
        )
        corrs = corrs[:top_k]

        df = df.join(og_target_col)
        charts = []
        other = lambda uniques, label: uniques[uniques != label][0] if uniques.size == 2 else "others"
        kde_bandwidth = lambda x: max(0.5 * x.std() * (x.size ** (-0.2)), 1e-2)
        for corr in corrs:
            if type(corr["col"]) is list:
                group, val = corr["col"]

                df[group].fillna('None', inplace=True)

                # Categorical variable
                if target_value is None:
                    # Target column is continuous
                    X = np.linspace(df[target_column].min(), df[target_column].max(), kde_steps)

                    x1 = df[target_column][df[group] == val]
                    x1 = x1[x1.notna()]
                    k1 = KernelDensity(bandwidth=kde_bandwidth(x1)).fit(x1.values.reshape((-1, 1)))

                    x2 = df[target_column][df[group] != val]
                    x2 = x2[x2.notna()]
                    k2 = KernelDensity(bandwidth=kde_bandwidth(x2)).fit(x2.values.reshape((-1, 1)))

                    charts.append({
                        "type": "kde",
                        "corr": corr["corr"],
                        "data": [
                            {
                                "value": val,
                                "y": np.exp(k1.score_samples(X.reshape((-1, 1)))).tolist(),
                                "x": X.tolist(),
                                "group": group,
                                "y_label": target_column,
                            },
                            {
                                "value": other(df[group].unique(), val),
                                "y": np.exp(k2.score_samples(X.reshape(-1, 1))).tolist(),
                                "x": X.tolist(),
                                "group": group,
                                "y_label": target_column,
                            }
                        ]
                    })
                else:
                    # Target value is also categorical
                    x = df[target_column].copy()
                    x[x != target_value] = other(df[target_column].unique(), target_value)

                    y = df[group].copy()
                    y[y != val] = other(df[group], val)

                    charts.append({
                        "type": "cm",
                        "corr": corr["corr"],
                        "data": {
                            "cm": pd.crosstab(x, y, dropna=False, normalize="index").to_dict(),
                            "corr": corr,
                        }
                    })

            else:
                if target_value is None:
                    X, y = df[corr["col"]], df[target_column]
                    idx = X.notna() & y.notna()
                    X, y = X[idx], y[idx]
                    clf = BayesianRidge(compute_score=True)
                    clf.fit(X.values.reshape((-1, 1)), y)
                    r2 = clf.score(X.values.reshape((-1, 1)), y)
                    x_pred = np.linspace(X.min(), X.max(), lr_steps)
                    y_mean, y_std = clf.predict(x_pred.reshape((-1, 1)), return_std=True)
                    charts.append({
                        "type": "lr",
                        "corr": corr["corr"],
                        "data": {
                            "x": X.tolist(),
                            "y": y.tolist(),
                            "x_pred": x_pred.tolist(),
                            "intercept": clf.intercept_,
                            "coef": clf.coef_[0],
                            "r2": r2,
                            "y_mean": y_mean.tolist(),
                            "y_std": y_std.tolist(),
                            "x_label": corr["col"],
                            "y_label": target_column,
                        }
                    })
                else:
                    col = corr["col"]
                    X = np.linspace(df[col].min(), df[col].max(), kde_steps)

                    x1 = df[col][df[target_column] == target_value]
                    x1 = x1[x1.notna()]
                    k1 = KernelDensity(bandwidth=kde_bandwidth(x1)).fit(x1.values.reshape((-1, 1)))

                    x2 = df[col][df[target_column] != target_value]
                    x2 = x2[x2.notna()]
                    k2 = KernelDensity(bandwidth=kde_bandwidth(x2)).fit(x2.values.reshape((-1, 1)))

                    charts.append({
                        "type": "kde",
                        "corr": corr["corr"],
                        "data": [
                            {
                                "value": target_value,
                                "y": np.exp(k1.score_samples(X.reshape((-1, 1)))).tolist(),
                                "x": X.tolist(),
                                "group": target_column,
                                "y_label": col,
                            },
                            {
                                "value": other(df[target_column].unique(), target_value),
                                "y": np.exp(k2.score_samples(X.reshape(-1, 1))).tolist(),
                                "x": X.tolist(),
                                "group": target_column,
                                "y_label": col,
                            }
                        ]
                    })

        runtime = time.time() - start

        return {
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "data": {
                "corr": corrs,
                "charts": charts,
            },
            "validations": [{"name": x.name, "level": x.level, "message": x.message} for x in failed_checks],
        }
