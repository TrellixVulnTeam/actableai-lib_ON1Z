from actableai.tasks import TaskType
from actableai.tasks.base import AAITask
from re import match


class AAIBayesianRegressionTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.BAYESIAN_REGRESSION)
    def run(
        self,
        df,
        features,
        target,
        priors=None,
        prediction_quantile_low=5,
        prediction_quantile_high=95,
        trials=1,
        polynomial_degree=1,
        validation_split=20,
        pdf_steps=100,
        predict_steps=100,
        normalize=False,
    ):
        """Return a Bayesian ridge regression model that predict df[target] from df[columns].

        Parameters
        ----------
        priors: dict, default=None
            Coefficient priors. If None, all priors are set to 0. If set, key is a tuple containing
            (column name, column value (for categorical columns, otherwise None), polynomial degree).
            If column is numeric, column value should be None. If column is categorical, polynomial degree
            must be 1. Value is a single number indicating the prior mean of that coefficient at
            given polynomial degree.

        """
        import time
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        from ray import tune
        from sklearn.linear_model import BayesianRidge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        from actableai.data_validation.params import BayesianRegressionDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.bayesian_regression.utils import expand_polynomial_categorical
        from actableai.utils.sanitize import sanitize_timezone

        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        data_validation_results = BayesianRegressionDataValidator().validate(
            target, features, df, polynomial_degree
        )
        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        if priors is None:
            priors = {}

        priors_values = {}
        for k in priors:
            if (k["control"] is not None) and (k["degree"] != 1):
                raise ValueError(
                    "Prior for categorical column can only have polynomial degree of 1",
                    k,
                )
            if k["control"] is None:
                if k["degree"] == 1:
                    priors_values["{}".format(k["column"])] = k["value"]
                else:
                    priors_values["{}^{}".format(k["column"], k["degree"])] = k["value"]
            else:
                priors_values[
                    "{}_{}^{}".format(k["column"], k["control"], k["degree"])
                ] = k["value"]

        y = df[target]
        feature_data = df[features]

        df_polynomial, orig_dummy_list = expand_polynomial_categorical(
            feature_data, polynomial_degree, normalize
        )

        X, X_predict, y = (
            df_polynomial[y.notna()],
            df_polynomial[y.isna()],
            y[y.notna()],
        )

        if validation_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split / 100.0
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        def mloss(report=False, features=None):
            if features is None:
                features = X.columns

            def mloss_(config):
                m = BayesianRidge(
                    n_iter=config["n_iter"],
                    alpha_1=config["alpha_1"],
                    alpha_2=config["alpha_2"],
                    lambda_1=config["lambda_1"],
                    lambda_2=config["lambda_2"],
                    alpha_init=config["alpha_init"],
                    lambda_init=config["lambda_init"],
                )

                # When priors are not 0, the target values need to be adjusted as in suggested in
                # https://stats.stackexchange.com/a/44483/41088
                b0 = np.zeros((len(features), 1))
                for i, feat in enumerate(features):
                    b0[i, 0] = priors_values.get(feat, 0)
                y_train_, y_test_ = y_train.copy(), y_test.copy()
                y_train_ -= (X_train[features].values @ b0).flatten()
                y_test_ -= (X_test[features].values @ b0).flatten()

                m.fit(X_train[features], y_train_)
                y_pred, y_std = m.predict(X_test[features], return_std=True)
                r2 = r2_score(y_test_, y_pred)
                rmse = mean_squared_error(y_pred, y_test_, squared=False)
                if report:
                    tune.report(r2=r2)

                # Re-adjust with priors
                m.coef_ += b0.flatten()

                return {
                    "m": m,
                    "r2": r2,
                    "rmse": rmse,
                    "y_pred": y_pred + (X_test[features].values @ b0).flatten(),
                    "y_std": y_std,
                }

            return mloss_

        def tune_model(features=None):
            return tune.run(
                mloss(report=True, features=features),
                metric="r2",
                mode="max",
                num_samples=trials,
                config={
                    "n_iter": tune.randint(10, 300),
                    "alpha_1": tune.uniform(1e-6, 1e-2),
                    "alpha_2": tune.uniform(1e-6, 1e-2),
                    "lambda_1": tune.uniform(1e-6, 1e-2),
                    "lambda_2": tune.uniform(1e-6, 1e-2),
                    "alpha_init": tune.uniform(1e-6, 1),
                    "lambda_init": tune.uniform(1e-6, 1),
                },
            )

        best_config = tune_model().best_config
        re = mloss()(best_config)
        m, r2 = re["m"], re["r2"]

        data = {
            "coeffs": None,
            "intercept": m.intercept_,
            "sigma": m.sigma_,
            "best_config": best_config,
            "evaluation": {
                "r2": r2,
                "rmse": mean_squared_error(y_test, m.predict(X_test), squared=False),
            },
        }

        # Generation prediction for validation set
        exdata = []
        if validation_split > 0:
            exdata = pd.concat(
                [
                    df.loc[X_test.index],
                    pd.DataFrame(
                        {
                            target + "_predicted": re["y_pred"],
                            target + "_std": re["y_std"],
                            target
                            + "_low": [
                                norm.ppf(
                                    prediction_quantile_low / 100.0,
                                    loc=loc,
                                    scale=scale,
                                )
                                for loc, scale in zip(re["y_pred"], re["y_std"])
                            ],
                            target
                            + "_high": [
                                norm.ppf(
                                    prediction_quantile_high / 100.0,
                                    loc=loc,
                                    scale=scale,
                                )
                                for loc, scale in zip(re["y_pred"], re["y_std"])
                            ],
                        }
                    ),
                ],
                axis=1,
            )
            data["validation_table"] = exdata

        # Generate predictions
        if X_predict.shape[0] > 0:
            y_pred, y_std = m.predict(X_predict, return_std=True)
            predictions = pd.concat(
                [
                    df.loc[X_predict.index].drop(columns=[target]),
                    pd.DataFrame(
                        {
                            target: y_pred,
                            target + "_std": y_std,
                            target
                            + "_low": [
                                norm.ppf(
                                    prediction_quantile_low / 100.0,
                                    loc=loc,
                                    scale=scale,
                                )
                                for loc, scale in zip(y_pred, y_std)
                            ],
                            target
                            + "_high": [
                                norm.ppf(
                                    prediction_quantile_high / 100.0,
                                    loc=loc,
                                    scale=scale,
                                )
                                for loc, scale in zip(y_pred, y_std)
                            ],
                        }
                    ),
                ],
                axis=1,
            )
            data["prediction_table"] = predictions

        multivariate_results = []
        univariate_results = []
        for cid, c in enumerate(df_polynomial.columns):
            x = np.linspace(
                m.coef_[cid] - 5 * np.sqrt(m.sigma_[cid, cid]),
                m.coef_[cid] + 5 * np.sqrt(m.sigma_[cid, cid]),
                num=pdf_steps,
            )
            y = norm.pdf(x, loc=m.coef_[cid], scale=np.sqrt(m.sigma_[cid, cid]))
            multivariate_results.append(
                {
                    "name": c,
                    "mean": m.coef_[cid],
                    "stds": np.sqrt(m.sigma_[cid, cid]),
                    "pdfs": [x, y],
                }
            )
            if c in orig_dummy_list:
                # Generate a univariate model
                df_predict = {}
                x = np.linspace(
                    min(df_polynomial[c]), max(df_polynomial[c]), num=predict_steps
                )
                for i in range(polynomial_degree if c in features else 1):
                    col = str(c)
                    col += "^{}".format(i + 1) if i != 0 else ""
                    df_predict[col] = x ** (i + 1)
                df_predict = pd.DataFrame(df_predict)
                best_config = tune_model(features=df_predict.columns).best_config

                r_ = mloss(features=df_predict.columns)(best_config)
                y_mean, y_std = r_["m"].predict(df_predict, return_std=True)
                pdfs = []
                for i in range(polynomial_degree if c in features else 1):
                    x_ = np.linspace(
                        r_["m"].coef_[i] - 5 * np.sqrt(r_["m"].sigma_[i, i]),
                        r_["m"].coef_[i] + 5 * np.sqrt(r_["m"].sigma_[i, i]),
                        num=pdf_steps,
                    )
                    y_ = norm.pdf(
                        x_, loc=r_["m"].coef_[i], scale=np.sqrt(r_["m"].sigma_[i, i])
                    )
                    pdfs.append([x_, y])
                univariate_results.append(
                    {
                        "name": c,
                        "x": x,
                        "coeffs": r_["m"].coef_,
                        "stds": np.sqrt(np.diagonal(r_["m"].sigma_)),
                        "pdfs": pdfs,
                        "y_mean": y_mean,
                        "y_std": y_std,
                        "r2": r_["r2"],
                        "rmse": r_["rmse"],
                    }
                )
        data["coeffs"] = {
            "univariate": univariate_results,
            "multivariate": multivariate_results,
        }
        runtime = time.time() - start
        return {
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "data": data,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
