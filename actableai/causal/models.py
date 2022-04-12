import time
from itertools import product

import numpy as np
import pandas as pd
from econml.dml import DML, CausalForestDML, NonParamDML, SparseLinearDML, LinearDML
from econml.drlearner import DRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, XLearner
from econml.iv.nnet import DeepIV
from econml.score.rscorer import RScorer
from hyperopt import hp
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest

from actableai.causal.predictors import SKLearnWrapper, LinearRegressionWrapper
from actableai.causal import autogluon_hyperparameters
from actableai.utils import random_directory
from actableai.utils.sklearn import sklearn_canonical_pipeline
from actableai.causal.params import SparseLinearDMLSingleContTreatmentAGParams
from actableai.regression import PolynomialLinearPredictor


from autogluon.tabular import TabularPredictor


class UntrainedModelException(ValueError):
    pass


class UnsupportedTargetUnitsMultipleTreatments(ValueError):
    pass


class UnsupportedTargetUnitsIVMethods(ValueError):
    pass


class AAICausalEstimator:
    def __init__(
        self,
        model_params=None,
        scorer=None,
        has_categorical_treatment=False,
        has_binary_outcome=False,
    ):
        """Construct an AAICausalEstimator object
        Args: model_params (list, optional): list of BaseCausalEstimatorParams objects. Defaults to None.
        """
        self.model_params = (
            {params.MODEL_NAME: params for params in model_params}
            if model_params is not None
            else {}
        )
        self.scorer = scorer
        self.estimator = None
        self.tune_results_df = None
        self.is_multiple_treatments = False
        self.is_multiple_outcomes = False
        self.has_categorical_treatment = has_categorical_treatment
        self.has_binary_outcome = has_binary_outcome

    def fit(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        label_t="t",
        label_y="y",
        target_units="ate",
        validation_ratio=0.2,
        trials=3,
        tune_params=None,
        max_concurrent=None,
        scheduler=None,
        stopper=None,
        cv="auto",
        feature_importance=False,
        model_directory=None,
        hyperparameters="auto",
        presets="medium_quality_faster_train",
        random_state=None,
        mc_iters="auto",
        remove_outliers=True,
        contamination=0.05,
        num_gpus=0,
    ):
        """Function fits a causal model with a single deterministic model.

        Args:
            Y (np.ndarray): (n × d_y) matrix or vector of length n. Outcomes for each sample
            T (np.ndarray): (n × d_t) matrix or vector of length n. Treatments for each sample
            X (np.ndarray, optional): (n × d_x) matrix. Defaults to None. Features for each sample
            W (np.ndarray, optional): (n × d_w) matrix. Defaults to None. Controls for each sample
            Z (np.ndarray, optional): (n × d_z) matrix. Defaults to None. Instruments for each sample
            target_units (str, optional): either "ate", "att" or "atc". Default to "ate"
            test_size (float): test size in percent. Defaults to 20
            trials (int, optional): number of trials for hyperparameter tuning experiment
            tune_params (dict, optional): dictionary of tune parameters
            max_concurrent (int, optional): max concurcent
            scheduler (object, optional): tune scheduler object
            stopper (object, optional): tune stopper object

        """
        start = time.time()

        try:
            if T.shape[1] > 1:
                self.is_multiple_treatments = True
        except IndexError:
            pass
        try:
            if Y.shape[1] > 1:
                self.is_multiple_outcomes = True
        except IndexError:
            pass

        if hyperparameters == "auto":
            if Y.shape[0] <= 200:
                if mc_iters == "auto":
                    mc_iters = 10
                if cv == "auto":
                    cv = 10
                hyperparameters = {
                    "LR": {},
                    PolynomialLinearPredictor: [{"degree": 2}],
                }
            else:
                hyperparameters = autogluon_hyperparameters()

        if mc_iters == "auto":
            mc_iters = 5
        if cv == "auto":
            cv = 5

        xw_col = []
        if X is not None:
            xw_col += list(X.columns)
        if W is not None:
            xw_col += list(W.columns)

        model_t = TabularPredictor(
            path=random_directory(model_directory),
            label=label_t,
            problem_type="multiclass"
            if self.has_categorical_treatment
            else "regression",
        )
        model_t = SKLearnWrapper(
            model_t,
            xw_col,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )
        model_y = TabularPredictor(
            path=random_directory(model_directory),
            label=label_y,
            problem_type="binary" if self.has_binary_outcome else "regression",
        )
        model_y = SKLearnWrapper(
            model_y,
            xw_col,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit={
                "num_gpus": num_gpus,
            },
        )
        self.estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            cv=cv,
            random_state=random_state,
            mc_iters=mc_iters,
            discrete_treatment=self.has_categorical_treatment,
        )

        # Remove outliers
        if remove_outliers:
            df_ = np.hstack([Y, T])
            if X is not None:
                df_ = np.hstack([df_, X])
            if W is not None:
                df_ = np.hstack([df_, W])
            df_ = pd.DataFrame(df_)
            outlier_clf = sklearn_canonical_pipeline(
                df_, IsolationForest(contamination=contamination)
            )
            is_inline = outlier_clf.fit_predict(df_) == 1
            Y, T = Y[is_inline], T[is_inline]
            if X is not None:
                X = X[is_inline]
            if W is not None:
                W = W[is_inline]

        self.estimator.fit(Y, T, X=X, W=W, cache_values=True)

        if feature_importance and (X is not None or W is not None):
            importances = []
            # Only run feature importance for first mc_iter to speed it up
            for _, m in enumerate(self.estimator.models_t[0]):
                importances.append(m.feature_importance())
            self.model_t_feature_importances = sum(importances) / cv
            self.model_t_feature_importances[
                "stderr"
            ] = self.model_t_feature_importances["stddev"] / np.sqrt(cv)
            self.model_t_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )

            importances = []
            for _, m in enumerate(self.estimator.models_y[0]):
                importances.append(m.feature_importance())
            self.model_y_feature_importances = sum(importances) / cv
            self.model_y_feature_importances[
                "stderr"
            ] = self.model_y_feature_importances["stddev"] / np.sqrt(cv)
            self.model_y_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )

        self.total_trial_time = time.time() - start

    def fit_search(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        target_units="ate",
        validation_ratio=0.2,
        trials=3,
        tune_params=None,
        max_concurrent=None,
        scheduler=None,
        stopper=None,
        cv=10,
    ):
        """Fit casaul model with model and hyper-parameter search.

        Args:
            Y (np.ndarray): (n × d_y) matrix or vector of length n. Outcomes for each sample
            T (np.ndarray): (n × d_t) matrix or vector of length n. Treatments for each sample
            X (np.ndarray, optional): (n × d_x) matrix. Defaults to None. Features for each sample
            W (np.ndarray, optional): (n × d_w) matrix. Defaults to None. Controls for each sample
            Z (np.ndarray, optional): (n × d_z) matrix. Defaults to None. Instruments for each sample
            target_units (str, optional): either "ate", "att" or "atc". Default to "ate"
            test_size (float): test size in percent. Defaults to 20
            trials (int, optional): number of trials for hyperparameter tuning experiment
            tune_params (dict, optional): dictionary of tune parameters
            max_concurrent (int, optional): max concurcent
            scheduler (object, optional): tune scheduler object
            stopper (object, optional): tune stopper object

        """
        # fit different causal estimators with train data
        id_train, id_val = train_test_split(
            np.arange(Y.shape[0]),
            test_size=validation_ratio,
        )
        if X is not None:
            X_train, X_val = X[id_train], X[id_val]
        else:
            X_train, X_val = None, None
        if W is not None:
            W_train, W_val = W[id_train], W[id_val]
        else:
            W_train, W_val = None, None
        if Z is not None:
            Z_train, Z_val = Z[id_train], Z[id_val]
        else:
            Z_train, Z_val = None, None
        Y_train, Y_val = Y[id_train], Y[id_val]
        T_train, T_val = T[id_train], T[id_val]

        try:
            if T.shape[1] > 1:
                self.is_multiple_treatments = True
        except IndexError:
            pass
        try:
            if Y.shape[1] > 1:
                self.is_multiple_outcomes = True
        except IndexError:
            pass

        def causal_estimation_score(params):
            """Function calculates CATE estimation score on validation features X_val

            Args:
                X_val (np.ndarray): (n_val x d_x) matrix. Validation features
                params (BaseCausalEstimatorParams): causal estimator parameter object
            """
            estimator = self.model_params[params["name"]].build_estimator(params=params)
            estimator_type = type(estimator)
            if estimator_type in [DomainAdaptationLearner]:
                estimator.fit(Y_train, T_train, X=X_train)
            elif estimator_type in [DeepIV]:
                estimator.fit(Y_train, T_train, X=X_train, Z=Z_train)
            else:
                estimator.fit(Y_train, T_train, X=X_train, W=W_train)
            try:
                self.scorer.fit(Y_val, T_val, X=X_val, W=W_val)
                if estimator_type not in [DeepIV]:
                    score = self.scorer.score(estimator)
                else:
                    Y_pred = estimator.predict(T_val, X_val)
                    score = np.sqrt(mean_squared_error(Y_pred, Y_val))
            except ValueError:
                # RScorer is not supported when X is None
                score = 0
            return score

        def trainable(params):
            """Function that is a trainable for tune

            Args:
                models (list): List of BaseCausalEstimatorParams objects
            """
            score = causal_estimation_score(params["model"])
            tune.report(score=score)

        # construct the config dict for HyperOptSearch
        models = []
        for name, p in self.model_params.items():
            if p is not None:
                params = p.tune_config()
                params["name"] = p.MODEL_NAME
                models.append(params)
        config = {"model": hp.choice("model", models)}
        algo = HyperOptSearch(config, metric="score", mode="min")
        if max_concurrent is not None:
            algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)
        if tune_params is None:
            tune_params = {}

        analysis = tune.run(
            trainable,
            search_alg=algo,
            num_samples=trials,
            scheduler=scheduler,
            stop=stopper,
            **tune_params
        )
        self.tune_results_df = analysis.results_df

        # get time of each trial
        time_total_s = 0
        for _, result in analysis.results.items():
            if result is not None and "time_total_s" in result:
                time_total_s += result["time_total_s"]

        start = time.time()

        best_config = analysis.get_best_config(metric="score", mode="max")
        temp_estimator = self.model_params[
            best_config["model"]["name"]
        ].build_estimator(params=best_config["model"])

        estimator_type = type(temp_estimator)
        if estimator_type in [DeepIV]:
            self.estimator = temp_estimator
            if target_units != "ate":
                raise UnsupportedTargetUnitsIVMethods()
            self.estimator.fit(Y, T, X=X, Z=Z)
        else:
            # DoWhy wrapper only support single treatment case
            if (
                (not self.is_multiple_treatments)
                and (X is not None)
                and (W is not None)
            ):
                self.estimator = temp_estimator.dowhy
                # refit with entire dataset
                self.estimator.fit(Y, T, X=X, W=W, target_units=target_units)
            else:
                self.estimator = temp_estimator
                if target_units != "ate":
                    raise UnsupportedTargetUnitsMultipleTreatments()
                self.estimator.fit(Y, T, X=X, W=W)

            model_t_R2 = self.estimator.models_t

        self.total_trial_time = time.time() - start + time_total_s

    def effect(self, X=None, T0=None, T1=None, alpha=0.05):
        """Compute heterogeneous treatment effect

        Args:
            X (np.ndarray, optional): (m, d_x) matrix. Features for each sample
            T0 (np.ndarray, optional): (m, d_t) maxtrix or vector of length m. Base treatment for each sample
            T1 (np.ndarray, optional): (m, d_t) maxtrix or vector of length m. Target treatment for each sample
        """
        if self.estimator is None:
            raise UntrainedModelException()

        # DeepIV inference not working, this is an open issue:
        # https://github.com/microsoft/EconML/issues/367
        lb, ub = None, None

        if (~self.is_multiple_treatments) and (~self.is_multiple_outcomes):
            if (T0 is not None) and (T1 is not None):
                effect = self.estimator.effect(X=X, T0=T0, T1=T1)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.effect_interval(
                        X=X, T0=T0, T1=T1, alpha=alpha
                    )
            else:
                effect = self.estimator.effect(X=X)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.effect_interval(X=X, alpha=alpha)
        else:
            if (T0 is not None) and (T1 is not None):
                effect = self.estimator.const_marginal_effect(X=X, T0=T0, T1=T1)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.const_marginal_effect_interval(
                        X=X, T0=T0, T1=T1, alpha=alpha
                    )
            else:
                effect = self.estimator.const_marginal_effect(X=X)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.const_marginal_effect_interval(
                        X=X, alpha=alpha
                    )

        return effect, lb, ub
