import os
from abc import ABC, abstractmethod

from tensorflow import keras
import numpy as np
from actableai.causal.predictors import SKLearnWrapper
from actableai.utils import random_directory
from autogluon.tabular import TabularPredictor
from econml.dml import CausalForestDML, LinearDML, NonParamDML, SparseLinearDML
from econml.drlearner import DRLearner
from econml.iv.nnet import DeepIV
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner
from econml.score import RScorer
from hyperopt import hp
from ray import tune
from sklearn import linear_model
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)
from sklearn.preprocessing import PolynomialFeatures


class UnsupportedCausalScenario(ValueError):
    pass


class BaseCausalEstimatorParams(ABC):
    @abstractmethod
    def tune_config(self):
        """Returns a configuration dictionary of the model parameters

        Returns:
            dict: Dictionary of parameters. Naming conventions:
        """
        raise NotImplementedError

    @abstractmethod
    def build_estimator(self, params={}):
        """Returns an estimator object given model config dictionary

        Args:
            params (dict, optional): Dictionary of model parameters. Defaults to {}.

        Returns:
            object: A causal estimator that supports .fit() function.
        """
        raise NotImplementedError


class LinearDMLSingleContTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleContTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="RandomForestRegressor",
        random_state=123,
        cv=5,
        mc_iters=None,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.random_state = random_state
        self.cv = cv
        self.mc_iters = mc_iters

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(n_jobs=2),
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            mc_iters=self.mc_iters,
        )


class LinearDMLSingleContTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleContTreatmentAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
        featurizer=None,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory
        self.model_count = 0

        self.random_state = random_state
        self.hyperparameters = hyperparameters
        self.presets = presets
        self.cv = cv
        self.featurizer = featurizer

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            featurizer=self.featurizer,
        )


class SparseLinearDMLSingleContTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleContTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="RandomForestRegressor",
        polyfeat_degree=(1, 4),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.polyfeat_degree = polyfeat_degree
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(n_jobs=2),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            random_state=params.get("random_state", self.random_state),
        )


class SparseLinearDMLSingleContTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleContTreatmentAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=4,
        hyperparameters=None,
        presets="best_quality",
        random_state=123,
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )

        return c

    def build_estimator(self, params={}):
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return SparseLinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            random_state=params.get("random_state", self.random_state),
            cv=self.cv,
        )


class CausalForestDMLSingleContTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "CausalForestDMLSingleContTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="RandomForestRegressor",
        criterion="mse",
        n_estimators=(100, 2000),
        min_impurity_decrease=(0.0005, 0.005),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["criterion"] = (
            hp.choice(f"{self.MODEL_NAME}_criterion", *self.criterion)
            if type(self.criterion) is tuple
            else self.criterion
        )
        c["n_estimators"] = (
            hp.randint(f"{self.MODEL_NAME}_n_estimators", *self.n_estimators)
            if type(self.n_estimators) is tuple
            else self.n_estimators
        )
        c["min_impurity_decrease"] = (
            hp.uniform(
                f"{self.MODEL_NAME}_min_impurity_decrease", *self.min_impurity_decrease
            )
            if type(self.min_impurity_decrease) is tuple
            else self.min_impurity_decrease
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return CausalForestDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(n_jobs=2),
            criterion=params.get("criterion", self.criterion),
            n_estimators=int(
                np.round(params.get("n_estimators", self.n_estimators) / 4) * 4
            ),
            min_impurity_decrease=params.get(
                "min_impurity_decrease", self.min_impurity_decrease
            ),
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLSingleBinaryTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleBinaryTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="RandomForestClassifier",
        min_samples_leaf=(5, 20),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=6,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLSingleBinaryTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleBinaryTreatmentAGParams"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="multiclass",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            discrete_treatment=True,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLSingleBinaryOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleBinaryOutcome"

    def __init__(
        self,
        model_y="RandomForestClassifier",
        model_t="RandomForestRegressor",
        min_samples_leaf=(5, 20),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_y=model_y_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
                model_t=model_t_class(n_jobs=2),
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=6,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLSingleBinaryOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLSingleBinaryOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="binary",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            discrete_treatment=True,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
        )


class SparseLinearDMLSingleBinaryTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleBinaryTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="RandomForestClassifier",
        min_samples_leaf=(5, 20),
        polyfeat_degree=(1, 4),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.min_samples_leaf = min_samples_leaf
        self.polyfeat_degree = polyfeat_degree
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=3,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLSingleBinaryTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleBinaryTreatmentAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=(1, 4),
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="binary",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return SparseLinearDML(
            model_y=model_y,
            model_t=model_t,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLSingleBinaryOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleBinaryOutcome"

    def __init__(
        self,
        model_y="RandomForestClassifier",
        model_t="RandomForestRegressor",
        min_samples_leaf=(5, 20),
        polyfeat_degree=(1, 4),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.min_samples_leaf = min_samples_leaf
        self.polyfeat_degree = polyfeat_degree
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_t=model_t_class(n_jobs=2),
            model_y=model_y_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=3,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLSingleBinaryOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLSingleBinaryOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=(1, 4),
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="binary",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return SparseLinearDML(
            model_t=model_t,
            model_y=model_y,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=True,
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class LinearDMLCategoricalTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="MultiTaskElasticNetCV",
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio", self.l1_ratio)
            if type(self.l1_ratio) is tuple
            else self.l1_ratio
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio", self.l1_ratio),
            ),
            discrete_treatment=False,  # no need if model_t already do one-hot encoding
            linear_first_stages=False,
            cv=6,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLCategoricalTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalTreatmentAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="multiclass",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLCategoricalOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalOutcome"

    def __init__(
        self,
        model_y="MultiTaskElasticNetCV",
        model_t="RandomForestRegressor",
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio", self.l1_ratio)
            if type(self.l1_ratio) is tuple
            else self.l1_ratio
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_t=model_t_class(n_jobs=2),
            model_y=model_y_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio", self.l1_ratio),
            ),
            discrete_treatment=False,  # no need if model_t already do one-hot encoding
            linear_first_stages=False,
            cv=6,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLCategoricalOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="multiclass",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return LinearDML(
            model_t=model_t,
            model_y=model_y,
            discrete_treatment=False,  # no need if model_t already do one-hot encoding
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLCategoricalTreatmentAndOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalTreatmentAndOutcome"

    def __init__(
        self,
        model_y="MultiTaskElasticNetCV",
        model_t="MultiTaskElasticNetCV",
        l1_ratio_y=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        l1_ratio_t=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.l1_ratio_y = l1_ratio_y
        self.l1_ratio_t = l1_ratio_t
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio_y"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio_y", self.l1_ratio_y)
            if type(self.l1_ratio_y) is tuple
            else self.l1_ratio_y
        )
        c["l1_ratio_t"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio_t", self.l1_ratio_t)
            if type(self.l1_ratio_t) is tuple
            else self.l1_ratio_t
        )

        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return LinearDML(
            model_t=model_t_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio_t", self.l1_ratio_t),
            ),
            model_y=model_y_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio_y", self.l1_ratio_y),
            ),
            discrete_treatment=False,  # no need if model_t already do one-hot encoding
            linear_first_stages=False,
            cv=6,
            random_state=params.get("random_state", self.random_state),
        )


class LinearDMLCategoricalTreatmentAndOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "LinearDMLCategoricalTreatmentAndOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        return {}

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="multiclass",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="multiclass",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )
        return LinearDML(
            model_t=model_t,
            model_y=model_y,
            discrete_treatment=False,  # no need if model_t already do one-hot encoding
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
        )


class SparseLinearDMLCategoricalTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalTreatment"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="MultiTaskElasticNetCV",
        polyfeat_degree=(1, 4),
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.polyfeat_degree = polyfeat_degree
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio", self.l1_ratio)
            if type(self.l1_ratio) is tuple
            else self.l1_ratio
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_y=model_y_class(n_jobs=2),
            model_t=model_t_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio", self.l1_ratio),
            ),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=3,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLCategoricalTreatmentAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalTreatmentAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=(1, 4),
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="regression",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="multiclass",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return SparseLinearDML(
            model_y=model_y,
            model_t=model_y,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLCategoricalOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalOutcome"

    def __init__(
        self,
        model_y="RandomForestRegressor",
        model_t="MultiTaskElasticNetCV",
        polyfeat_degree=(1, 4),
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.polyfeat_degree = polyfeat_degree
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio", self.l1_ratio)
            if type(self.l1_ratio) is tuple
            else self.l1_ratio
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_t=model_t_class(n_jobs=2),
            model_y=model_y_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio", self.l1_ratio),
            ),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=3,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLCategoricalOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=(1, 4),
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="multiclass",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="regression",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return SparseLinearDML(
            model_t=model_t,
            model_y=model_y,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLCategoricalTreatmentAndOutcomeParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalTreatmentAndOutcome"

    def __init__(
        self,
        model_y="MultiTaskElasticNetCV",
        model_t="MultiTaskElasticNetCV",
        polyfeat_degree=(1, 4),
        l1_ratio_y=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        l1_ratio_t=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        random_state=123,
    ):
        self.model_y = model_y
        self.model_t = model_t
        self.polyfeat_degree = polyfeat_degree
        self.l1_ratio_y = l1_ratio_y
        self.l1_ratio_t = l1_ratio_t
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_y"] = (
            hp.choice(f"{self.MODEL_NAME}_model_y", *self.model_y)
            if type(self.model_y) is tuple
            else self.model_y
        )
        c["model_t"] = (
            hp.choice(f"{self.MODEL_NAME}_model_t", *self.model_t)
            if type(self.model_t) is tuple
            else self.model_t
        )
        c["l1_ratio_y"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio_y", self.l1_ratio_y)
            if type(self.l1_ratio_y) is tuple
            else self.l1_ratio_y
        )
        c["l1_ratio_t"] = (
            hp.choice(f"{self.MODEL_NAME}_l1_ratio_t", self.l1_ratio_t)
            if type(self.l1_ratio_t) is tuple
            else self.l1_ratio_t
        )
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y_class = globals()[params.get("model_y", self.model_y)]
        model_t_class = globals()[params.get("model_t", self.model_t)]
        return SparseLinearDML(
            model_t=model_t_class(
                n_jobs=2, l1_ratio=params.get("l1_ratio_t", self.l1_ratio_t)
            ),
            model_y=model_y_class(
                n_jobs=2,
                l1_ratio=params.get("l1_ratio_y", self.l1_ratio_y),
            ),
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=3,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class SparseLinearDMLCategoricalTreatmentAndOutcomeAGParams(BaseCausalEstimatorParams):
    MODEL_NAME = "SparseLinearDMLCategoricalTreatmentAndOutcomeAG"

    def __init__(
        self,
        label_t,
        label_y,
        model_directory,
        polyfeat_degree=(1, 4),
        hyperparameters=None,
        random_state=123,
        presets="best_quality",
        cv=10,
    ):
        self.label_t = label_t
        self.label_y = label_y

        self.model_directory = model_directory

        self.polyfeat_degree = polyfeat_degree
        self.hyperparameters = hyperparameters

        self.random_state = random_state
        self.presets = presets
        self.cv = cv

    def tune_config(self):
        c = {}
        c["polyfeat_degree"] = (
            hp.randint(f"{self.MODEL_NAME}_polyfeat_degree", *self.polyfeat_degree)
            if type(self.polyfeat_degree) is tuple
            else self.polyfeat_degree
        )
        return c

    def build_estimator(self, params={}):
        model_y = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_y,
            problem_type="multiclass",
        )
        model_y = SKLearnWrapper(
            model_y, hyperparameters=self.hyperparameters, presets=self.presets
        )
        model_t = TabularPredictor(
            path=random_directory(self.model_directory),
            label=self.label_t,
            problem_type="multiclass",
        )
        model_t = SKLearnWrapper(
            model_t, hyperparameters=self.hyperparameters, presets=self.presets
        )

        return SparseLinearDML(
            model_t=model_t,
            model_y=model_y,
            featurizer=PolynomialFeatures(
                degree=params.get("polyfeat_degree", self.polyfeat_degree)
            ),
            discrete_treatment=False,  # no need if using MultitaskElasticNet
            linear_first_stages=False,
            cv=self.cv,
            random_state=params.get("random_state", self.random_state),
            tol=1e-3,
        )


class DomainAdaptationLearnerSingleBinaryTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "DomainAdaptationLearnerSingleBinaryTreatmentParams"

    def __init__(
        self,
        models="RandomForestRegressor",
        final_models="RandomForestRegressor",
        propensity_model="RandomForestClassifier",
        min_samples_leaf=(5, 20),
        random_state=123,
    ):
        self.models = models
        self.final_models = final_models
        self.propensity_model = propensity_model
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["models"] = (
            hp.choice(f"{self.MODEL_NAME}_models", *self.models)
            if type(self.models) is tuple
            else self.models
        )
        c["final_models"] = (
            hp.choice(f"{self.MODEL_NAME}_final_models", *self.final_models)
            if type(self.final_models) is tuple
            else self.final_models
        )
        c["propensity_model"] = (
            hp.choice(f"{self.MODEL_NAME}_propensity_model", *self.propensity_model)
            if type(self.propensity_model) is tuple
            else self.propensity_model
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )
        return c

    def build_estimator(self, params):
        models_class = globals()[params.get("models", self.models)]
        final_models_class = globals()[params.get("final_models", self.final_models)]
        propensity_model_class = globals()[
            params.get("propensity_model", self.propensity_model)
        ]
        return DomainAdaptationLearner(
            models=models_class(n_jobs=2),
            final_models=final_models_class(n_jobs=2),
            propensity_model=propensity_model_class(n_jobs=2),
        )


class DRLearnerSingleBinaryTreatmentParams(BaseCausalEstimatorParams):
    MODEL_NAME = "DRLearnerSingleBinaryTreatmentParams"

    def __init__(
        self,
        model_propensity="RandomForestClassifier",
        model_regression="RandomForestRegressor",
        model_final="RandomForestRegressor",
        min_samples_leaf=(5, 20),
        random_state=123,
    ):
        self.model_propensity = model_propensity
        self.model_regression = model_regression
        self.model_final = model_final
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["model_propensity"] = (
            hp.choice(f"{self.MODEL_NAME}_model_propensity", *self.model_propensity)
            if type(self.model_propensity) is tuple
            else self.model_propensity
        )
        c["model_regression"] = (
            hp.choice(f"{self.MODEL_NAME}_model_regression", *self.model_regression)
            if type(self.model_regression) is tuple
            else self.model_regression
        )
        c["model_final"] = (
            hp.choice(f"{self.MODEL_NAME}_model_final", *self.model_final)
            if type(self.model_final) is tuple
            else self.model_final
        )
        c["min_samples_leaf"] = (
            hp.randint(f"{self.MODEL_NAME}_min_samples_leaf", *self.min_samples_leaf)
            if type(self.min_samples_leaf) is tuple
            else self.min_samples_leaf
        )
        return c

    def build_estimator(self, params):
        model_propensity_class = globals()[
            params.get("model_propensity", self.model_propensity)
        ]
        model_regression_class = globals()[
            params.get("model_regression", self.model_regression)
        ]
        model_final_class = globals()[params.get("model_final", self.model_final)]
        return DRLearner(
            model_propensity=model_propensity_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            model_regression=model_regression_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            model_final=model_final_class(
                n_jobs=2,
                min_samples_leaf=params.get("min_samples_leaf", self.min_samples_leaf),
            ),
            random_state=params.get("random_state", self.random_state),
        )


class DeepIVParams(BaseCausalEstimatorParams):
    MODEL_NAME = "DeepIVParams"

    def __init__(
        self,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        num_gaussian_mixtures=(2, 12),
        num_keras_fit_epoch=(20, 50),
        random_state=123,
    ):
        self.num_instruments = num_instruments
        self.num_effect_modifiers = num_effect_modifiers
        self.num_treatments = num_treatments
        self.num_gaussian_mixtures = num_gaussian_mixtures
        self.num_keras_fit_epoch = num_keras_fit_epoch
        self.random_state = random_state

    def tune_config(self):
        c = {}
        c["num_gaussian_mixtures"] = (
            hp.randint(
                f"{self.MODEL_NAME}_num_gaussian_mixtures", *self.num_gaussian_mixtures
            )
            if type(self.num_gaussian_mixtures) is tuple
            else self.num_gaussian_mixtures
        )
        c["num_keras_fit_epoch"] = (
            hp.randint(
                f"{self.MODEL_NAME}_num_keras_fit_epoch", *self.num_keras_fit_epoch
            )
            if type(self.num_keras_fit_epoch) is tuple
            else self.num_keras_fit_epoch
        )
        return c

    def build_estimator(self, params={}):
        keras_fit_options = {
            "epochs": params.get("num_keras_fit_epoch", self.num_keras_fit_epoch),
            "validation_split": 0.1,
            "callbacks": [
                keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
            ],
        }

        model_t = keras.Sequential(
            [
                keras.layers.Dense(
                    128,
                    activation="relu",
                    input_shape=(self.num_instruments + self.num_effect_modifiers,),
                ),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.17),
            ]
        )
        model_y = keras.Sequential(
            [
                keras.layers.Dense(
                    128,
                    activation="relu",
                    input_shape=(self.num_treatments + self.num_effect_modifiers,),
                ),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(1),
            ]
        )
        return DeepIV(
            n_components=params.get(
                "num_gaussian_mixtures", self.num_gaussian_mixtures
            ),
            m=lambda z, x: model_t(keras.layers.concatenate([z, x])),
            h=lambda t, x: model_y(keras.layers.concatenate([t, x])),
            n_samples=1,
            use_upper_bound_loss=False,
            n_gradient_samples=1,
            optimizer="adam",
            first_stage_options=keras_fit_options,
            second_stage_options=keras_fit_options,
        )


def get_model_params(
    is_single_treatment,
    has_categorical_treatment,
    is_single_binary_treatment,
    is_single_outcome,
    has_categorical_outcome,
    is_single_binary_outcome,
    has_effect_modifiers_and_common_causes,
    label_t,
    label_y,
    model_directory,
    hyperparameters=None,
    presets="best_quality",
    cv=10,
    featurizer=None,
    mc_iters=None,
):
    model_params = None
    if (
        is_single_treatment
        and (not has_categorical_treatment)
        and is_single_outcome
        and (not has_categorical_outcome)
    ):
        model_params = [
            LinearDMLSingleContTreatmentParams(cv=cv, mc_iters=mc_iters),
            # LinearDMLSingleContTreatmentAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # presets=presets,
            # cv=cv,
            # featurizer=featurizer,
            # ),
            # SparseLinearDMLSingleContTreatmentParams(),
        ]
        # if has_effect_modifiers_and_common_causes:
        # model_params.extend(
        # [
        # LinearDMLSingleContTreatmentAGParams(
        # label_t=label_t,
        # label_y=label_y,
        # model_directory=model_directory,
        # hyperparameters=hyperparameters,
        # ),
        # SparseLinearDMLSingleContTreatmentAGParams(
        # label_t=label_t,
        # label_y=label_y,
        # model_directory=model_directory,
        # hyperparameters=hyperparameters,
        # ),
        # ]
        # )
    elif (
        ((not is_single_treatment) or has_categorical_treatment)
        and is_single_outcome
        and (not has_categorical_outcome)
    ):
        if is_single_binary_treatment:
            model_params = [
                LinearDMLSingleBinaryTreatmentAGParams(
                    label_t=label_t,
                    label_y=label_y,
                    model_directory=model_directory,
                    hyperparameters=hyperparameters,
                    presets=presets,
                    cv=cv,
                ),
                # SparseLinearDMLSingleBinaryTreatmentParams(),
            ]
            # if has_effect_modifiers_and_common_causes:
            # model_params.extend(
            # [
            # LinearDMLSingleBinaryTreatmentAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # SparseLinearDMLSingleBinaryTreatmentAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # ]
            # )
        else:
            model_params = [
                LinearDMLCategoricalTreatmentAGParams(
                    label_t=label_t,
                    label_y=label_y,
                    model_directory=model_directory,
                    hyperparameters=hyperparameters,
                    presets=presets,
                    cv=cv,
                ),
                # SparseLinearDMLCategoricalTreatmentParams(),
            ]
            # if has_effect_modifiers_and_common_causes:
            # model_params.extend(
            # [
            # LinearDMLCategoricalTreatmentAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # SparseLinearDMLCategoricalTreatmentAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # ]
            # )
    elif (
        is_single_treatment
        and (not has_categorical_treatment)
        and ((not is_single_outcome) or has_categorical_outcome)
    ):
        if is_single_binary_outcome:
            model_params = [
                LinearDMLSingleBinaryOutcomeAGParams(
                    label_t=label_t,
                    label_y=label_y,
                    model_directory=model_directory,
                    hyperparameters=hyperparameters,
                    presets=presets,
                    cv=cv,
                ),
                # SparseLinearDMLSingleBinaryOutcomeParams(),
            ]
            # if has_effect_modifiers_and_common_causes:
            # model_params.extend(
            # [
            # LinearDMLSingleBinaryOutcomeAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # SparseLinearDMLSingleBinaryOutcomeAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # ]
            # )
        else:
            model_params = [
                LinearDMLCategoricalOutcomeAGParams(
                    label_t=label_t,
                    label_y=label_y,
                    model_directory=model_directory,
                    hyperparameters=hyperparameters,
                    presets=presets,
                    cv=cv,
                ),
                # SparseLinearDMLCategoricalOutcomeParams(),
            ]
            # if has_effect_modifiers_and_common_causes:
            # model_params.extend(
            # [
            # LinearDMLCategoricalOutcomeAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # SparseLinearDMLCategoricalOutcomeAGParams(
            # label_t=label_t,
            # label_y=label_y,
            # model_directory=model_directory,
            # hyperparameters=hyperparameters,
            # ),
            # ]
            # )
    else:
        model_params = [
            LinearDMLCategoricalTreatmentAndOutcomeAGParams(
                label_t=label_t,
                label_y=label_y,
                model_directory=model_directory,
                hyperparameters=hyperparameters,
                presets=presets,
                cv=cv,
            ),
            # SparseLinearDMLCategoricalTreatmentAndOutcomeParams(),
        ]
        # if has_effect_modifiers_and_common_causes:
        # model_params.extend(
        # [
        # LinearDMLCategoricalTreatmentAndOutcomeAGParams(
        # label_t=label_t,
        # label_y=label_y,
        # model_directory=model_directory,
        # hyperparameters=hyperparameters,
        # ),
        # SparseLinearDMLCategoricalTreatmentAndOutcomeAGParams(
        # label_t=label_t,
        # label_y=label_y,
        # model_directory=model_directory,
        # hyperparameters=hyperparameters,
        # ),
        # ]
        # )

    return model_params


def get_rscorer(
    is_single_treatment,
    has_categorical_treatment,
    is_single_binary_treatment,
    is_single_outcome,
    has_categorical_outcome,
    is_single_binary_outcome,
):
    reg = lambda: RandomForestRegressor(min_samples_leaf=10)
    clf = lambda: RandomForestClassifier(min_samples_leaf=10)
    mten = lambda: MultiTaskElasticNetCV()
    if (
        is_single_treatment
        and (not has_categorical_treatment)
        and is_single_outcome
        and (not has_categorical_outcome)
    ):
        scorer = RScorer(
            model_y=reg(),
            model_t=reg(),
            discrete_treatment=False,
            cv=3,
            mc_iters=3,
            mc_agg="median",
        )
    elif (
        ((not is_single_treatment) or has_categorical_treatment)
        and is_single_outcome
        and (not has_categorical_outcome)
    ):
        if is_single_binary_treatment:
            scorer = RScorer(
                model_y=reg(),
                model_t=clf(),
                discrete_treatment=True,
                cv=3,
                mc_iters=3,
                mc_agg="median",
            )
        else:
            scorer = RScorer(
                model_y=reg(),
                model_t=mten(),
                discrete_treatment=True,
                cv=3,
                mc_iters=3,
                mc_agg="median",
            )
    elif (
        is_single_treatment
        and (not has_categorical_treatment)
        and ((not is_single_outcome) or has_categorical_outcome)
    ):
        if is_single_binary_outcome:
            scorer = RScorer(
                model_y=clf(),
                model_t=reg(),
                discrete_treatment=False,
                cv=3,
                mc_iters=3,
                mc_agg="median",
            )
        else:
            scorer = RScorer(
                model_y=mten(),
                model_t=reg(),
                discrete_treatment=False,
                cv=3,
                mc_iters=3,
                mc_agg="median",
            )
    else:
        scorer = RScorer(
            model_y=mten(),
            model_t=mten(),
            discrete_treatment=False,
            cv=3,
            mc_iters=3,
            mc_agg="median",
        )
    return scorer
