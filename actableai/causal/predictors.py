from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from actableai.utils import (
    memory_efficient_hyperparameters,
    fast_categorical_hyperparameters,
)


class UnsupportedPredictorType(ValueError):
    pass


class UnsupportedProblemType(ValueError):
    pass


class DataFrameTransformer(TransformerMixin):
    """DataFrame Transformer to convert List to DataFrame

    Args:
        TransformerMixin (_type_): Base Class Transformer of SKLearn
    """
    def fit_transform(self, X, y=None, x_w_columns=None, **fit_params):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X.tolist())
            if x_w_columns is not None and len(x_w_columns) != 0:
                df.columns = x_w_columns
            return df
        if isinstance(X, list) and len(np.array(X).shape) != 2:
            raise Exception("List must be two dimensional to be converted to DataFrame")
        if isinstance(X, list) or isinstance(X, dict):
            df = pd.DataFrame(X)
            if x_w_columns is not None:
                df.columns = x_w_columns
            return df
        raise TypeError


class LinearRegressionWrapper(LinearRegression):

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred, sample_weight=sample_weight),
            "y": y,
            "y_pred": y_pred,
            "sample_weight": sample_weight
        }

class SKLearnWrapper:
    def __init__(
        self,
        ag_predictor: TabularPredictor,
        x_w_columns:Optional[List]=None,
        hyperparameters:Optional[List]=None,
        presets:Optional[str]="best_quality",
        ag_args_fit:Optional[List]=None
    ):
        """Construct a sklearn wrapper object

        Args:
            ag_predictor (TabularPredictor): AutoGluon Tabular Predictor
            x_w_columns (Optional[List], optional): Name of common_causes and effect modifiers (order matters). Defaults to None.
            hyperparameters (Optional[List], optional): HyperParameter for TabularPredictor. Defaults to None.
            presets (Optional[str], optional): Presets for TabularPredictor. Defaults to "best_quality".
            ag_args_fit (Optional[List], optional): Args fit for Tabular Predictor. Defaults to None.

        Raises:
            UnsupportedPredictorType: Ensure that we only use TabularPredictor
        """
        if type(ag_predictor) is not TabularPredictor:
            raise UnsupportedPredictorType()
        self.ag_predictor = ag_predictor
        if hyperparameters is None:
            self.hyperparameters = "default"
        else:
            self.hyperparameters = hyperparameters
        self.presets = presets
        self._df_transformer = DataFrameTransformer()
        self.ag_args_fit = ag_args_fit
        self.train_data = None
        self.x_w_columns=x_w_columns

    def fit(self, X, y, sample_weight=None):
        label = self.ag_predictor.label
        train_data = self._df_transformer.fit_transform(X, x_w_columns=self.x_w_columns)
        train_data[label] = y
        train_data = TabularDataset(train_data)
        self.ag_predictor.sample_weight = sample_weight

        self.ag_predictor.fit(
            train_data=train_data,
            presets=self.presets,
            hyperparameters=self.hyperparameters,
            ag_args_fit=self.ag_args_fit or {},
        )
        self.train_data = train_data

    def feature_importance(self):
        if self.train_data is None:
            raise Exception("The predictor needs to be fitted before")
        return self.ag_predictor.feature_importance(self.train_data)

    def predict(self, X):
        test_data = self._df_transformer.fit_transform(X, x_w_columns=self.x_w_columns)
        test_data = TabularDataset(test_data)
        y_pred = self.ag_predictor.predict(test_data).values
        return y_pred

    def predict_proba(self, X):
        test_data = self._df_transformer.fit_transform(X, x_w_columns=self.x_w_columns)
        y_pred_proba = self.ag_predictor.predict_proba(test_data)
        return y_pred_proba.values

    def score(self, X, y, sample_weight=None):
        test_data = self._df_transformer.fit_transform(X, x_w_columns=self.x_w_columns)
        y_pred = self.ag_predictor.predict(test_data).values
        if self.ag_predictor.problem_type in ["binary", "multiclass"]:
            return {
                "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight
            }
        elif self.ag_predictor.problem_type == "regression":
            return {
                "r2": r2_score(y, y_pred, sample_weight=sample_weight),
                "y": y,
                "y_pred": y_pred,
                "sample_weight": sample_weight
            }
        else:
            raise UnsupportedProblemType()
