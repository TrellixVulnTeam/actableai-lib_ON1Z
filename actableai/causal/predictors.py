from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class UnsupportedPredictorType(ValueError):
    """Raised when the predictor type is not supported."""
    pass


class UnsupportedProblemType(ValueError):
    """Raised when the problem type is not supported."""
    pass


class DataFrameTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, column_names: Optional[List[str]] = None) -> None:
        """DataFrame Transformer to transform lists and np.ndarray in DataFrames

        Args:
            column_names: Names of the columns for
                new DataFrame, if None the columns be RangeIndex(0, n).
                Number of column names must be the same as the number of columns.
                Defaults to None.
        """
        super().__init__()
        self.column_names = column_names

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X.tolist())
            if self.column_names is not None and len(self.column_names) > 0:
                df.columns = self.column_names
            return df
        if isinstance(X, List) and len(np.array(X).shape) != 2:
            df = pd.DataFrame(X)
            if self.column_names is not None:
                df.columns = self.column_names
            return df
        raise TypeError(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


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
        x_w_columns: Optional[List] = None,
        hyperparameters: Optional[List] = None,
        presets: Optional[str] = "best_quality",
        ag_args_fit: Optional[List] = None
    ):
        """Construct a sklearn wrapper object

        Args:
            ag_predictor: AutoGluon Tabular Predictor
            x_w_columns: Name of common_causes and
                effect modifiers (order matters). Defaults to None.
            hyperparameters: HyperParameter for
                TabularPredictor. Defaults to None.
                presets (Optional[str], optional): Presets for TabularPredictor
                Defaults to "best_quality".
            ag_args_fit: Args fit for Tabular
                Predictor. Defaults to None.

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
        self._df_transformer = DataFrameTransformer(x_w_columns)
        self.ag_args_fit = ag_args_fit
        self.train_data = None

    def fit(self, X, y, sample_weight=None):
        label = self.ag_predictor.label
        train_data = self._df_transformer.fit_transform(X)
        train_data[label] = y
        train_data = TabularDataset(train_data)
        self.ag_predictor.sample_weight = sample_weight

        self.ag_predictor.fit(
            train_data=train_data,
            presets=self.presets,
            hyperparameters=self.hyperparameters,
            ag_args_fit=self.ag_args_fit or {},
        )
        pd.set_option("chained_assignment", "warn")
        self.train_data = train_data
        return self

    def feature_importance(self):
        if self.train_data is None:
            raise Exception("The predictor needs to be fitted before")
        return self.ag_predictor.feature_importance(self.train_data)

    def predict(self, X):
        test_data = self._df_transformer.fit_transform(X)
        test_data = TabularDataset(test_data)
        y_pred = self.ag_predictor.predict(test_data).values
        return y_pred

    def predict_proba(self, X):
        test_data = self._df_transformer.fit_transform(X)
        y_pred_proba = self.ag_predictor.predict_proba(test_data)
        return y_pred_proba.values

    def score(self, X, y, sample_weight=None):
        test_data = self._df_transformer.fit_transform(X)
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
