import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression

class OneHotEncodingTransformer():

    def __init__(self, df):
        num_cols = df._get_numeric_data().columns
        self._num_col_ids = []
        for i, c in enumerate(df.columns):
            if c in num_cols:
                self._num_col_ids.append(i)

        self._cat_col_ids = list(set(range(df.shape[1])) - set(self._num_col_ids))
        if len(self._cat_col_ids) > 0:
            self._transformer = OneHotEncoder(sparse=False)
            self._transformer.fit(df.iloc[:, self._cat_col_ids])

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        if len(self._cat_col_ids) == 0:
            return X
        return np.hstack([X[:, self._num_col_ids], self._transformer.transform(X[:, self._cat_col_ids])])


class PolynomialLinearPredictor(AbstractModel):

    def _preprocess(self, X:pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            degree = self._get_model_params().get("degree", 2)
            self.poly_scaler = make_pipeline(
                SimpleImputer(strategy="mean"),
                StandardScaler(),
                PolynomialFeatures(degree),
            )
            self.poly_scaler.fit(X)

        return self.poly_scaler.transform(X)

    def _fit(self, X:pd.DataFrame, y:pd.Series, **kwargs):
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        del params["degree"]

        if self.problem_type in ["regression", "softclass"]:
            self.model = Ridge(**params)
        else:
            self.model = LogisticRegression(solver="liblinear", **params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            "degree": 2,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

