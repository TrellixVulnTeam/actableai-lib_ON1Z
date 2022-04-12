from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype, is_object_dtype

from actableai.regression import PolynomialLinearPredictor


def autogluon_hyperparameters():
    return {
        "LR": {},
        "RF": {},
        # "CAT": {}
        "XGB": {},
        "KNN": {},
        "XT": {},
        # PolynomialLinearPredictor: [
        # {"degree": 2},
        # {"degree": 3},
        # {"degree": 4},
        # ],
    }


class OneHotEncodingTransformer:
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
        return np.hstack(
            [
                X[:, self._num_col_ids],
                self._transformer.transform(X[:, self._cat_col_ids]),
            ]
        )


def has_categorical_column(df, columns):
    return len(df[columns].select_dtypes(include=np.number).columns) < len(
        df[columns].columns
    )


def prepare_sanitize_data(
    pd_table: pd.DataFrame,
    treatments: List[str],
    outcomes: List[str],
    effect_modifiers: List[str],
    common_causes: List[str],
) -> pd.DataFrame:
    """Drop NAN rows for treatments and outcomes then impute effect_modifiers and common_causes

    Args:
        pd_table (pd.DataFrame): Table to prepare
        treatments (List[str]): List of treatments
        outcomes (List[str]): List of outcomes
        effect_modifiers (List[str]): List of effect modifiers
        common_causes (List[str]): List of common causes

    Returns:
        pd.DataFrame: New table sanitized and imputed
    """
    # drop rows with NaN treatments or outcomes
    pd_table = pd_table[~pd_table[treatments + outcomes].isnull().any(axis=1)]

    # Impute with Mean numeric
    for c in pd_table:
        if c in effect_modifiers + common_causes and is_numeric_dtype(pd_table[c]):
            pd_table[c] = SimpleImputer(strategy="mean").fit_transform(pd_table[[c]])

    # String nan for categorical
    for c in pd_table:
        if c in effect_modifiers + common_causes and is_object_dtype(pd_table[c]):
            pd_table[c] = np.where(pd_table[c].isna(), "nan", pd_table[c])

    return pd_table
