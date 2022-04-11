import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import Mock, MagicMock
from actableai.utils.preprocessing import  PercentageTransformer, impute_df

@pytest.fixture(scope="function")
def df():
    return pd.DataFrame({
        "x": [np.nan, 1, 2, 3, 4, 5],
        "y": ["a", "a", "b", np.nan, "b", "b"]
    })


class TestImputDf():

    def test_impute_df_default(self, df):
        df_ = df.copy()
        impute_df(df_)
        assert np.equal(df_["x"], np.asarray([0, 1, 2, 3, 4, 5])).all()
        assert np.equal(df_["y"], np.asarray(["a", "a", "b", "NA", "b", "b"])).all()

    def test_impute_df(self, df):
        df_ = df.copy()
        numeric_imputer = Mock()
        numeric_imputer.fit_transform = MagicMock(return_value=0)

        categorical_imputer = Mock()
        categorical_imputer.fit_transform = MagicMock(return_value=1)

        impute_df(df_, numeric_imputer, categorical_imputer)
        assert_frame_equal(numeric_imputer.fit_transform.call_args[0][0], df[["x"]])
        assert_frame_equal(categorical_imputer.fit_transform.call_args[0][0], df[["y"]])

    def test_no_categorical_cols(self):
        df_ = pd.DataFrame({
            "x": [np.nan, 1, 2, 3, 4, 5],
        })
        impute_df(df_)

    def test_no_numeric_cols(self):
        df_ = pd.DataFrame({
            "y": ["a", "a", "b", np.nan, "b", "b"]
        })
        impute_df(df_)

class TestPercentageTransformer:
    def test_transform(self):
        pt = PercentageTransformer()
        arr = pt.fit_transform(pd.DataFrame({
            'x': ["1.15%", "1.15%", "1.15%", "1.15"]
        }))
        assert arr is not None
        assert arr.isna().sum()[0] == 1
        assert list(arr['x'])[:3] == [1.15, 1.15, 1.15]

    def test_selector(self):
        df = pd.DataFrame({
            'x': ["1.15%", "1.15%", "1.15%", "1.15"],
            'y': ["1.15%", "1.15", "1.15", "1.15"]
        })
        assert list(PercentageTransformer.selector(df)) == [True, False]
