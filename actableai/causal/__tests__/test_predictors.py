import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import pytest

from actableai.causal.predictors import DataFrameTransformer, SKLearnWrapper

class TestDataFrameTransformer:
    def test_fit_transform(self):
        dft = DataFrameTransformer()
        df = dft.fit_transform(
            np.array([[0, 1, 2]])
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)

    def test_fit_transform_col_names(self):
        dft = DataFrameTransformer()
        df = dft.fit_transform(
            np.array([[0, 1, 2]]), x_w_columns=['a', 'b', 'c']
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)
        assert list(df.columns) == ['a', 'b', 'c']

class TestSKLearnWrapper:
    def test_init(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b'], None, "medium_quality_faster_train", None)
        assert sklw.ag_args_fit is None
        assert sklw.ag_predictor is not None
        assert sklw.presets == "medium_quality_faster_train"
        assert TabularPredictor is not None

    def test_fit(self):
        width, height = 10, 20
        sklw = SKLearnWrapper(TabularPredictor('y'), [str(i) for i in range(width)], None, "medium_quality_faster_train", None)
        X = np.random.randint(0, 100, size=(height, width))
        y = np.array(np.random.randint(0, 100, size=(height, 1)))
        sklw.fit(X, y)
        assert sklw.ag_predictor.sample_weight is None
        assert sklw.train_data is not None

    def test_feature_importance(self):
        width, height = 10, 20
        sklw = SKLearnWrapper(TabularPredictor('y'), [str(i) for i in range(width)], None, "medium_quality_faster_train", None)
        X = np.random.randint(0, 100, size=(height, width))
        y = np.array(np.random.randint(0, 100, size=(height, 1)))
        sklw.fit(X, y)
        feat_imp = sklw.feature_importance()
        assert sklw.ag_predictor.sample_weight is None
        assert sklw.train_data is not None
        assert set(feat_imp.index) == set([str(i) for i in range(width)])

    def test_feature_importance_no_fit(self):
        sklw = SKLearnWrapper(TabularPredictor('y'), ['a', 'b', 'c'], None, "medium_quality_faster_train", None)
        with pytest.raises(Exception):
            sklw.feature_importance()
