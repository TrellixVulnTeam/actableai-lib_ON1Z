import numpy as np
from autogluon.core import space
from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from actableai.third_parties.skgarden.quantile import ensemble


def ag_quantile_hyperparameters(quantile_low=5, quantile_high=95):
    """Returns a dictionnary of Quantile Regressor Model for AutoGluon hyperparameters.

    Args:
        quantile_low: Low bound of the quantile. Default is 5.
        quantile_high High bound of the quantile. Default is 95.

    Returns:
        dictionnary: Models for AutoGluon hyperparameters.
    """
    return {
        ExtraTreesQuantileRegressor: {
            "max_depth": space.Int(3, 32),
        },
        RandomForestQuantileRegressor: {
            "max_depth": space.Int(3, 32),
        },
        CatBoostQuantileRegressor: {
            "quantile_low": quantile_low,
            "quantile_high": quantile_high,
        },
    }


class ExtraTreesQuantileRegressor(AbstractModel):
    """Extra Trees Quantile Regressor AutoGluon Model

    Args:
        AbstractModel: Base class for all AutoGluon models.
    """

    def __init__(self, **kwargs):
        """See https://scikit-garden.github.io/api/#extratreesquantileregressor
        for more information on the parameters.
        """
        super().__init__(**kwargs)
        self.model = ensemble.ExtraTreesQuantileRegressor(**kwargs)

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        self.model.fit(X, y, **kwargs)
        return self.model

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X


class RandomForestQuantileRegressor(AbstractModel):
    """Random Forest Quantile Regressor AutoGluon Model

    Args:
        AbstractModel: Base class for all AutoGluon models.
    """

    def __init__(self, **kwargs):
        """See https://scikit-garden.github.io/api/#skgardenquantilerandomforestquantileregressor
        for more information on the parameters.
        """
        super().__init__(**kwargs)
        self.model = ensemble.RandomForestQuantileRegressor(**kwargs)

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        self.model.fit(X, y, **kwargs)
        return self.model

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X


class CatBoostQuantileRegressor(AbstractModel):
    """CatBoost Quantile Regressor AutoGluon Model

    Args:
        AbstractModel: Base class for all AutoGluon models.
    """

    def __init__(self, **kwargs):
        """See https://catboost.ai/en/docs/concepts/python-reference_catboostregressor
        for more information on the parameters.
        """
        super().__init__(**kwargs)

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(params)

        self.quantile_low = params.pop("quantile_low")
        self.quantile_high = params.pop("quantile_high")

        self.model = CatBoostRegressor(loss_function="Quantile:alpha=0.5", **params)
        self.model.fit(X, y)

        self.model_low = CatBoostRegressor(
            loss_function="Quantile:alpha={:.2f}".format(self.quantile_low / 100.0),
            **params,
        )
        self.model_low.fit(X, y)

        self.model_high = CatBoostRegressor(
            loss_function="Quantile:alpha={:.2f}".format(self.quantile_high / 100.0),
            **params,
        )
        self.model_high.fit(X, y)

    def _predict_proba(self, X, quantile=None, X_train=None, y_train=None, **kwargs):
        X = self.preprocess(X)
        if quantile is None:
            return self.model.predict(X, **kwargs)
        elif quantile == self.quantile_low:
            return self.model_low.predict(X, **kwargs)
        elif quantile == self.quantile_high:
            return self.model_high.predict(X, **kwargs)
        else:
            raise ValueError("Quantile is invalid")

    def _set_default_params(self):
        default_params = {
            "quantile_low": 5,
            "quantile_high": 95,
            "verbose": 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
