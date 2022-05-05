from typing import Union, Iterator, Tuple, Dict, Any, Iterable

import pandas as pd

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor

from actableai.timeseries.utils import handle_features_dataset


class AAITimeSeriesPredictor:
    """Custom Wrapper around GluonTS Predictor."""

    def __init__(
        self,
        predictor: Predictor,
        keep_feat_static_real: bool = True,
        keep_feat_static_cat: bool = True,
        keep_feat_dynamic_real: bool = True,
        keep_feat_dynamic_cat: bool = True,
    ):
        """AAITimeSeriesPredictor Constructor.

        Args:
            predictor: Underlying GluonTS predictor.
            keep_feat_static_real: If False the real static features will be filtered
                out.
            keep_feat_static_cat: If False the categorical static features will be
                filtered out.
            keep_feat_dynamic_real: If False the real dynamic features will be filtered
                out.
            keep_feat_dynamic_cat: If False the categorical dynamic features will be
                filtered out.
        """
        self.predictor = predictor

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def make_evaluation_predictions(
        self, data: Iterable[Dict[str, Any]], num_samples: int
    ) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
        """Wrapper around the GluonTS `make_evaluation_predictions` function.

        Args:
            data: Data used for evaluation.
            num_samples: Number of samples to draw on the model when evaluating.

        Returns:
            - Iterator containing the evaluation forecasts.
            - Iterator containing the original sampled time series.
        """
        data = handle_features_dataset(
            data,
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )

        return make_evaluation_predictions(data, self.predictor, num_samples)

    def predict(self, data: Iterable[Dict[str, Any]], **kwargs) -> Iterator[Forecast]:
        """Run prediction.

        Args:
            data: Data used for prediction.

        Returns:
            Predictions results
        """
        data = handle_features_dataset(
            data,
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )

        return self.predictor.predict(data, **kwargs)
