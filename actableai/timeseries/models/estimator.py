from typing import Union, Optional, Iterable, Any, Dict

from actableai.timeseries.utils import handle_features_dataset

from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor


class AAITimeSeriesEstimator:
    """Custom Wrapper around GluonTS Estimator."""

    def __init__(
        self,
        estimator: Estimator,
        keep_feat_static_real: bool = True,
        keep_feat_static_cat: bool = True,
        keep_feat_dynamic_real: bool = True,
        keep_feat_dynamic_cat: bool = True,
    ):
        """AAITimeSeriesEstimator Constructor.

        Args:
            estimator: Underlying GluonTS estimator.
            keep_feat_static_real: If False the real static features will be filtered
                out.
            keep_feat_static_cat: If False the categorical static features will be
                filtered out.
            keep_feat_dynamic_real: If False the real dynamic features will be filtered
                out.
            keep_feat_dynamic_cat: If False the categorical dynamic features will be
                filtered out.
        """
        self.estimator = estimator

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def train(
        self,
        training_data: Iterable[Dict[str, Any]],
        validation_data: Iterable[Dict[str, Any]] = None,
    ) -> Predictor:
        """Train estimator.

        Args:
            training_data: Training data.
            validation_data: Validation data, used for tuning.

        Returns:
            GluonTS trained predictor.
        """
        training_data = handle_features_dataset(
            training_data,
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )

        if validation_data is not None:
            validation_data = handle_features_dataset(
                validation_data,
                self.keep_feat_static_real,
                self.keep_feat_static_cat,
                self.keep_feat_dynamic_real,
                self.keep_feat_dynamic_cat,
            )

        return self.estimator.train(training_data, validation_data)
