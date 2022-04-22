from gluonts.evaluation.backtest import make_evaluation_predictions

from actableai.timeseries.util import handle_features_dataset


class AAITimeSeriesPredictor:
    """
    Custom Wrapper around GluonTS Predictor
    """

    def __init__(
        self,
        predictor,
        keep_feat_static_real=True,
        keep_feat_static_cat=True,
        keep_feat_dynamic_real=True,
        keep_feat_dynamic_cat=True,
    ):
        """
        TODO write documentation
        """
        self.predictor = predictor

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def make_evaluation_predictions(self, data, num_samples):
        """
        TODO write documentation
        """
        data = handle_features_dataset(
            data,
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )

        return make_evaluation_predictions(data, self.predictor, num_samples)

    def predict(self, data, **kwargs):
        """
        TODO write documentation
        """
        data = handle_features_dataset(
            data,
            self.keep_feat_static_real,
            self.keep_feat_static_cat,
            self.keep_feat_dynamic_real,
            self.keep_feat_dynamic_cat,
        )

        return self.predictor.predict(data, **kwargs)
