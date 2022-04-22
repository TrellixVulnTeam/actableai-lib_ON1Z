from actableai.timeseries.util import handle_features_dataset


class AAITimeSeriesEstimator:
    """
    Custom Wrapper around GluonTS Estimator
    """

    def __init__(
        self,
        estimator,
        keep_feat_static_real=True,
        keep_feat_static_cat=True,
        keep_feat_dynamic_real=True,
        keep_feat_dynamic_cat=True,
    ):
        """
        TODO write documentation
        """
        self.estimator = estimator

        self.keep_feat_static_real = keep_feat_static_real
        self.keep_feat_static_cat = keep_feat_static_cat
        self.keep_feat_dynamic_real = keep_feat_dynamic_real
        self.keep_feat_dynamic_cat = keep_feat_dynamic_cat

    def train(self, training_data, validation_data=None):
        """
        TODO write documentation
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
