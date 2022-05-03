from actableai.timeseries.models.params import BaseParams

from gluonts.model.rotbaum import TreeEstimator


class TreePredictorParams(BaseParams):
    """
    Parameters class for Tree Predictor Model
    """

    def __init__(
        self,
        use_feat_dynamic_real,
        use_feat_dynamic_cat,
        model_params=None,
        method=("QRX", "QuantileRegression"),
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        context_length=(1, 100),
        max_workers=None,
        max_n_datapts=1000000,
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="TreePredictor",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=use_feat_dynamic_real,
            handle_feat_dynamic_cat=use_feat_dynamic_cat,
        )

        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        # TreePredictordoes not handle static features properly (even if it is advertised otherwise)
        self.use_feat_static_real = False
        self.model_params = model_params
        self.method = method
        self.context_length = context_length
        self.quantiles = quantiles
        self.max_workers = max_workers
        self.max_n_datapts = max_n_datapts

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "method": self._choice("method", self.method),
            "context_length": self._randint("context_length", self.context_length),
        }

    def build_estimator(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """
        return TreeEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=params.get("context_length", self.context_length),
            use_feat_dynamic_cat=self.use_feat_dynamic_cat,
            use_feat_dynamic_real=self.use_feat_dynamic_real,
            use_feat_static_real=self.use_feat_static_real,
            model_params=self.model_params,
            method=params.get("method", self.method),
            quantiles=self.quantiles,
            max_workers=self.max_workers,
            max_n_datapts=self.max_n_datapts,
        )