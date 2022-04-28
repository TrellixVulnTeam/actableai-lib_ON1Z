from actableai.timeseries.models.params import BaseParams

from gluonts.model.prophet import ProphetPredictor


class ProphetParams(BaseParams):
    """
    Parameter class for Prophet Model
    """

    def __init__(
        self, growth=("linear"), seasonality_mode=("additive", "multiplicative")
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="Prophet",
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True,
        )

        self.growth = growth
        self.seasonality_mode = seasonality_mode

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "growth": self._choice("growth", self.growth),
            "seasonality_mode": self._choice("seasonality_mode", self.seasonality_mode),
        }

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """
        return ProphetPredictor(
            freq,
            prediction_length=prediction_length,
            prophet_params={
                "growth": params.get("growth", self.growth),
                "seasonality_mode": params.get(
                    "seasonality_mode", self.seasonality_mode
                ),
            },
        )
