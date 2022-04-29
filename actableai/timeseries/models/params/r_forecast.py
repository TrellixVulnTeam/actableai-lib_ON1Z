from actableai.timeseries.models.params import BaseParams

from gluonts.model.r_forecast import RForecastPredictor


class RForecastParams(BaseParams):
    """
    Parameters class for RForecast Model
    """

    def __init__(
        self, method_name=("tbats", "thetaf", "stlar", "arima", "ets"), period=None
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="RForecast",
            is_multivariate_model=False,
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True,
        )

        self.method_name = method_name
        self.period = period

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "method_name": self._choice("method_name", self.method_name),
            "period": self._randint("period", self.period),
        }

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """
        return RForecastPredictor(
            freq,
            prediction_length,
            params.get("method_name", self.method_name),
            period=params.get("period", self.period),
        )
