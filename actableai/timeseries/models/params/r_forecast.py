from typing import Optional, Union, Tuple, Dict, Any

from actableai.timeseries.models.params import BaseParams

from gluonts.model.r_forecast import RForecastPredictor


class RForecastParams(BaseParams):
    """Parameters class for RForecast Model."""

    def __init__(
        self,
        method_name: Union[Tuple[str, ...], str] = (
            "tbats",
            "thetaf",
            "stlar",
            "arima",
            "ets",
        ),
        period: Optional[Union[Tuple[int, int], int]] = None,
    ):
        """RForecastParams Constructor.

        Args:
            method_name: Name of the method, it tuples it represents the different
                values to choose from.
            period: Period to be used (this is called `frequency` in the R forecast
                package),  if tuple it represents the minimum and maximum (excluded)
                value.
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

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "method_name": self._choice("method_name", self.method_name),
            "period": self._randint("period", self.period),
        }

    def build_predictor(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> RForecastPredictor:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built predictor.
        """
        return RForecastPredictor(
            freq,
            prediction_length,
            params.get("method_name", self.method_name),
            period=params.get("period", self.period),
        )
