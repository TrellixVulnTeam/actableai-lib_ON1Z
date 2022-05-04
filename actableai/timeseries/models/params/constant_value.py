from typing import Tuple, Union, Dict, Any

from actableai.timeseries.models.params import BaseParams

from gluonts.model.trivial.constant import ConstantValuePredictor


class ConstantValueParams(BaseParams):
    """Parameters class for the Constant Value Model."""

    def __init__(self, value: Union[Tuple[int, int], int] = (0, 100)):
        """ConstantValueParams Constructor.

        Args:
            value: Value to return, if tuple it represents minimum and maximum
                (excluded) value.
        """

        super().__init__(
            model_name="ConstantValue",
            is_multivariate_model=False,
            has_estimator=False,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.value = value

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {"value": self._uniform("value", self.value)}

    def build_predictor(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> ConstantValuePredictor:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built predictor.
        """

        return ConstantValuePredictor(
            value=params.get("value", self.value),
            prediction_length=prediction_length,
            freq=freq,
        )
