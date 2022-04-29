from actableai.timeseries.models.params import BaseParams

from gluonts.model.trivial.constant import ConstantValuePredictor


class ConstantValueParams(BaseParams):
    """
    Parameters classs for the Constant Value Model
    """

    def __init__(self, value=(0, 100)):
        """
        TODO write documentation
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

    def tune_config(self):
        """
        TODO write documentation
        """
        return {"value": self._uniform("value", self.value)}

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """

        return ConstantValuePredictor(
            value=params.get("value", self.value),
            prediction_length=prediction_length,
            freq=freq,
        )
