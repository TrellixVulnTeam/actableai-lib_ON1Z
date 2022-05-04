from typing import Optional, Dict, Tuple, Any, Union, List

from actableai.timeseries.models.params import BaseParams

from gluonts.model.rotbaum import TreeEstimator


class TreePredictorParams(BaseParams):
    """Parameters class for Tree Predictor Model."""

    def __init__(
        self,
        use_feat_dynamic_real: bool,
        use_feat_dynamic_cat: bool,
        model_params: Optional[Dict] = None,
        method: Union[Tuple[str, ...], str] = ("QRX", "QuantileRegression"),
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        context_length: Union[Tuple[int, int], int] = (1, 100),
        max_workers: Optional[int] = None,
        max_n_datapts: int = 1000000,
    ):
        """TreePredictorParams Constructor.

        Args:
            use_feat_dynamic_real: Whether to use the `feat_dynamic_real` field from
                the data.
            use_feat_dynamic_cat: Whether to use the `feat_dynamic_cat` field from
                the data.
            model_params: Parameters which will be passed to the model.
            method: Method to use ["QRX", "QuantileRegression", "QRF"], if tuple it
                represents the different values to choose from.
            quantiles: List of quantiles to predict.
            context_length: Number of time units that condition the predictions, if
                tuple it represents the minimum and maximum (excluded) value.
            max_workers: Maximum number of workers to use, if None no parallelization
                will be done.
            max_n_datapts: Maximium number of data points to use.
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
        # TreePredictor does not handle static features properly (even if it is
        # advertised otherwise)
        self.use_feat_static_real = False
        self.model_params = model_params
        self.method = method
        self.context_length = context_length
        self.quantiles = quantiles
        self.max_workers = max_workers
        self.max_n_datapts = max_n_datapts

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "method": self._choice("method", self.method),
            "context_length": self._randint("context_length", self.context_length),
        }

    def build_estimator(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any], **kwargs
    ) -> TreeEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
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
