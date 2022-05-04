from typing import Callable, Any, Dict, Union, Tuple, Optional
from hyperopt import hp

from mxnet.context import Context
from torch import device as torch_device_type

from gluonts.mx.distribution import DistributionOutput
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor


class BaseParams:
    """Base class for Time Series Model Parameters."""

    def __init__(
        self,
        model_name: str,
        is_multivariate_model: bool,
        has_estimator: bool = True,
        handle_feat_static_real: bool = True,
        handle_feat_static_cat: bool = True,
        handle_feat_dynamic_real: bool = False,
        handle_feat_dynamic_cat: bool = False,
    ):
        """BaseParams Constructor.

        Args:
            model_name: Name of the model.
            is_multivariate_model: Flag to indicate whether the underlying model is a
                multivariate model or not.
            has_estimator: Flag to indicate whether the `build_estimator` function is
                implemented or not.
            handle_feat_static_real: Whether the underlying model is handling static
                real features or not.
            handle_feat_static_cat: Whether the underlying model is handling static cat
                features or not.
            handle_feat_dynamic_real: Whether the underlying model is handling dynamic
                real features or not.
            handle_feat_dynamic_cat: Whether the underlying model is handling dynamic
                cat features or not.
        """
        self.model_name = model_name
        self.is_multivariate_model = is_multivariate_model
        self.has_estimator = has_estimator
        self.handle_feat_static_real = handle_feat_static_real
        self.handle_feat_static_cat = handle_feat_static_cat
        self.handle_feat_dynamic_real = handle_feat_dynamic_real
        self.handle_feat_dynamic_cat = handle_feat_dynamic_cat

    def _hp_param(self, func: Callable, param_name: str, *args, **kwargs) -> Any:
        """Util function used to call hyperopt parameter function.

        Args:
            func: Hyperopt function to call.
            param_name: Name of the parameter.
            args: Arguments to pass to the function.
            kwargs: Named arguments to pass to the function.

        Returns:
            Result of the function.
        """
        return func(f"{self.model_name}_{param_name}", *args, **kwargs)

    def _choice(self, param_name: str, options: Union[Tuple[Any, ...], Any]) -> Any:
        """Util function to represent a parameter selection over a list.

        Args:
            param_name: Name of the parameter.
            options: List to choose from.

        Returns:
            Choose parameter value.
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.choice, param_name, options)

    def _randint(self, param_name: str, options: Union[Tuple[int, int], int]) -> int:
        """Util function to represent a parameter selection of an integer.

        Args:
            param_name: Name of the parameter.
            options: Tuple representing the minimum and maximum (excluded) integer.

        Returns:
            Choose parameter value.
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.randint, param_name, *options)

    def _uniform(
        self, param_name: str, options: Union[Tuple[float, float], float]
    ) -> float:
        """Util function to represent a parameter selection of a floating point.

        Args:
            param_name: Name of the parameter.
            options: Tuple representing the minimum and maximum (excluded) float.

        Returns:
            Choose parameter value.
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.uniform, param_name, *options)

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {}

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        target_dim: int,
        distr_output: DistributionOutput,
        params: Dict[str, Any],
    ) -> Optional[Estimator]:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            device: pytorch device.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.

        Returns:
            Built estimator.
        """
        return None

    def build_predictor(
        self, *, freq: str, prediction_length: int, params: Dict[str, Any]
    ) -> Optional[Predictor]:
        """Build a predictor from the underlying model using selected parameters.

        Args:
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            params: Selected parameters from the hyperparameter space.

        Returns:
            Built predictor.
        """
        return None
