from typing import Union, Tuple, Optional, Dict, Any

from actableai.timeseries.models.params import BaseParams

from mxnet.context import Context

from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import DistributionOutput
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator


class FeedForwardParams(BaseParams):
    """Parameter class for Feed Forward Model."""

    def __init__(
        self,
        hidden_layer_1_size: Union[Tuple[int, int], int] = 40,
        hidden_layer_2_size: Optional[Union[Tuple[int, int], int]] = 40,
        hidden_layer_3_size: Optional[Union[Tuple[int, int], int]] = None,
        epochs: Union[Tuple[int, int], int] = (1, 100),
        learning_rate: Union[Tuple[float, float], float] = (0.001, 0.01),
        context_length: Union[Tuple[int, int], int] = (1, 100),
        l2: Union[Tuple[float, float], float] = (1e-08, 0.01),
        mean_scaling: bool = True,
    ):
        """FeedForwardParams Constructor.

        Args:
            hidden_layer_1_size: Dimension of first layer, if tuple it represents the
                minimum and maximum (excluded) value.
            hidden_layer_2_size: Dimension of second layer, if tuple it represents the
                minimum and maximum (excluded) value.
            hidden_layer_3_size: Dimension of third layer, if tuple it represents the
                minimum and maximum (excluded) value.
            epochs: Number of epochs, if tuple it represents the minimum and maximum
                (excluded) value.
            learning_rate: Learning rate parameter, if tuple it represents the minimum
                and maximum (excluded) value.
            context_length: Number of time units that condition the predictions, if
                tuple it represents the minimum and maximum (excluded) value.
            l2: L2 regularization parameter, if tuple it represents the minimum and
                maximum (excluded) value.
            mean_scaling: Scale the network input by the data mean and the network
                output by its inverse.
        """
        super().__init__(
            model_name="FeedFoward",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.hidden_layer_1_size = hidden_layer_1_size
        self.hidden_layer_2_size = hidden_layer_2_size
        self.hidden_layer_3_size = hidden_layer_3_size
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.mean_scaling = mean_scaling
        self.epochs = epochs
        self.l2 = l2

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "hidden_layer_1_size": self._randint(
                "hidden_layer_1_size", self.hidden_layer_1_size
            ),
            "hidden_layer_2_size": self._randint(
                "hidden_layer_2_size", self.hidden_layer_2_size
            ),
            "hidden_layer_3_size": self._randint(
                "hidden_layer_3_size", self.hidden_layer_3_size
            ),
            "epochs": self._randint("epochs", self.epochs),
            "learning_rate": self._uniform("learning_rate", self.learning_rate),
            "context_length": self._randint("context_length", self.context_length),
            "l2": self._uniform("l2", self.l2),
        }

    def build_estimator(
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        distr_output: DistributionOutput,
        params: Dict[str, Any],
        **kwargs
    ) -> SimpleFeedForwardEstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        hidden_layer_1_size = params.get(
            "hidden_layer_1_size", self.hidden_layer_1_size
        )
        hidden_layer_2_size = params.get(
            "hidden_layer_2_size", self.hidden_layer_2_size
        )
        hidden_layer_3_size = params.get(
            "hidden_layer_3_size", self.hidden_layer_3_size
        )
        hidden_layer_size = []
        if hidden_layer_1_size is not None:
            hidden_layer_size.append(hidden_layer_1_size)
        if hidden_layer_2_size is not None:
            hidden_layer_size.append(hidden_layer_2_size)
        if hidden_layer_3_size is not None:
            hidden_layer_size.append(hidden_layer_3_size)

        return SimpleFeedForwardEstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_hidden_dimensions=hidden_layer_size,
            context_length=params.get("context_length", prediction_length),
            mean_scaling=self.mean_scaling,
            distr_output=distr_output,
            batch_normalization=False,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay=params.get("l2", self.l2),
                hybridize=False,
            ),
        )
