from actableai.timeseries.models.params import BaseParams

from gluonts.mx.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator


class FeedForwardParams(BaseParams):
    """
    Parameter class for Feed Forward Model
    """

    def __init__(
        self,
        hidden_layer_1_size=None,
        hidden_layer_2_size=None,
        hidden_layer_3_size=None,
        epochs=(1, 100),
        learning_rate=(0.001, 0.01),
        context_length=(1, 100),
        l2=(1e-08, 0.01),
        mean_scaling=True,
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="FeedFoward",
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

    def tune_config(self):
        """
        TODO write documentation
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
        self, *, ctx, freq, prediction_length, distr_output, params, **kwargs
    ):
        """
        TODO write documentation
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
