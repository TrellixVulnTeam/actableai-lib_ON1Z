from actableai.timeseries.models.params import BaseParams

from pts import Trainer
from pts.model.transformer_tempflow import TransformerTempFlowEstimator


class TransformerTempFlowParams(BaseParams):
    """
    Parameter class for Tramsformer Temp Flow Model
    """

    def __init__(
        self,
        epochs=(1, 100),
        d_model=(4, 8, 12, 16),
        num_heads=(1, 2, 4),
        context_length=(1, 100),
        flow_type="MAF",
        learning_rate=(0.0001, 0.01),
        l2=(0.0001, 0.01),
        scaling=False,
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="TransformerTempFlow",
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.epochs = epochs
        self.d_model = d_model
        self.num_heads = num_heads
        self.context_length = context_length
        self.flow_type = flow_type
        self.l2 = l2
        self.learning_rate = learning_rate
        self.scaling = scaling

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "epochs": self._randint("epochs", self.epochs),
            "d_model": self._choice("d_model", self.d_model),
            "num_heads": self._choice("num_heads", self.num_heads),
            "context_length": self._randint("context_length", self.context_length),
            "flow_type": self._choice("flow_type", self.flow_type),
            "learning_rate": self._uniform("learning_rate", self.learning_rate),
            "l2": self._uniform("l2", self.l2),
        }

    def build_estimator(
        self, *, device, freq, prediction_length, target_dim, params, **kwargs
    ):
        """
        TODO write documentation
        """
        return TransformerTempFlowEstimator(
            device=device,
            d_model=params.get("d_model", self.d_model),
            num_heads=params.get("num_heads", self.num_heads),
            target_dim=target_dim,
            prediction_length=prediction_length,
            context_length=params.get("context_length", self.context_length),
            flow_type=params.get("flow_type", self.flow_type),
            dequantize=True,
            freq=freq,
            pick_incomplete=True,
            scaling=self.scaling,
            trainer=Trainer(
                device=device,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay=params.get("l2", self.l2),
                num_batches_per_epoch=100,
                batch_size=32,
                num_workers=0,
            ),
        )
