from actableai.timeseries.models.params import BaseParams

from gluonts.mx.trainer import Trainer
from gluonts.model.deepvar import DeepVAREstimator


class DeepVARParams(BaseParams):
    """
    Parameter class for Deep VAR Model
    """

    def __init__(
        self,
        epochs=(1, 100),
        num_layers=2,
        num_cells=40,
        cell_type=("lstm"),
        dropout_rate=(0, 0.5),
        rank=5,
        embedding_dimension=(3, 10),
        context_length=(1, 100),
        learning_rate=(0.0001, 0.01),
        l2=(1e-4, 0.01),
        scaling=False,
    ):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="DeepVAR",
            is_multivariate_model=True,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False,
        )

        self.epochs = epochs
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.rank = rank
        self.embedding_dimension = embedding_dimension
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.scaling = scaling
        self.l2 = l2

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "context_length": self._randint("context_length", self.context_length),
            "epochs": self._randint("epochs", self.epochs),
            "num_layers": self._randint("num_layers", self.num_layers),
            "num_cells": self._randint("num_cells", self.num_cells),
            "dropout_rate": self._uniform("dropout_rate", self.dropout_rate),
            "learning_rate": self._uniform("learning_raet", self.learning_rate),
            "rank": self._randint("rank", self.rank),
            "cell_type": self._choice("cell_type", self.cell_type),
            "l2": self._uniform("l2", self.l2),
        }

    def build_estimator(
        self, *, ctx, freq, prediction_length, target_dim, params, **kwargs
    ):
        """
        TODO write documentation
        """
        return DeepVAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            cell_type=params.get("cell_type", self.cell_type),
            dropout_rate=params.get("dropout_rate", self.dropout_rate),
            num_layers=params.get("num_layers", self.num_layers),
            num_cells=params.get("num_cells", self.num_cells),
            context_length=params.get("context_length", prediction_length),
            num_parallel_samples=100,
            rank=params.get("rank", self.rank),
            scaling=self.scaling,
            pick_incomplete=True,
            conditioning_length=100,
            use_marginal_transformation=False,
            target_dim=target_dim,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay=params.get("l2", self.l2),
                hybridize=True,
            ),
        )
