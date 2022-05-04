from typing import Union, Tuple, Dict, Any

from actableai.timeseries.models.params import BaseParams

from mxnet.context import Context

from gluonts.mx.trainer import Trainer
from gluonts.model.deepvar import DeepVAREstimator


class DeepVARParams(BaseParams):
    """Parameter class for Deep VAR Model."""

    def __init__(
        self,
        epochs: Union[Tuple[int, int], int] = (1, 100),
        num_cells: Union[Tuple[int, int], int] = 40,
        num_layers: Union[Tuple[int, int], int] = 2,
        cell_type: Union[Tuple[str, ...], str] = "lstm",
        dropout_rate: Union[Tuple[float, float], float] = (0, 0.5),
        rank: Union[Tuple[int, int], int] = 5,
        embedding_dimension: Union[Tuple[int, int], int] = (3, 10),
        context_length: Union[Tuple[int, int], int] = (1, 100),
        learning_rate: Union[Tuple[float, float], float] = (0.0001, 0.01),
        l2: Union[Tuple[float, float], float] = (1e-4, 0.01),
        scaling: bool = False,
    ):
        """DeepVARParams Constructor.

        Args:
            epochs: Number of epochs, if tuple it represents the minimum and maximum
                (excluded) value.
            num_cells: Number of RNN cells for each layer, if tuple it represents the
                minimum and maximum (excluded) value.
            num_layers: Number of RNN layers, if tuple it represents the minimum and
                maximum (excluded) value.
            cell_type: Type of recurrent cells to use ["lstm", "gru"], if tuple it
                represents the different values to choose from.
            dropout_rate: Dropout regularization parameter, if tuple it represents the
                minimum and maximum (excluded) value.
            rank: Rank for the LowrankMultivariateGaussianOutput, if tuple it represents
                the minimum and maximum (excluded) value.
            embedding_dimension: Dimension of the embeddings for categorical features,
                if tuple it represents the minimum and maximum (excluded) value.
            context_length: Number of steps to unroll the RNN for before computing
                predictions, if tuple it represents the minimum and maximum (excluded)
                value.
            learning_rate: Learning rate parameter, if tuple it represents the minimum
                and maximum (excluded) value.
            l2: L2 regularization parameter, if tuple it represents the minimum and
                maximum (excluded) value.
            scaling: Whether to automatically scale the target values.
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

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
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
        self,
        *,
        ctx: Context,
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Dict[str, Any],
        **kwargs
    ) -> DeepVAREstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
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
