from typing import Tuple, Union, Dict, Any

from actableai.timeseries.models.params import BaseParams

from mxnet.context import Context

from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import DistributionOutput
from gluonts.model.deepar import DeepAREstimator


class DeepARParams(BaseParams):
    """Parameter class for Deep AR Model."""

    def __init__(
        self,
        epochs: Union[Tuple[int, int], int] = (1, 100),
        num_cells: Union[Tuple[int, int], int] = (1, 40),
        num_layers: Union[Tuple[int, int], int] = (1, 16),
        dropout_rate: Union[Tuple[float, float], float] = (0, 0.5),
        learning_rate: Union[Tuple[float, float], float] = (0.0001, 0.01),
        context_length: Union[Tuple[int, int], int] = (1, 100),
        l2: Union[Tuple[float, float], float] = (1e-8, 0.01),
        scaling: bool = True,
        use_feat_dynamic_real: bool = False,
        impute_missing_values: bool = True,
    ):
        """DeepARParams Constructor.

        Args:
            epochs: Number of epochs, if tuple it represents the minimum and maximum
                (excluded) value.
            num_cells: Number of RNN cells for each layer, if tuple it represents the
                minimum and maximum (excluded) value.
            num_layers: Number of RNN layers, if tuple it represents the minimum and
                maximum (excluded) value.
            dropout_rate: Dropout regularization parameter, if tuple it represents the
                minimum and maximum (excluded) value.
            learning_rate: Learning rate parameter, if tuple it represents the minimum
                and maximum (excluded) value.
            context_length: Number of steps to unroll the RNN for before computing
                predictions, if tuple it represents the minimum and maximum (excluded)
                value.
            l2: L2 regularization parameter, if tuple it represents the minimum and
                maximum (excluded) value.
            scaling: Whether to automatically scale the target values.
            use_feat_dynamic_real: Whether to use the `feat_dynamic_real` field from
                the data.
            impute_missing_values: Whether to impute the missing values during training
                by using the current model parameters. Recommended if the dataset
                contains many missing values. However, this is a lot slower than the
                default mode.
        """
        super().__init__(
            model_name="DeepAR",
            is_multivariate_model=False,
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=use_feat_dynamic_real,
            handle_feat_dynamic_cat=False,
        )

        self.context_length = context_length
        self.epochs = epochs
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        # DeepAR does not handle static features properly (even if it is advertised otherwise)
        self.use_feat_static_cat = False
        self.use_feat_static_real = False
        self.cardinality = None
        self.cell_type = "lstm"
        self.scaling = scaling
        self.l2 = l2
        self.impute_missing_value = impute_missing_values

    def tune_config(self) -> Dict[str, Any]:
        """Select parameters in the pre-defined hyperparameter space.

        Returns:
            Selected parameters.
        """
        return {
            "context_length": self._randint("context_length", self.context_length),
            "epochs": self._randint("epochs", self.epochs),
            "num_cells": self._randint("num_cells", self.num_cells),
            "num_layers": self._randint("num_layers", self.num_layers),
            "dropout_rate": self._uniform("dropout_rate", self.dropout_rate),
            "learning_rate": self._uniform("learning_rate", self.learning_rate),
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
    ) -> DeepAREstimator:
        """Build an estimator from the underlying model using selected parameters.

        Args:
            ctx: mxnet context.
            freq: Frequency of the time series used.
            prediction_length: Length of the prediction that will be forecasted
            distr_output: Distribution output to use.
            params: Selected parameters from the hyperparameter space.
            kwargs: Ignored arguments.

        Returns:
            Built estimator.
        """
        return DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            cell_type=self.cell_type,
            use_feat_dynamic_real=self.use_feat_dynamic_real,
            use_feat_static_cat=self.use_feat_static_cat,
            use_feat_static_real=self.use_feat_static_real,
            cardinality=self.cardinality,
            scaling=self.scaling,
            dropout_rate=params.get("dropout_rate", self.dropout_rate),
            num_layers=params.get("num_layers", self.num_layers),
            num_cells=params.get("num_cells", self.num_cells),
            context_length=params.get("context_length", prediction_length),
            distr_output=distr_output,
            impute_missing_values=self.impute_missing_value,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                hybridize=False,
                weight_decay=params.get("l2", self.l2),
            ),
        )
