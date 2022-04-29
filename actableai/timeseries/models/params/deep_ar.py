from actableai.timeseries.models.params import BaseParams

from gluonts.mx.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator


class DeepARParams(BaseParams):
    """
    Parameter class for Deep AR Model
    """

    def __init__(
        self,
        epochs=(1, 100),
        num_cells=(1, 40),
        num_layers=(1, 16),
        dropout_rate=(0, 0.5),
        learning_rate=(0.0001, 0.01),
        batch_size=(16, 128),
        context_length=(1, 100),
        l2=(1e-8, 0.01),
        scaling=True,
        use_feat_dynamic_real=False,
        impute_missing_values=True,
    ):
        """
        TODO write documentation
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
        self.batch_size = batch_size
        self.use_feat_dynamic_real = use_feat_dynamic_real
        # DeepAR does not handle static features properly (even if it is advertised otherwise)
        self.use_feat_static_cat = False
        self.use_feat_static_real = False
        self.cardinality = None
        self.cell_type = "lstm"
        self.scaling = scaling
        self.l2 = l2
        self.impute_missing_value = impute_missing_values

    def tune_config(self):
        """
        TODO write documentation
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
        self, *, ctx, freq, prediction_length, distr_output, params, **kwargs
    ):
        """
        TODO write documentation
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
