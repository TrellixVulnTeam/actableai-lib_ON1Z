from hyperopt import hp

from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.gpvar import GPVAREstimator
from gluonts.model.r_forecast import RForecastPredictor

from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer as PtsTrainer

from actableai.timeseries.feedforward import FeedForwardEstimator


class BaseParams(object):
    def tune_config(self):
        return {}


class RForecastParams(BaseParams):
    MODEL_NAME = "RForecast"

    def __init__(self, method_name=("ets", "arima")):
        self.method_name = method_name

    def tune_config(self):
        c = {}

        c["method_name"] = (
            hp.choice(f"{self.MODEL_NAME}_method_name", self.method_name)
            if type(self.method_name) is tuple
            else self.method_name
        )

        return c

    def build_predictor(self, freq, prediction_length, params):
        return RForecastPredictor(
            freq, prediction_length, params.get("method_name", self.method_name)
        )


class ProphetParams(object):
    MODEL_NAME = "Prophet"

    def __init__(
        self, growth=("linear"), seasonality_mode=("additive", "multiplicative")
    ):
        self.growth = growth
        self.seasonality_mode = seasonality_mode

    def tune_config(self):
        c = {}

        c["growth"] = (
            hp.choice(f"{self.MODEL_NAME}_growth", self.growth)
            if type(self.growth) is tuple
            else self.growth
        )

        c["seasonality_mode"] = (
            hp.choice(f"{self.MODEL_NAME}_seasonality_mode", self.seasonality_mode)
            if type(self.seasonality_mode) is tuple
            else self.seasonality_mode
        )

        return c

    def build_predictor(self, freq, prediction_length, params):
        return ProphetPredictor(
            freq,
            prediction_length=prediction_length,
            prophet_params={
                "growth": params.get("growth", self.growth),
                "seasonality_mode": params.get(
                    "seasonality_mode", self.seasonality_mode
                ),
            },
        )


class FeedForwardParams(object):
    MODEL_NAME = "FeedForward"

    def __init__(
        self,
        hidden_layer_size=(2, 40),
        epochs=(1, 100),
        learning_rate=(0.001, 0.01),
        context_length=(1, 100),
        l2=(1e-08, 0.01),
        mean_scaling=False,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.mean_scaling = mean_scaling
        self.epochs = epochs
        self.l2 = l2

    def tune_config(self):
        c = {}

        c["hidden_layer_size"] = (
            hp.randint(f"{self.MODEL_NAME}_hidden_layer_size", *self.hidden_layer_size)
            if type(self.hidden_layer_size) is tuple
            else self.hidden_layer_size
        )

        c["epochs"] = (
            hp.randint(f"{self.MODEL_NAME}_epochs", *self.epochs)
            if type(self.epochs) is tuple
            else self.epochs
        )

        c["learning_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_learning_rate", *self.learning_rate)
            if type(self.learning_rate) is tuple
            else self.learning_rate
        )

        c["context_length"] = (
            hp.randint(f"{self.MODEL_NAME}_context_length", *self.context_length)
            if type(self.context_length) is tuple
            else self.context_length
        )

        c["l2"] = (
            hp.uniform(f"{self.MODEL_NAME}_l2", *self.l2)
            if type(self.l2) is tuple
            else self.l2
        )

        return c

    def build(self, ctx, freq, prediction_length, distr_output, params):
        return FeedForwardEstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_hidden_dimensions=[
                params.get("hidden_layer_size", self.hidden_layer_size)
            ],
            context_length=params.get("context_length", prediction_length),
            mean_scaling=self.mean_scaling,
            distr_output=distr_output,
            batch_normalization=True,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                post_epoch_callback=params.get("post_epoch_callback"),
                weight_decay=params.get("l2", self.l2),
                hybridize=False,
            ),
        )


class DeepARParams(object):
    MODEL_NAME = "DeepAR"

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
        scaling=False,
    ):
        self.context_length = context_length
        self.epochs = epochs
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_feat_dynamic_real = False
        self.use_feat_static_cat = False
        self.cell_type = "lstm"
        self.scaling = scaling
        self.l2 = l2

    def tune_config(self):
        s = {}
        s["context_length"] = (
            hp.randint(f"{self.MODEL_NAME}_context_length", *self.context_length)
            if type(self.context_length) is tuple
            else self.context_length
        )

        s["epochs"] = (
            hp.randint(f"{self.MODEL_NAME}_epochs", *self.epochs)
            if type(self.epochs) is tuple
            else self.epochs
        )

        s["num_cells"] = (
            hp.randint(f"{self.MODEL_NAME}_num_cells", *self.num_cells)
            if type(self.num_cells) is tuple
            else self.num_cells
        )

        s["num_layers"] = (
            hp.randint(f"{self.MODEL_NAME}_num_layers", *self.num_layers)
            if type(self.num_layers) is tuple
            else self.num_layers
        )

        s["dropout_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_dropout_rate", *self.dropout_rate)
            if type(self.dropout_rate) is tuple
            else self.dropout_rate
        )

        s["learning_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_learning_rate", *self.learning_rate)
            if type(self.learning_rate) is tuple
            else self.learning_rate
        )

        s["l2"] = (
            hp.uniform(f"{self.MODEL_NAME}_l2", *self.l2)
            if type(self.l2) is tuple
            else self.l2
        )

        return s

    def build(self, ctx, freq, prediction_length, distr_output, params):
        return DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            cell_type=self.cell_type,
            use_feat_dynamic_real=self.use_feat_dynamic_real,
            use_feat_static_cat=self.use_feat_dynamic_real,
            scaling=self.scaling,
            dropout_rate=params.get("dropout_rate", self.dropout_rate),
            num_layers=params.get("num_layers", self.num_layers),
            num_cells=params.get("num_cells", self.num_cells),
            context_length=params.get("context_length", prediction_length),
            distr_output=distr_output,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                post_epoch_callback=params.get("post_epoch_callback"),
                hybridize=False,
                weight_decay=params.get("l2", self.l2),
            ),
        )


class TransformerTempFlowParams(object):
    MODEL_NAME = "TransformerTempFlow"

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
        self.epochs = epochs
        self.d_model = d_model
        self.num_heads = num_heads
        self.context_length = context_length
        self.flow_type = flow_type
        self.l2 = l2
        self.learning_rate = learning_rate
        self.scaling = scaling

    def tune_config(self):
        c = {}
        c["epochs"] = (
            hp.randint(f"{self.MODEL_NAME}_epochs", *self.epochs)
            if type(self.epochs) is tuple
            else self.epochs
        )

        c["d_model"] = (
            hp.choice(f"{self.MODEL_NAME}_d_model", self.d_model)
            if type(self.d_model) is tuple
            else self.d_model
        )

        c["num_heads"] = (
            hp.choice(f"{self.MODEL_NAME}_num_heads", self.num_heads)
            if type(self.num_heads) is tuple
            else self.num_heads
        )

        c["context_length"] = (
            hp.randint(f"{self.MODEL_NAME}_context_length", *self.context_length)
            if type(self.context_length) is tuple
            else self.context_length
        )

        c["flow_type"] = (
            hp.choice(f"{self.MODEL_NAME}_flow_type", self.flow_type)
            if type(self.flow_type) is tuple
            else self.flow_type
        )

        c["learning_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_learning_rate", *self.learning_rate)
            if type(self.learning_rate) is tuple
            else self.learning_rate
        )

        c["l2"] = (
            hp.uniform(f"{self.MODEL_NAME}_l2", *self.l2)
            if type(self.l2) is tuple
            else self.l2
        )

        return c

    def build(self, device, freq, prediction_length, target_dim, params):
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
            trainer=PtsTrainer(
                device=device,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay=params.get("l2", self.l2),
                num_batches_per_epoch=100,
                batch_size=32,
                num_workers=0,
            ),
        )


class DeepVARParams(object):
    MODEL_NAME = "DeepVAR"

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
        c = {}
        c["context_length"] = (
            hp.randint(f"{self.MODEL_NAME}_context_length", *self.context_length)
            if type(self.context_length) is tuple
            else self.context_length
        )

        c["epochs"] = (
            hp.randint(f"{self.MODEL_NAME}_epochs", *self.epochs)
            if type(self.epochs) is tuple
            else self.epochs
        )

        c["num_layers"] = (
            hp.randint(f"{self.MODEL_NAME}_num_layers", *self.num_layers)
            if type(self.num_layers) is tuple
            else self.num_layers
        )

        c["num_cells"] = (
            hp.randint(f"{self.MODEL_NAME}_num_cells", *self.num_cells)
            if type(self.num_cells) is tuple
            else self.num_cells
        )

        c["dropout_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_dropout_rate", *self.dropout_rate)
            if type(self.dropout_rate) is tuple
            else self.dropout_rate
        )

        c["learning_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_learning_rate", *self.learning_rate)
            if type(self.learning_rate) is tuple
            else self.learning_rate
        )

        c["rank"] = (
            hp.randint(f"{self.MODEL_NAME}_rank", *self.rank)
            if type(self.rank) is tuple
            else self.rank
        )

        c["cell_type"] = (
            hp.choice(f"{self.MODEL_NAME}_cell_type", self.cell_type)
            if type(self.cell_type) is tuple
            else self.cell_type
        )

        c["l2"] = (
            hp.uniform(f"{self.MODEL_NAME}_l2", *self.l2)
            if type(self.l2) is tuple
            else self.l2
        )

        return c

    def build(self, ctx, freq, prediction_length, target_dim, params):
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
                post_epoch_callback=params.get("post_epoch_callback"),
                weight_decay=params.get("l2", self.l2),
                hybridize=True,
            ),
        )


class GPVarParams(object):
    MODEL_NAME = "GPVar"

    def __init__(
        self,
        epochs=(1, 100),
        num_layers=(1, 32),
        num_cells=(1, 100),
        cell_type=("lstm", "gru"),
        dropout_rate=(0, 0.5),
        rank=(1, 20),
        context_length=(1, 100),
        learning_rate=(0.0001, 0.01),
        l2=(1e-4, 0.01),
    ):
        self.epochs = epochs
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.rank = rank
        self.context_length = context_length
        self.l2 = l2

    def tune_config(self):
        s = {}
        s["context_length"] = (
            hp.randint(f"{self.MODEL_NAME}_context_length", *self.context_length)
            if type(self.context_length) is tuple
            else self.context_length
        )

        s["epochs"] = (
            hp.randint(f"{self.MODEL_NAME}_epochs", *self.epochs)
            if type(self.epochs) is tuple
            else self.epochs
        )

        s["num_cells"] = (
            hp.randint(f"{self.MODEL_NAME}_num_cells", *self.num_cells)
            if type(self.num_cells) is tuple
            else self.num_cells
        )

        s["num_layers"] = (
            hp.randint(f"{self.MODEL_NAME}_num_layers", *self.num_layers)
            if type(self.num_layers) is tuple
            else self.num_layers
        )

        s["dropout_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_dropout_rate", *self.dropout_rate)
            if type(self.dropout_rate) is tuple
            else self.dropout_rate
        )

        s["learning_rate"] = (
            hp.uniform(f"{self.MODEL_NAME}_learning_rate", *self.learning_rate)
            if type(self.learning_rate) is tuple
            else self.learning_rate
        )

        s["rank"] = (
            hp.randint(f"{self.MODEL_NAME}_rank", *self.rank)
            if type(self.rank) is tuple
            else self.rank
        )

        s["cell_type"] = (
            hp.choice(f"{self.MODEL_NAME}_cell_type", self.cell_type)
            if type(self.cell_type) is tuple
            else self.cell_type
        )

        s["l2"] = (
            hp.uniform(f"{self.MODEL_NAME}_l2", *self.l2)
            if type(self.l2) is tuple
            else self.l2
        )

        return s

    def build(self, ctx, freq, prediction_length, target_dim, params):
        return GPVAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            cell_type=params.get("cell_type", self.cell_type),
            dropout_rate=params.get("dropout_rate", self.dropout_rate),
            num_layers=params.get("num_layers", self.num_layers),
            num_cells=params.get("num_cells", self.num_cells),
            context_length=params.get("context_length", prediction_length),
            num_parallel_samples=100,
            rank=params.get("rank", self.rank),
            scaling=False,
            pick_incomplete=True,
            shuffle_target_dim=True,
            conditioning_length=100,
            use_marginal_transformation=False,
            target_dim=target_dim,
            trainer=Trainer(
                ctx=ctx,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                post_epoch_callback=params.get("post_epoch_callback"),
                weight_decay=params.get("l2", self.l2),
                hybridize=False,
            ),
        )
