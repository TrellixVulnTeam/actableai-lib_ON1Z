from hyperopt import hp

from gluonts.mx.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.gpvar import GPVAREstimator
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.rotbaum import TreeEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.trivial.constant import ConstantValuePredictor

from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer as PtsTrainer


class BaseParams:
    """
    Base class for Time Series Model Parameters
    """

    def __init__(self,
                 model_name,
                 has_estimator=True,
                 handle_feat_static_real=True,
                 handle_feat_static_cat=True,
                 handle_feat_dynamic_real=False,
                 handle_feat_dynamic_cat=False):
        """
        TODO write documentation
        """
        self.model_name = model_name
        self.has_estimator = has_estimator
        self.handle_feat_static_real = handle_feat_static_real
        self.handle_feat_static_cat = handle_feat_static_cat
        self.handle_feat_dynamic_real = handle_feat_dynamic_real
        self.handle_feat_dynamic_cat = handle_feat_dynamic_cat

    def _hp_param(self, func, param_name, *args, **kwargs):
        """
        TODO write documentation
        """
        return func(f"{self.model_name}_{param_name}", *args, **kwargs)

    def _choice(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.choice, param_name, options)

    def _randint(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.randint, param_name, *options)

    def _uniform(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.uniform, param_name, *options)


    def tune_config(self):
        """
        TODO write documentation
        """
        return {}

    def build_estimator(self,
                        *,
                        ctx,
                        device,
                        freq,
                        prediction_length,
                        target_dim,
                        distr_output,
                        params):
        """
        TODO write documentation
        """
        return None

    def build_predictor(self, *, freq, prediction_length, params):
        """
        TODO write documentation
        """
        return None


class RForecastParams(BaseParams):
    """
    Parameters class for RForecast Model
    """

    def __init__(self, method_name=("tbats", "thetaf", "stlar", "arima", "ets"), period=None):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="RForecast",
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True
        )

        self.method_name = method_name
        self.period = period

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "method_name": self._choice("method_name", self.method_name),
            "period": self._randint("period", self.period)
        }

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """
        return RForecastPredictor(
            freq,
            prediction_length,
            params.get("method_name", self.method_name),
            period=params.get("period", self.period),
        )


class TreePredictorParams(BaseParams):
    """
    Parameters class for Tree Predictor Model
    """

    def __init__(self,
                 use_feat_dynamic_real,
                 use_feat_dynamic_cat,
                 model_params=None,
                 method=("QRX", "QuantileRegression"),
                 quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                 context_length=(1, 100),
                 max_workers=None,
                 max_n_datapts=1000000):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="TreePredictor",
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=use_feat_dynamic_real,
            handle_feat_dynamic_cat=use_feat_dynamic_cat
        )

        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        # TreePredictordoes not handle static features properly (even if it is advertised otherwise)
        self.use_feat_static_real = False
        self.model_params = model_params
        self.method = method
        self.context_length = context_length
        self.quantiles = quantiles
        self.max_workers = max_workers
        self.max_n_datapts = max_n_datapts

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "method": self._choice("method", self.method),
            "context_length": self._randint("context_length", self.context_length)
        }

    def build_estimator(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
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
            max_n_datapts=self.max_n_datapts
        )


class ProphetParams(BaseParams):
    """
    Parameter class for Prophet Model
    """

    def __init__(self, growth=("linear"), seasonality_mode=("additive", "multiplicative")):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="Prophet",
            has_estimator=False,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=True,
            handle_feat_dynamic_cat=True
        )

        self.growth = growth
        self.seasonality_mode = seasonality_mode

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "growth": self._choice("growth", self.growth),
            "seasonality_mode": self._choice("seasonality_mode", self.seasonality_mode)
        }

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """
        return ProphetPredictor(
            freq,
            prediction_length=prediction_length,
            prophet_params = {
                "growth": params.get("growth", self.growth),
                "seasonality_mode": params.get("seasonality_mode", self.seasonality_mode),
            }
        )


class FeedForwardParams(BaseParams):
    """
    Parameter class for Feed Forward Model
    """

    def __init__(self,
                 hidden_layer_1_size=None,
                 hidden_layer_2_size=None,
                 hidden_layer_3_size=None,
                 epochs=(1, 100),
                 learning_rate=(0.001, 0.01),
                 context_length=(1, 100),
                 l2=(1e-08, 0.01),
                 mean_scaling=True):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="FeedFoward",
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False
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
            "hidden_layer_1_size": self._randint("hidden_layer_1_size", self.hidden_layer_1_size),
            "hidden_layer_2_size": self._randint("hidden_layer_2_size", self.hidden_layer_2_size),
            "hidden_layer_3_size": self._randint("hidden_layer_3_size", self.hidden_layer_3_size),
            "epochs": self._randint("epochs", self.epochs),
            "learning_rate": self._uniform("learning_rate", self.learning_rate),
            "context_length": self._randint("context_length", self.context_length),
            "l2": self._uniform("l2", self.l2)
        }

    def build_estimator(self, *, ctx, freq, prediction_length, distr_output, params, **kwargs):
        """
        TODO write documentation
        """
        hidden_layer_1_size = params.get("hidden_layer_1_size", self.hidden_layer_1_size)
        hidden_layer_2_size = params.get("hidden_layer_2_size", self.hidden_layer_2_size)
        hidden_layer_3_size = params.get("hidden_layer_3_size", self.hidden_layer_3_size)
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
            )
        )


class DeepARParams(BaseParams):
    """
    Parameter class for Deep AR Model
    """

    def __init__(self,
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
                 impute_missing_values=True):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="DeepAR",
            has_estimator=True,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=use_feat_dynamic_real,
            handle_feat_dynamic_cat=False
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
        self.scaling=scaling
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
            "l2": self._uniform("l2", self.l2)
        }

    def build_estimator(self, *, ctx, freq, prediction_length, distr_output, params, **kwargs):
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
                ctx=ctx, epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                hybridize=False,
                weight_decay=params.get("l2", self.l2),
            )
        )


class TransformerTempFlowParams(BaseParams):
    """
    Parameter class for Tramsformer Temp Flow Model
    """

    def __init__(self,
                 epochs=(1, 100),
                 d_model=(4, 8, 12, 16),
                 num_heads=(1, 2, 4),
                 context_length=(1, 100),
                 flow_type="MAF",
                 learning_rate=(0.0001, 0.01),
                 l2=(0.0001, 0.01),
                 scaling=False):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="TransformerTempFlow",
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False
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
            "l2": self._uniform("l2", self.l2)
        }

    def build_estimator(self, *, device, freq, prediction_length, target_dim, params, **kwargs):
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
            trainer=PtsTrainer(
                device=device,
                epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay = params.get("l2", self.l2),
                num_batches_per_epoch=100,
                batch_size=32,
                num_workers=0,
            )
        )


class DeepVARParams(BaseParams):
    """
    Parameter class for Deep VAR Model
    """

    def __init__(self,
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
                 scaling=False):
        """
        TODO write documentation
        """
        super().__init__(
            "DeepVAR",
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False
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
            "l2": self._uniform("l2", self.l2)
        }

    def build_estimator(self, *, ctx, freq, prediction_length, target_dim, params, **kwargs):
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
                ctx=ctx, epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay = params.get("l2", self.l2),
                hybridize=True,
            )
        )


class GPVarParams(BaseParams):
    """
    Parameter class for GP Var Model
    """

    def __init__(self,
                 epochs=(1, 100),
                 num_layers=(1, 32),
                 num_cells=(1, 100),
                 cell_type=("lstm", "gru"),
                 dropout_rate=(0, 0.5),
                 rank=(1, 20),
                 context_length=(1, 100),
                 learning_rate=(0.0001, 0.01),
                 l2=(1e-4, 0.01)):
        """
        TODO write documentation
        """
        super().__init__(
            model_name="GPVar",
            has_estimator=True,
            handle_feat_static_real=True,
            handle_feat_static_cat=True,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False
        )

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
            "rank": self._randint("rank", self.rank),
            "cell_type": self._choice("cell_type", self.cell_type),
            "l2": self._uniform("l2", self.l2)
        }


    def build_estimator(self, *, ctx, freq, prediction_length, target_dim, params, **kwargs):
        """
        TODO write documentation
        """
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
                ctx=ctx, epochs=params.get("epochs", self.epochs),
                learning_rate=params.get("learning_rate", self.learning_rate),
                weight_decay = params.get("l2", self.l2),
                hybridize=False,
            )
        )


class ConstantValueParams(BaseParams):
    """
    Parameters classs for the Constant Value Model
    """

    def __init__(self, value=(0, 100)):
        """
        TODO write documentation
        """

        super().__init__(
            model_name="ConstantValue",
            has_estimator=False,
            handle_feat_static_real=False,
            handle_feat_static_cat=False,
            handle_feat_dynamic_real=False,
            handle_feat_dynamic_cat=False
        )

        self.value = value

    def tune_config(self):
        """
        TODO write documentation
        """
        return {
            "value": self._uniform("value", self.value)
        }

    def build_predictor(self, *, freq, prediction_length, params, **kwargs):
        """
        TODO write documentation
        """

        return ConstantValuePredictor(
            value=params.get("value", self.value),
            prediction_length=prediction_length,
            freq=freq
        )

