import time
import json
import ray
import visions
import numpy as np
import pandas as pd
from ray import tune
from hyperopt import hp, fmin, tpe, space_eval
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from gluonts.dataset.common import ListDataset
from gluonts.mx.distribution.student_t import StudentTOutput
from gluonts.mx.distribution.poisson import PoissonOutput
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.evaluation import Evaluator

from actableai.timeseries import util
from actableai.timeseries.estimator import AAITimeSeriesEstimator
from actableai.timeseries.predictor import AAITimeSeriesPredictor
from actableai.timeseries.exceptions import (
    InvalidFrequencyException,
    UntrainedModelException,
)
from actableai.timeseries.params import (
    ProphetParams,
    FeedForwardParams,
    DeepARParams,
    GPVarParams,
    RForecastParams,
    TransformerTempFlowParams,
    DeepVARParams,
    TreePredictorParams,
)


class AAITimeSeriesForecaster(object):
    """FIXME update documentation
    This timeseries forecaster does an extensive search of alogrithms and their hyperparameters to choose the best
    algorithm for a given data set.

    Parameters
    ----------
    prediction_length : how many steps in future the model should forecast.

    mx_ctc: MXNet context where the model shall run.

    torch_device: torch device where the model shall run.

    """

    def __init__(self, prediction_length, mx_ctx, torch_device, model_params=None):
        """
        TODO write documentation
        """
        self.prediction_length = prediction_length
        self.mx_ctx = mx_ctx
        self.torch_device = torch_device
        self.freq = None
        self.predictors = None

        self.model_params = (
            {params.model_name: params for params in model_params}
            if model_params is not None
            else {}
        )

    def _get_shift_target_columns(self):
        """
        TODO write documentation
        """
        return {
            target_column: f"_{target_column}_shift"
            for target_column in self.target_columns
        }

    def _pre_process_data(self, df_dict, training=True, keep_future=True):
        """
        TODO write documentation
        """
        if self.target_dim <= 1:
            return df_dict

        df_dict_new = {}
        for group in df_dict.keys():
            # Create the shifted dataframe
            df_shift = df_dict[group]
            if not training and self.has_dynamic_features:
                df_shift = df_shift.iloc[: -self.prediction_length]

            df_shift = df_shift[self.target_columns].shift(
                self.prediction_length, freq=self.freq
            )
            # Rename columns
            df_shift = df_shift.rename(columns=self.shift_target_columns_dict)

            if not keep_future:
                df_shift = df_shift.loc[df_shift.index.isin(df_dict[group].index)]

            # Add new features
            df_dict_new[group] = pd.concat([df_dict[group], df_shift], axis=1)

            df_dict_new[group] = df_dict_new[group].iloc[self.prediction_length :]

        return df_dict_new

    @staticmethod
    def _create_predictor(
        model_params,
        params,
        data,
        freq,
        distr_output,
        prediction_length,
        mx_ctx,
        torch_device,
    ):
        model_params_class = model_params.get(params["name"])

        keep_feat_static_real = model_params_class.handle_feat_static_real
        keep_feat_static_cat = model_params_class.handle_feat_static_cat
        keep_feat_dynamic_real = model_params_class.handle_feat_dynamic_real
        keep_feat_dynamic_cat = model_params_class.handle_feat_dynamic_cat

        if model_params_class.has_estimator:
            gluonts_estimator = model_params_class.build_estimator(
                ctx=mx_ctx,
                device=torch_device,
                freq=freq,
                prediction_length=prediction_length,
                target_dim=1,
                distr_output=distr_output,
                params=params,
            )

            estimator = AAITimeSeriesEstimator(
                gluonts_estimator,
                keep_feat_static_real,
                keep_feat_static_cat,
                keep_feat_dynamic_real,
                keep_feat_dynamic_cat,
            )

            predictor = estimator.train(training_data=data)
        else:
            predictor = model_params_class.build_predictor(
                freq=freq, prediction_length=prediction_length, params=params
            )

        return AAITimeSeriesPredictor(
            predictor,
            keep_feat_static_real,
            keep_feat_static_cat,
            keep_feat_dynamic_real,
            keep_feat_dynamic_cat,
        )

    @classmethod
    def _trainable(cls, params):
        np.random.seed(params["seed"])
        predictor = cls._create_predictor(
            params["model_params"],
            params["model"],
            params["train_data_partial"],
            params["freq"],
            params["distr_output"],
            params["prediction_length"],
            params["mx_ctx"],
            params["torch_device"],
        )

        forecast_it, ts_it = predictor.make_evaluation_predictions(
            params["tune_data"], num_samples=100
        )

        evaluator = Evaluator(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        agg_metrics, item_metrics = evaluator(
            ts_it, forecast_it, num_series=len(params["tune_data"])
        )

        if not params["use_ray"]:
            return {params["loss"]: agg_metrics[params["loss"]]}

        tune.report(**{params["loss"]: agg_metrics[params["loss"]]})

    def _fit_predictor(
        self,
        train_data,
        train_data_partial,
        tune_data,
        use_ray,
        trials,
        loss,
        tune_params,
        max_concurrent,
        verbose,
        seed,
    ):
        """
        TODO write documentation
        """
        first_group_targets = train_data.list_data[0]["target"]
        if (first_group_targets >= 0).all() and first_group_targets in visions.Integer:
            self.distr_output = PoissonOutput()
        else:
            self.distr_output = StudentTOutput()

        def objective_function(params):
            return {"loss": self._trainable(params)[loss], "status": "ok"}

        models = []
        for name, p in self.model_params.items():
            if p is not None:
                params = p.tune_config()
                params["name"] = p.model_name
                models.append(params)

        config = {
            "model_params": self.model_params,
            "model": hp.choice("model", models),
            "seed": seed,
            "freq": self.freqGluon,
            "distr_output": self.distr_output,
            "prediction_length": self.prediction_length,
            "mx_ctx": self.mx_ctx,
            "torch_device": self.torch_device,
            "train_data_partial": train_data_partial,
            "tune_data": tune_data,
            "loss": loss,
            "use_ray": use_ray,
        }

        # get time of each trial
        time_total_s = 0

        if use_ray:
            algo = HyperOptSearch(
                config, metric=loss, mode="min", random_state_seed=seed
            )
            if max_concurrent is not None:
                algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

            if tune_params is None:
                tune_params = {}

            np.random.seed(seed)
            analysis = tune.run(
                self._trainable,
                search_alg=algo,
                num_samples=trials,
                verbose=verbose,
                **tune_params,
            )

            for _, result in analysis.results.items():
                if result is not None and "time_total_s" in result:
                    time_total_s += result["time_total_s"]

            start = time.time()

            params = analysis.get_best_config(metric=loss, mode="min")
        else:
            start = time.time()

            best = fmin(
                fn=objective_function, space=config, algo=tpe.suggest, max_evals=trials
            )
            params = space_eval(space=config, hp_assignment=best)

        predictor = self._create_predictor(
            self.model_params,
            params["model"],
            train_data,
            self.freqGluon,
            self.distr_output,
            self.prediction_length,
            self.mx_ctx,
            self.torch_device,
        )

        total_trial_time = time.time() - start + time_total_s
        return predictor, params, total_trial_time

    def fit(
        self,
        df_dict,
        freq,
        target_columns,
        *,
        real_static_feature_dict=None,
        cat_static_feature_dict=None,
        real_dynamic_feature_columns=None,
        cat_dynamic_feature_columns=None,
        group_dict=None,
        trials=3,
        loss="mean_wQuantileLoss",
        tune_params=None,
        max_concurrent=None,
        tune_samples=3,
        use_ray=True,
        verbose=3,
        seed=123,
        sampling_method="random",
    ):
        """
        FIXME documentation

        Parameters
        ----------
        df: Input data frame with its index being the date time, all columns are be forecasted.
        trials
        loss
        tune_params
        max_concurrent
        tune_samples: Number of random samples taken for evaluation during hyper-parameter tuning.

        use_ray: bool
            If True will use ray to fit the model

        Returns
        -------

        """
        if real_static_feature_dict is None:
            real_static_feature_dict = {}
        if cat_static_feature_dict is None:
            cat_static_feature_dict = {}
        if real_dynamic_feature_columns is None:
            real_dynamic_feature_columns = []
        if cat_dynamic_feature_columns is None:
            cat_dynamic_feature_columns = []
        if group_dict is None:
            group_dict = {}

        self.target_columns = target_columns
        self.group_dict = group_dict
        self.target_dim = len(target_columns)
        self.freq = freq
        self.has_dynamic_features = (
            len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) > 0
        )

        self.shift_target_columns_dict = self._get_shift_target_columns()

        self.real_static_feature_dict = real_static_feature_dict
        self.cat_static_feature_dict = cat_static_feature_dict
        self.real_dynamic_feature_columns = real_dynamic_feature_columns
        self.cat_dynamic_feature_columns = cat_dynamic_feature_columns
        self.shift_target_columns = list(self.shift_target_columns_dict.values())

        if self.freq is None:
            raise InvalidFrequencyException()
        self.freqGluon = util.find_gluonts_freq(self.freq)

        # Pre-process data
        df_dict_clean = self._pre_process_data(
            df_dict, training=True, keep_future=False
        )

        self.predictors = {}
        self.tuned_model_params = {}
        self.total_trial_time = 0

        # Build predictors for each target
        for target_column in self.target_columns:
            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            # Gather training, and validation data
            (
                train_data,
                train_data_partial,
                tune_data,
            ) = self._generate_train_and_valid_data(
                df_dict_clean,
                target_column,
                self.real_static_feature_dict,
                self.cat_static_feature_dict,
                real_dynamic_feature_columns,
                self.cat_dynamic_feature_columns,
                tune_samples,
                sampling_method=sampling_method,
            )

            (
                self.predictors[target_column],
                self.tuned_model_params[target_column],
                target_trial_time,
            ) = self._fit_predictor(
                train_data,
                train_data_partial,
                tune_data,
                use_ray,
                trials,
                loss,
                tune_params,
                max_concurrent,
                verbose,
                seed,
            )
            self.total_trial_time += target_trial_time

        return self

    def refit(self, df_dict):
        """Refit data with the best trained model params. Typically used to re-train with validation data included."""
        # Pre-process data
        df_dict_clean = self._pre_process_data(
            df_dict, training=True, keep_future=False
        )

        # Refit with saved best model params
        for target_column in self.target_columns:
            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            train_data = util.dataframe_to_list_dataset(
                df_dict_clean,
                [target_column],
                self.freqGluon,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
                group_dict=self.group_dict,
                prediction_length=self.prediction_length,
                training=True,
            )
            self.predictors[target_column] = self._create_predictor(
                self.model_params,
                self.tuned_model_params[target_column]["model"],
                train_data,
                self.freqGluon,
                self.distr_output,
                self.prediction_length,
                self.mx_ctx,
                self.torch_device,
            )

        return self

    def score(
        self, df_dict, num_samples=100, quantiles=[0.05, 0.5, 0.95], num_workers=0
    ):
        """
        TODO write documentation
        """
        if self.predictors is None:
            raise UntrainedModelException()

        df_dict_clean = self._pre_process_data(
            df_dict, training=True, keep_future=False
        )

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}
        df_item_metrics = pd.DataFrame()
        df_agg_metrics = pd.DataFrame()

        for target_index, target_column in enumerate(self.target_columns):
            if target_column not in self.predictors:
                raise UntrainedModelException()

            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            valid_data = util.dataframe_to_list_dataset(
                df_dict_clean,
                [target_column],
                self.freqGluon,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
                group_dict=self.group_dict,
                prediction_length=self.prediction_length,
                training=True,
            )

            forecast_it, ts_it = self.predictors[
                target_column
            ].make_evaluation_predictions(valid_data, num_samples)

            tss = list(ts_it)
            forecasts = list(forecast_it)

            evaluator = Evaluator(quantiles=quantiles, num_workers=num_workers)
            target_agg_metrics, df_target_item_metrics = evaluator(
                tss, forecasts, num_series=len(valid_data)
            )

            # Add predictions
            for (group, df_group), forecast in zip(df_dict_clean.items(), forecasts):
                df_predictions = util.forecast_to_dataframe(
                    forecast, target_column, df_group.index[-self.prediction_length :]
                )

                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_predictions], ignore_index=True
                )

            # post-process metrics
            # item_metrics
            df_target_item_metrics["target"] = target_column
            df_target_item_metrics["group"] = list(df_dict_clean.keys())
            df_item_metrics = pd.concat(
                [df_item_metrics, df_target_item_metrics], ignore_index=True
            )

            # agg_metrics
            df_agg_metrics = pd.concat(
                [
                    df_agg_metrics,
                    pd.DataFrame([{"target": target_column, **target_agg_metrics}]),
                ],
                ignore_index=True,
            )

        df_item_metrics_dict = {}
        for group, df_group in df_item_metrics.groupby("group"):
            df_item_metrics_dict[group] = df_group

        return {
            "predictions": df_predictions_dict,
            "item_metrics": df_item_metrics_dict,
            "agg_metrics": df_agg_metrics,
        }

    def predict(self, df_dict):
        if self.predictors is None:
            raise UntrainedModelException()

        df_dict_clean = self._pre_process_data(
            df_dict, training=False, keep_future=True
        )

        if not self.has_dynamic_features and self.target_dim <= 1:
            future_dates_dict = {
                group: pd.date_range(
                    df.index[-1], periods=self.prediction_length + 1, freq=self.freq
                )[1:]
                for group, df in df_dict_clean.items()
            }
        else:
            future_dates_dict = {
                group: df.index[-self.prediction_length :]
                for group, df in df_dict_clean.items()
            }

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}

        for target_column in self.target_columns:
            if target_column not in self.predictors:
                raise UntrainedModelException()

            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            data = util.dataframe_to_list_dataset(
                df_dict_clean,
                [target_column],
                self.freqGluon,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
                group_dict=self.group_dict,
                prediction_length=self.prediction_length,
                training=False,
            )

            forecasts = self.predictors[target_column].predict(data)
            for group, forecast in zip(df_dict_clean.keys(), forecasts):
                df_predictions = util.forecast_to_dataframe(
                    forecast, target_column, future_dates_dict[group]
                )

                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_predictions], ignore_index=True
                )

        return {"predictions": df_predictions_dict}

    def _generate_train_and_valid_data(
        self,
        df_dict,
        target_column,
        real_static_feature_dict,
        cat_static_feature_dict,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
        tune_samples,
        sampling_method="random",
    ):
        """
        TODO write documentation
        """
        train_data = util.dataframe_to_list_dataset(
            df_dict,
            [target_column],
            self.freqGluon,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
            group_dict=self.group_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        train_data_partial = util.dataframe_to_list_dataset(
            df_dict,
            [target_column],
            self.freqGluon,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
            group_dict=self.group_dict,
            prediction_length=self.prediction_length,
            slice_df=slice(-self.prediction_length - tune_samples),
            training=True,
        )

        tune_data_list = []
        for i in range(tune_samples):
            slice_function = None
            if sampling_method == "random":
                slice_function = lambda df: slice(
                    np.random.randint(2 * self.prediction_length + 1, df.shape[0] + 1)
                )
            elif sampling_method == "last":
                slice_function = lambda df: slice(df.shape[0] - i)
            else:
                raise Exception("Unkown sampling method")

            tune_data_list.append(
                util.dataframe_to_list_dataset(
                    df_dict,
                    [target_column],
                    self.freqGluon,
                    real_static_feature_dict=real_static_feature_dict,
                    cat_static_feature_dict=cat_static_feature_dict,
                    real_dynamic_feature_columns=real_dynamic_feature_columns,
                    cat_dynamic_feature_columns=cat_dynamic_feature_columns,
                    group_dict=self.group_dict,
                    prediction_length=self.prediction_length,
                    slice_df=slice_function,
                    training=True,
                )
            )

        # Merge all samples into the same ListDataset
        tune_data = None
        if len(tune_data_list) > 0:
            freq = self.freqGluon
            list_data = []

            for tune_data_sample in tune_data_list:
                list_data += tune_data_sample.list_data

            tune_data = ListDataset(list_data, freq, one_dim_target=True)

        return train_data, train_data_partial, tune_data
