import visions
import time

import numpy as np
import pandas as pd
from functools import partial

from hyperopt import hp, fmin, tpe, space_eval

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from gluonts.mx.distribution.student_t import StudentTOutput
from gluonts.mx.distribution.poisson import PoissonOutput
from gluonts.evaluation import Evaluator, MultivariateEvaluator

from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.util import (
    generate_train_valid_data,
    dataframe_to_list_dataset,
    forecast_to_dataframe,
)


class AAITimeSeriesSimpleModel(AAITimeSeriesBaseModel):
    """
    TODO write documentation
    """

    def __init__(
        self,
        target_columns,
        prediction_length,
        freq,
        group_dict=None,
        real_static_feature_dict=None,
        cat_static_feature_dict=None,
        real_dynamic_feature_columns=None,
        cat_dynamic_feature_columns=None,
    ):
        """
        TODO write documentation
        """
        super().__init__(
            target_columns,
            prediction_length,
            freq,
            group_dict,
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

        self.predictor = None
        self.best_params = None

        self.model_params_dict = None
        self.distr_output = None
        self.mx_ctx = None
        self.torch_device = None

    @staticmethod
    def _create_predictor(
        model_params_dict,
        params,
        data,
        freq_gluon,
        distr_output,
        prediction_length,
        target_dim,
        mx_ctx,
        torch_device,
    ):
        """
        TODO write documentation
        """
        model_params_class = model_params_dict[params["model"]["model_name"]]

        keep_feat_static_real = model_params_class.handle_feat_static_real
        keep_feat_static_cat = model_params_class.handle_feat_static_cat
        keep_feat_dynamic_real = model_params_class.handle_feat_dynamic_real
        keep_feat_dynamic_cat = model_params_class.handle_feat_dynamic_cat

        if model_params_class.has_estimator:
            gluonts_estimator = model_params_class.build_estimator(
                ctx=mx_ctx,
                device=torch_device,
                freq=freq_gluon,
                prediction_length=prediction_length,
                target_dim=target_dim,
                distr_output=distr_output,
                params=params["model"],
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
                freq=freq_gluon,
                prediction_length=prediction_length,
                params=params["model"],
            )

        return AAITimeSeriesPredictor(
            predictor,
            keep_feat_static_real,
            keep_feat_static_cat,
            keep_feat_dynamic_real,
            keep_feat_dynamic_cat,
        )

    @classmethod
    def _trainable(
        cls,
        params,
        *,
        model_params_dict,
        train_data_partial,
        tune_data,
        loss,
        freq_gluon,
        distr_output,
        prediction_length,
        target_dim,
        use_ray,
        mx_ctx,
        torch_device,
    ):
        """
        TODO write documentation
        """
        predictor = cls._create_predictor(
            model_params_dict,
            params,
            train_data_partial,
            freq_gluon,
            distr_output,
            prediction_length,
            target_dim,
            mx_ctx,
            torch_device,
        )

        forecast_it, ts_it = predictor.make_evaluation_predictions(
            tune_data, num_samples=100
        )

        evaluator_class = None
        if target_dim <= 1:
            evaluator_class = Evaluator
        else:
            evaluator_class = MultivariateEvaluator
        evaluator = evaluator_class(
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], num_workers=None
        )

        agg_metrics, _ = evaluator(ts_it, forecast_it, num_series=len(tune_data))

        if not use_ray:
            return agg_metrics[loss]

        tune.report(**{loss: agg_metrics[loss]})

    @classmethod
    def _objective(
        cls,
        params,
        *,
        model_params_dict,
        train_data_partial,
        tune_data,
        loss,
        freq_gluon,
        distr_output,
        prediction_length,
        target_dim,
        use_ray,
        mx_ctx,
        torch_device,
    ):

        """
        TODO write documentation
        """
        return {
            "loss": cls._trainable(
                params,
                model_params_dict=model_params_dict,
                train_data_partial=train_data_partial,
                tune_data=tune_data,
                loss=loss,
                freq_gluon=freq_gluon,
                distr_output=distr_output,
                prediction_length=prediction_length,
                target_dim=target_dim,
                use_ray=use_ray,
                mx_ctx=mx_ctx,
                torch_device=torch_device,
            ),
            "status": "ok",
        }

    def fit(
        self,
        df_dict,
        model_params,
        mx_ctx,
        torch_device,
        *,
        loss="mean_wQuantileLoss",
        trials=3,
        max_concurrent=None,
        use_ray=True,
        tune_samples=3,
        sampling_method="random",
        random_state=None,
        ray_tune_kwargs=None,
        verbose=1,
        fit_full=True,
    ):
        """
        TODO write documentation
        """
        self.mx_ctx = mx_ctx
        self.torch_device = torch_device

        self.model_params_dict = {
            model_param_class.model_name: model_param_class
            for model_param_class in model_params
            if model_param_class is not None
        }

        train_data, train_data_partial, tune_data = generate_train_valid_data(
            df_dict,
            self.target_columns,
            self.prediction_length,
            self.freq_gluon,
            self.group_dict,
            self.real_static_feature_dict,
            self.cat_static_feature_dict,
            self.real_dynamic_feature_columns,
            self.cat_dynamic_feature_columns,
            tune_samples,
            sampling_method,
        )

        first_group_targets = train_data.list_data[0]["target"]
        if (first_group_targets >= 0).all() and first_group_targets in visions.Integer:
            self.distr_output = PoissonOutput()
        else:
            self.distr_output = StudentTOutput()

        models_search_space = []
        for model_name, model_param_class in self.model_params_dict.items():
            if model_param_class is not None:
                models_search_space.append(
                    {
                        "model_name": model_name,
                        **model_param_class.tune_config(),
                    }
                )
        search_space = {"model": hp.choice("model", models_search_space)}

        trials_time_total = 0

        if use_ray:
            algo = HyperOptSearch(
                search_space, metric=loss, mode="min", random_state_seed=random_state
            )
            if max_concurrent is not None:
                algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

            if ray_tune_kwargs is None:
                ray_tune_kwargs = {}

            trainable = partial(
                self._trainable,
                model_params_dict=self.model_params_dict,
                train_data_partial=train_data_partial,
                tune_data=tune_data,
                loss=loss,
                freq_gluon=self.freq_gluon,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
                target_dim=len(self.target_columns),
                use_ray=use_ray,
                mx_ctx=self.mx_ctx,
                torch_device=self.torch_device,
            )

            analysis = tune.run(
                trainable,
                search_alg=algo,
                num_samples=trials,
                verbose=verbose,
                **ray_tune_kwargs,
            )

            for result in analysis.results.values():
                if result is not None:
                    trials_time_total += result.get("time_total_s", 0)

            start = time.time()

            self.best_params = analysis.get_best_config(metric=loss, mode="min")
        else:
            start = time.time()

            objective = partial(
                self._objective,
                model_params_dict=self.model_params_dict,
                train_data_partial=train_data_partial,
                tune_data=tune_data,
                loss=loss,
                freq_gluon=self.freq_gluon,
                distr_output=self.distr_output,
                prediction_length=self.prediction_length,
                target_dim=len(self.target_columns),
                use_ray=use_ray,
                mx_ctx=self.mx_ctx,
                torch_device=self.torch_device,
            )

            best = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=trials,
            )
            self.best_params = space_eval(space=search_space, hp_assignment=best)

        if fit_full:
            final_train_data = train_data
        else:
            final_train_data = train_data_partial

        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            data=final_train_data,
            freq_gluon=self.freq_gluon,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            target_dim=len(self.target_columns),
            mx_ctx=self.mx_ctx,
            torch_device=self.torch_device,
        )

        return time.time() - start + trials_time_total

    def refit(self, df_dict):
        """
        TODO write documentation
        """
        if self.predictor is None:
            raise UntrainedModelException()

        train_data = dataframe_to_list_dataset(
            df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_dict=self.group_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            data=train_data,
            freq_gluon=self.freq_gluon,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            target_dim=len(self.target_columns),
            mx_ctx=self.mx_ctx,
            torch_device=self.torch_device,
        )

    def score(
        self, df_dict, num_samples=100, quantiles=[0.05, 0.5, 0.95], num_workers=None
    ):
        """
        TODO write documentation
        """
        if self.predictor is None:
            raise UntrainedModelException()

        df_predictions_dict = {}

        valid_data = dataframe_to_list_dataset(
            df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_dict=self.group_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        forecast_it, ts_it = self.predictor.make_evaluation_predictions(
            valid_data, num_samples
        )

        ts_list = list(ts_it)
        forecast_list = list(forecast_it)

        rmse = lambda target, forecast: np.sqrt(np.mean(np.square(target - forecast)))

        # Evaluate
        evaluator_class = None
        if len(self.target_columns) <= 1:
            evaluator_class = Evaluator
        else:
            evaluator_class = MultivariateEvaluator
        evaluator = evaluator_class(
            quantiles=quantiles,
            num_workers=num_workers,
            custom_eval_fn={"custom_RMSE": [rmse, "mean", "median"]},
        )

        agg_metrics, df_item_metrics = evaluator(
            ts_list, forecast_list, num_series=len(valid_data)
        )

        # Add predictions
        for (group, df_group), forecast in zip(df_dict.items(), forecast_list):
            df_predictions_dict[group] = forecast_to_dataframe(
                forecast,
                self.target_columns,
                df_group.index[-self.prediction_length :],
                quantiles=quantiles,
            )

        # Post-process metrics
        # item_metrics
        target_list = []
        for target in self.target_columns:
            target_list += [target] * len(df_dict)
        df_item_metrics["target"] = target_list
        df_item_metrics["group"] = list(df_dict.keys()) * len(self.target_columns)
        df_item_metrics = df_item_metrics.reset_index(drop=True)
        df_item_metrics = df_item_metrics.rename(columns={"custom_RMSE": "RMSE"})

        # agg_metrics
        if len(self.target_columns) <= 1:
            df_agg_metrics = pd.DataFrame(
                [{"target": self.target_columns[0], **agg_metrics}]
            )
        else:
            metric_list = list(agg_metrics.keys())[
                (len(agg_metrics) // (len(self.target_columns) + 1))
                * len(self.target_columns) :
            ]
            df_agg_metrics = pd.DataFrame(columns=["target"] + metric_list)

            for target_index, target_column in enumerate(self.target_columns):
                target_agg_metrics = {
                    metric: agg_metrics[f"{target_index}_{metric}"]
                    for metric in metric_list
                }
                df_agg_metrics = pd.concat(
                    [
                        df_agg_metrics,
                        pd.DataFrame([{"target": target_column, **target_agg_metrics}]),
                    ],
                    ignore_index=True,
                )

        df_agg_metrics = df_agg_metrics.drop(columns="RMSE").rename(
            columns={"custom_RMSE": "RMSE"}
        )

        df_item_metrics_dict = {}
        for group, df_group in df_item_metrics.groupby("group"):
            df_item_metrics_dict[group] = df_group

        return df_predictions_dict, df_item_metrics_dict, df_agg_metrics

    def predict(self, df_dict, quantiles=[0.05, 0.5, 0.95]):
        """
        TODO write documentation
        """
        if self.predictor is None:
            raise UntrainedModelException()

        if not self.has_dynamic_features:
            future_dates_dict = {
                group: pd.date_range(
                    df.index[-1], periods=self.prediction_length + 1, freq=self.freq
                )[1:]
                for group, df in df_dict.items()
            }
        else:
            future_dates_dict = {
                group: df.index[-self.prediction_length :]
                for group, df in df_dict.items()
            }

        df_predictions_dict = {}

        data = dataframe_to_list_dataset(
            df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_dict=self.group_dict,
            prediction_length=self.prediction_length,
            training=False,
        )

        forecast_list = self.predictor.predict(data)
        for group, forecast in zip(df_dict.keys(), forecast_list):
            df_predictions_dict[group] = forecast_to_dataframe(
                forecast,
                self.target_columns,
                future_dates_dict[group],
                quantiles=quantiles,
            )

        return df_predictions_dict
