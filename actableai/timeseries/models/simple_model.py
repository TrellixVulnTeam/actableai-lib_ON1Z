import visions
import time

from typing import Dict, List, Optional, Tuple, Any, Iterable

import numpy as np
import pandas as pd
from functools import partial

from hyperopt import hp, fmin, tpe, space_eval

from mxnet.context import Context

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.distribution.student_t import StudentTOutput
from gluonts.mx.distribution.poisson import PoissonOutput
from gluonts.evaluation import Evaluator, MultivariateEvaluator

from actableai.timeseries.models.params import BaseParams
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
    """Simple Time Series Model,"""

    def __init__(
        self,
        target_columns: List[str],
        prediction_length: int,
        freq: str,
        group_dict: Optional[Dict[Tuple[Any], int]] = None,
        real_static_feature_dict: Optional[Dict[Tuple[Any], List[float]]] = None,
        cat_static_feature_dict: Optional[Dict[Tuple[Any], List[Any]]] = None,
        real_dynamic_feature_columns: Optional[List[str]] = None,
        cat_dynamic_feature_columns: Optional[List[str]] = None,
    ):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            target_columns: List of columns to forecast.
            prediction_length: Length of the prediction to forecast.
            freq: Frequency of the time series.
            group_dict: Dictionary containing the unique label for each group.
            real_static_feature_dict: Dictionary containing a list of real features for
                each group.
            cat_static_feature_dict: Dictionary containing a list of categorical
                features for each group.
            real_dynamic_feature_columns: List of columns containing real features.
            cat_dynamic_feature_columns: List of columns containing categorical
                features.
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

    @staticmethod
    def _create_predictor(
        model_params_dict: Dict[str, BaseParams],
        params: Dict[str, Any],
        data: Iterable[Dict[str, Any]],
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_dim: int,
        mx_ctx: Context,
    ) -> AAITimeSeriesPredictor:
        """Create and train a predictor.

        Args:
            model_params_dict: Dictionary containing the different model params for
                each model.
            params: Hyperparameter choose by the tuning.
            data: Data to use for training.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            mx_ctx: mxnet context.

        Returns:
            Trained predictor.
        """
        model_params_class = model_params_dict[params["model"]["model_name"]]

        keep_feat_static_real = model_params_class.handle_feat_static_real
        keep_feat_static_cat = model_params_class.handle_feat_static_cat
        keep_feat_dynamic_real = model_params_class.handle_feat_dynamic_real
        keep_feat_dynamic_cat = model_params_class.handle_feat_dynamic_cat

        if model_params_class.has_estimator:
            gluonts_estimator = model_params_class.build_estimator(
                ctx=mx_ctx,
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
        params: Dict[str, Any],
        *,
        model_params_dict: Dict[str, BaseParams],
        train_data_partial: Iterable[Dict[str, Any]],
        tune_data: Iterable[Dict[str, Any]],
        loss: str,
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_dim: int,
        use_ray: bool,
        mx_ctx: Context,
    ) -> Optional[float]:
        """Create, train, and evaluate a model with specific hyperparameter.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_data_partial: Data to use for training.
            tune_data: Data to use for tuning.
            loss: Loss to return.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            If `use_ray` is False return the loss. Else will report the loss to ray
            tune.
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
        params: Dict[str, Any],
        *,
        model_params_dict: Dict[str, BaseParams],
        train_data_partial: Iterable[Dict[str, Any]],
        tune_data: Iterable[Dict[str, Any]],
        loss: str,
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_dim: int,
        use_ray: bool,
        mx_ctx: Context,
    ) -> Dict[str, Any]:
        """Create, train, and evaluate a model with specific hyperparameter. Used by
            hyperopt.

        Args:
            params: Hyperparameter choose by the tuning.
            model_params_dict: Dictionary containing the different model params for
                each model.
            train_data_partial: Data to use for training.
            tune_data: Data to use for tuning.
            loss: Loss to return.
            freq_gluon: GluonTS frequency of the time series.
            distr_output: Distribution output to use.
            prediction_length: Length of the prediction that will be forecasted.
            target_dim: Target dimension (number of columns to predict).
            use_ray: Whether ray is used for tuning or not.
            mx_ctx: mxnet context.

        Returns:
            Dictionary containing the loss and the status.
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
            ),
            "status": "ok",
        }

    def fit(
        self,
        df_dict: Dict[Tuple[Any], pd.DataFrame],
        model_params: List[BaseParams],
        mx_ctx: Context,
        *,
        loss: str = "mean_wQuantileLoss",
        trials: int = 3,
        max_concurrent: Optional[int] = None,
        use_ray: bool = True,
        tune_samples: int = 3,
        sampling_method: str = "random",
        random_state: Optional[int] = None,
        ray_tune_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fit_full: bool = True,
    ) -> float:
        """Tune and fit the model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            model_params: List of models parameters to run the tuning search on.
            mx_ctx: mxnet context.
            loss: Loss to minimize when tuning.
            trials: Number of trials for hyperparameter search.
            max_concurrent: Maximum number of concurrent ray task.
            use_ray: If True ray will be used for hyperparameter tuning.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].
            random_state: Random state to use for reproducibility.
            ray_tune_kwargs: Named parameters to pass to ray's `tune` function.
            verbose: Verbose level.
            fit_full: If True the model will be fit after tuning using all the data
                (tuning data).

        Returns:
            Total time spent for tuning.
        """
        self.mx_ctx = mx_ctx

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
        )

        return time.time() - start + trials_time_total

    def refit(self, df_dict: Dict[Tuple[Any], pd.DataFrame]):
        """Fit previously tuned model.

        Args:
            df_dict: Dictionary containing the time series for each group.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
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
        )

    def score(
        self,
        df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        num_samples: int = 100,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        num_workers: Optional[int] = None,
    ) -> Tuple[
        Dict[Tuple[Any, ...], pd.DataFrame],
        Dict[Tuple[Any, ...], pd.DataFrame],
        pd.DataFrame,
    ]:
        """Evaluate model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            num_samples: Number of dataset samples to use for evaluation
            quantiles: List of quantiles to use for evaluation.
            num_workers: Maximum number of workers to use, if None no parallelization
                will be done.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            - Dictionary containing the predicted time series for each group.
            - Dictionary containing the metrics for each target for each group.
            - Dataframe containing the aggregated metrics for each target.
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

    def predict(
        self,
        df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        """Make a prediction using the model.

        Args:
            df_dict: Dictionary containing the time series for each group.
            quantiles: Quantiles to predict.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.

        Returns:
            Dictionary containing the predicted time series for each group.
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
