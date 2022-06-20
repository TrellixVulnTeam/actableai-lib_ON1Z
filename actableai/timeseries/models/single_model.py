import mxnet as mx
import numpy as np
import pandas as pd
import time
import visions
from functools import partial
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.distribution.poisson import PoissonOutput
from gluonts.mx.distribution.student_t import StudentTOutput
from hyperopt import hp, fmin, tpe, space_eval
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from typing import Dict, List, Optional, Tuple, Any, Iterable, Union

from actableai.callbacks.time_series import AAITimeSeriesCallback
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.estimator import AAITimeSeriesEstimator
from actableai.timeseries.models.params.base import BaseParams
from actableai.timeseries.models.predictor import AAITimeSeriesPredictor
from actableai.timeseries.models.evaluator import AAITimeSeriesEvaluator
from actableai.timeseries.utils import (
    dataframe_to_list_dataset,
    forecast_to_dataframe,
)


class AAITimeSeriesSingleModel(AAITimeSeriesBaseModel):
    """Simple Time Series Model,"""

    def __init__(
        self,
        target_columns: List[str],
        prediction_length: int,
        freq: str,
        group_label_dict: Optional[Dict[Tuple[Any, ...], int]] = None,
        real_static_feature_dict: Optional[Dict[Tuple[Any, ...], List[float]]] = None,
        cat_static_feature_dict: Optional[Dict[Tuple[Any, ...], List[Any]]] = None,
        real_dynamic_feature_columns: Optional[List[str]] = None,
        cat_dynamic_feature_columns: Optional[List[str]] = None,
    ):
        """AAITimeSeriesBaseModel Constructor.

        Args:
            target_columns: List of columns to forecast.
            prediction_length: Length of the prediction to forecast.
            freq: Frequency of the time series.
            group_label_dict: Dictionary containing the unique label for each group.
            real_static_feature_dict: Dictionary containing a list of real static
                features for each group.
            cat_static_feature_dict: Dictionary containing a list of categorical static
                features for each group.
            real_dynamic_feature_columns: List of columns containing real dynamic
                features.
            cat_dynamic_feature_columns: List of columns containing categorical dynamic
                features.
        """
        super().__init__(
            target_columns,
            prediction_length,
            freq,
            group_label_dict,
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        )

        self.predictor = None
        self.best_params = None

        self.model_params_dict = None
        self.distr_output = None

    @staticmethod
    def _create_predictor(
        model_params_dict: Dict[str, BaseParams],
        params: Dict[str, Any],
        data: Iterable[Dict[str, Any]],
        freq_gluon: str,
        distr_output: DistributionOutput,
        prediction_length: int,
        target_dim: int,
        mx_ctx: mx.Context,
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
        mx_ctx: mx.Context,
        callback: Optional[AAITimeSeriesCallback],
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

        evaluator = AAITimeSeriesEvaluator(
            n_targets=target_dim,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            num_workers=None,
        )
        agg_metrics, df_item_metrics = evaluator(
            ts_it, forecast_it, num_series=len(tune_data)
        )

        if callback is not None:
            # FIXME process the evaluations
            callback.on_tune_model_trained(
                model_params=params,
                agg_metrics=agg_metrics,
                df_item_metrics=df_item_metrics,
            )

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
        mx_ctx: mx.Context,
        callback: Optional[AAITimeSeriesCallback],
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
                callback=callback,
            ),
            "status": "ok",
        }

    def _generate_train_valid_data(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        tune_samples: int,
        sampling_method: str = "random",
    ) -> Tuple[
        Iterable[Dict[str, Any]], Iterable[Dict[str, Any]], Iterable[Dict[str, Any]]
    ]:
        """Generate and split train and validation data for tuning.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            tune_samples: Number of dataset samples to use when tuning.
            sampling_method: Method used when extracting the samples for the tuning
                ["random", "last"].

        Returns:
            - Training ListDataset.
            - Training ListDataset (partial without tuning).
            - Tuning ListDataset.
        """
        from gluonts.dataset.common import ListDataset

        train_data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        train_data_partial = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
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
                dataframe_to_list_dataset(
                    group_df_dict,
                    self.target_columns,
                    self.freq_gluon,
                    real_static_feature_dict=self.real_static_feature_dict,
                    cat_static_feature_dict=self.cat_static_feature_dict,
                    real_dynamic_feature_columns=self.real_dynamic_feature_columns,
                    cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
                    group_label_dict=self.group_label_dict,
                    prediction_length=self.prediction_length,
                    slice_df=slice_function,
                    training=True,
                )
            )

        # Merge all samples into the same ListDataset
        tune_data = None
        if len(tune_data_list) > 0:
            list_data = []

            for tune_data_sample in tune_data_list:
                list_data += tune_data_sample.list_data

            tune_data = ListDataset(
                list_data,
                self.freq_gluon,
                one_dim_target=(len(self.target_columns) == 1),
            )

        return train_data, train_data_partial, tune_data

    def fit(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        model_params: List[BaseParams],
        *,
        mx_ctx: Optional[mx.Context] = mx.cpu(),
        loss: str = "mean_wQuantileLoss",
        trials: int = 1,
        max_concurrent: Optional[int] = 1,
        use_ray: bool = True,
        tune_samples: int = 3,
        sampling_method: str = "random",
        random_state: Optional[int] = None,
        ray_tune_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        fit_full: bool = True,
        callback: Optional[AAITimeSeriesCallback] = None,
    ) -> float:
        """Tune and fit the model.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            model_params: List of models parameters to run the tuning search on.
            mx_ctx: mxnet context, CPU by default.
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
        self.model_params_dict = {
            model_param_class.model_name: model_param_class
            for model_param_class in model_params
            if model_param_class is not None
        }

        # Split data
        train_data, train_data_partial, tune_data = self._generate_train_valid_data(
            group_df_dict,
            tune_samples,
            sampling_method,
        )

        # Choose distribution output
        first_group_targets = train_data.list_data[0]["target"]
        if (first_group_targets >= 0).all() and first_group_targets in visions.Integer:
            self.distr_output = PoissonOutput()
        else:
            self.distr_output = StudentTOutput()

        # Set up search space
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

        # Tune hyperparameters
        if use_ray:
            algo = HyperOptSearch(
                search_space, metric=loss, mode="min", random_state_seed=random_state
            )
            if max_concurrent is not None:
                algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

            if ray_tune_kwargs is None:
                ray_tune_kwargs = {}

            trainable = tune.with_parameters(
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
                mx_ctx=mx_ctx,
                callback=callback,
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
                mx_ctx=mx_ctx,
                callback=callback,
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

        # Create final model using best parameters
        self.predictor = self._create_predictor(
            model_params_dict=self.model_params_dict,
            params=self.best_params,
            data=final_train_data,
            freq_gluon=self.freq_gluon,
            distr_output=self.distr_output,
            prediction_length=self.prediction_length,
            target_dim=len(self.target_columns),
            mx_ctx=mx_ctx,
        )

        return time.time() - start + trials_time_total

    def refit(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        mx_ctx: Optional[mx.Context] = mx.cpu(),
    ):
        """Fit previously tuned model.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
            mx_ctx: mxnet context, CPU by default.

        Raises:
            UntrainedModelException: If the model has not been trained/tuned before.
        """
        if self.predictor is None:
            raise UntrainedModelException()

        train_data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
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
            mx_ctx=mx_ctx,
        )

    def score(
        self,
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
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
            group_df_dict: Dictionary containing the time series for each group.
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
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=True,
        )

        forecast_it, ts_it = self.predictor.make_evaluation_predictions(
            valid_data, num_samples
        )

        ts_list = list(ts_it)
        forecast_list = list(forecast_it)

        # Create custom metric function
        rmse = lambda target, forecast: np.sqrt(np.mean(np.square(target - forecast)))

        # Evaluate
        evaluator = AAITimeSeriesEvaluator(
            n_targets=len(self.target_columns),
            quantiles=quantiles,
            num_workers=num_workers,
            custom_eval_fn={"custom_RMSE": [rmse, "mean", "median"]},
        )
        agg_metrics, df_item_metrics = evaluator(
            ts_list, forecast_list, num_series=len(valid_data)
        )

        # Add predictions
        for (group, df_group), forecast in zip(group_df_dict.items(), forecast_list):
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
            target_list += [target] * len(group_df_dict)
        df_item_metrics["target"] = target_list
        df_item_metrics["group"] = list(group_df_dict.keys()) * len(self.target_columns)
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
        group_df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> Dict[Tuple[Any, ...], pd.DataFrame]:
        """Make a prediction using the model.

        Args:
            group_df_dict: Dictionary containing the time series for each group.
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
                for group, df in group_df_dict.items()
            }
        else:
            future_dates_dict = {
                group: df.index[-self.prediction_length :]
                for group, df in group_df_dict.items()
            }

        df_predictions_dict = {}

        data = dataframe_to_list_dataset(
            group_df_dict,
            self.target_columns,
            self.freq_gluon,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            group_label_dict=self.group_label_dict,
            prediction_length=self.prediction_length,
            training=False,
        )

        forecast_list = self.predictor.predict(data)
        for group, forecast in zip(group_df_dict.keys(), forecast_list):
            df_predictions_dict[group] = forecast_to_dataframe(
                forecast,
                self.target_columns,
                future_dates_dict[group],
                quantiles=quantiles,
            )

        return df_predictions_dict
