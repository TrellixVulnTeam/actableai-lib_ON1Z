import time
import ray
import visions
import numpy as np
import pandas as pd
from ray import tune
from hyperopt import hp, fmin, tpe, space_eval
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from gluonts.dataset.common import ListDataset
from gluonts.distribution.student_t import StudentTOutput
from gluonts.distribution.poisson import PoissonOutput
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator

from actableai.timeseries import util
from actableai.timeseries.params import (
    ProphetParams,
    FeedForwardParams,
    DeepARParams,
    GPVarParams,
    RForecastParams,
    TransformerTempFlowParams,
    DeepVARParams,
)


class InvalidFrequencyException(ValueError):
    pass


class UntrainedModelException(ValueError):
    pass


class MismatchedDataDimensionException(ValueError):
    pass


class EmptyModelParams(ValueError):
    pass


class AAITimeseriesForecaster(object):
    """This timeseries forecaster does an extensive search of alogrithms and their hyperparameters to choose the best
    algorithm for a given data set.

    Parameters
    ----------
    prediction_length : how many steps in future the model should forecast.

    mx_ctc: MXNet context where the model shall run.

    torch_device: torch device where the model shall run.

    """

    def __init__(
        self,
        prediction_length,
        mx_ctx,
        torch_device,
        univariate_model_params=None,
        multivariate_model_params=None,
    ):
        self.prediction_length = prediction_length
        self.mx_ctx = mx_ctx
        self.torch_device = torch_device
        self.freq, self.predictor = None, None

        self.univariate_model_params = (
            {params.MODEL_NAME: params for params in univariate_model_params}
            if univariate_model_params is not None
            else {}
        )

        self.multivariate_model_params = (
            {params.MODEL_NAME: params for params in multivariate_model_params}
            if multivariate_model_params is not None
            else {}
        )

    def fit(
        self,
        df,
        trials=3,
        loss="mean_wQuantileLoss",
        tune_params=None,
        max_concurrent=None,
        eval_samples=3,
        use_ray=True,
        seed=123,
    ):
        """

        Parameters
        ----------
        df: Input data frame with its index being the date time, all columns are be forecasted.
        trials
        loss
        tune_params
        max_concurrent
        eval_samples: Number of random samples taken for evaluation during hyper-parameter tuning.

        use_ray: bool
            If True will use ray to fit the model

        Returns
        -------

        """
        self.target_dim = len(df.columns)
        self.freq = util.findFred(df.index)
        self.scaler = None
        # Interpolate missing values with median value in the range
        df = df.resample(self.freq).median().asfreq(self.freq)

        if self.freq is None:
            raise InvalidFrequencyException()

        self.freqGluon = util.findFredForGluon(self.freq)

        if self.target_dim == 1:
            if not self.univariate_model_params:
                raise EmptyModelParams("univariate_model_params can't be None or empty")
            if all(df >= 0) and df in visions.Integer:
                distr_output = PoissonOutput()
            else:
                distr_output = StudentTOutput()
            self.predictor, self.total_trial_time = self._create_univariate_predictor(
                df[df.columns[0]],
                distr_output,
                trials,
                loss,
                tune_params,
                max_concurrent,
                eval_samples,
                use_ray,
                seed,
            )
        else:
            if not self.multivariate_model_params:
                raise EmptyModelParams(
                    "multivariate_model_params can't be None or empty"
                )
            (
                self.predictor,
                self.total_trial_time,
                self.scaler,
            ) = self._create_multivariate_predictor(
                df,
                trials,
                loss,
                tune_params,
                max_concurrent,
                eval_samples,
                use_ray,
                seed,
            )

    def score(self, df, num_samples=100, quantiles=[0.1, 0.5, 0.95], num_workers=0):
        if self.predictor is None:
            raise UntrainedModelException()

        target_dim = len(df.columns)
        result = (
            self._score_univariate(
                df[df.columns[0]], num_samples, quantiles, num_workers
            )
            if target_dim == 1
            else self._score_multivariate(df, num_samples, quantiles, num_workers)
        )

        result["item_metrics"]["item_id"] = df.columns
        result["item_metrics"].index = df.columns
        return result

    def predict(self, df):
        if self.predictor is None:
            raise UntrainedModelException()

        if self.target_dim != len(df.columns):
            raise MismatchedDataDimensionException()
        if self.target_dim > 1:
            df = util.minmax_scaler_transform(df, self.scaler)

        data = ListDataset(
            [
                {
                    "start": df.index[0],
                    "target": [df[col].values for col in df.columns]
                    if self.target_dim > 1
                    else df[df.columns[0]],
                }
            ],
            freq=self.freqGluon,
            one_dim_target=self.target_dim == 1,
        )

        future = util.make_future_dataframe(
            self.prediction_length, df.index, self.freqGluon, include_history=False
        )
        p_date = future["ds"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        predictions = self.predictor.predict(data)

        if self.target_dim > 1:
            predictions = list(predictions)
            predictions = util.inverse_transform(predictions, self.scaler)

        values = self._quantiles(predictions)
        return {"date": p_date, "values": values}

    def _quantiles(self, predictions):
        if self.target_dim == 1:
            values = [
                [
                    {
                        "q5": p.quantile(0.05).astype(float),
                        "q50": p.quantile(0.50).astype(float),
                        "q95": p.quantile(0.95).astype(float),
                    }
                ]
                for p in predictions
            ]
        else:
            quantiles = [
                [
                    p.quantile(0.05).astype(float),
                    p.quantile(0.50).astype(float),
                    p.quantile(0.95).astype(float),
                ]
                for p in predictions
            ]
            values = [
                [
                    {
                        "q5": q5,
                        "q50": q50,
                        "q95": q95,
                    }
                    for q5, q50, q95 in zip(q5s.T, q50s.T, q95s.T)
                ]
                for q5s, q50s, q95s in quantiles
            ]
        return values

    def _create_multivariate_predictor(
        self, df, trials, loss, tune_params, max_concurrent, eval_samples, use_ray, seed
    ):
        target_dim = len(df.columns)
        df, scaler = util.minmax_scaler_fit_transform(df)

        train_data = ListDataset(
            [
                {
                    "start": df.index[0],
                    "target": [
                        df[col].values[: df.shape[0] - self.prediction_length]
                        for col in df.columns
                    ],
                }
            ],
            freq=self.freqGluon,
            one_dim_target=False,
        )

        valid_data = []
        for i in range(eval_samples):
            length = np.random.randint(2 * self.prediction_length, df.shape[0] + 1)
            valid_data.append(
                {
                    "start": df.index[0],
                    "target": [df[col][:length].values for col in df.columns],
                }
            )

        valid_data = ListDataset(valid_data, freq=self.freqGluon, one_dim_target=False)

        def build_estimator(params):
            estimator = None
            if params["name"] == FeedForwardParams.MODEL_NAME:
                estimator = self.multivariate_model_params[params["name"]].build(
                    self.mx_ctx,
                    self.freqGluon,
                    self.prediction_length,
                    MultivariateGaussianOutput(dim=df.shape[1]),
                    params,
                )
            elif params["name"] == GPVarParams.MODEL_NAME:
                estimator = self.multivariate_model_params[params["name"]].build(
                    self.mx_ctx,
                    self.freqGluon,
                    self.prediction_length,
                    target_dim,
                    params,
                )
            elif params["name"] == TransformerTempFlowParams.MODEL_NAME:
                estimator = self.multivariate_model_params[params["name"]].build(
                    self.torch_device,
                    self.freqGluon,
                    self.prediction_length,
                    target_dim,
                    params,
                )
            elif params["name"] == DeepVARParams.MODEL_NAME:
                estimator = self.multivariate_model_params[params["name"]].build(
                    self.mx_ctx,
                    self.freqGluon,
                    self.prediction_length,
                    target_dim,
                    params,
                )
            return estimator

        def create_predictor(params):
            estimator = build_estimator(params)
            predictor = estimator.train(training_data=train_data)
            return predictor

        def trainable(params):
            np.random.seed(params["seed"])
            predictor = create_predictor(params["model"])
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=valid_data, predictor=predictor, num_samples=100
            )
            forecasts = list(forecast_it)
            tss = list(ts_it)
            evaluator = MultivariateEvaluator(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            agg_metrics, item_metrics = evaluator(
                iter(tss), iter(forecasts), num_series=len(valid_data)
            )

            if not use_ray:
                return {loss: agg_metrics[loss]}

            tune.report(**{loss: agg_metrics[loss]})

        def objective_function(params):
            train_results = trainable(params)

            return {"loss": train_results[loss], "status": "ok"}

        models = []
        for name, p in self.multivariate_model_params.items():
            if p is not None:
                params = p.tune_config()
                params["name"] = p.MODEL_NAME
                models.append(params)
        config = {
            "model": hp.choice("model", models),
            "seed": seed,
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

            analysis = tune.run(
                trainable, search_alg=algo, num_samples=trials, **tune_params
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

        predictor = create_predictor(params["model"])

        time_taken = time.time() - start + time_total_s

        return predictor, time_taken, scaler

    def _score_multivariate(self, df, num_samples, quantiles, num_workers):
        df = util.minmax_scaler_transform(df, self.scaler)

        valid_data = ListDataset(
            [{"start": df.index[0], "target": [df[col].values for col in df.columns]}],
            freq=self.freqGluon,
            one_dim_target=False,
        )

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=valid_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_samples=num_samples,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        forecasts = util.inverse_transform(forecasts, self.scaler)

        org_val = self.scaler.inverse_transform(tss[0].values)
        org_val = pd.DataFrame(org_val)
        org_val.index = tss[0].index
        tss[0] = org_val

        evaluator = MultivariateEvaluator(quantiles=quantiles, num_workers=num_workers)
        agg_metrics, item_metrics = evaluator(
            iter(tss), iter(forecasts), num_series=len(valid_data)
        )
        dates = (
            df.index[-self.prediction_length :].strftime("%Y-%m-%d %H:%M:%S").tolist()
        )
        return {
            "item_metrics": item_metrics,
            "agg_metrics": agg_metrics,
            "dates": dates,
            "values": self._quantiles(forecasts),
        }

    def _create_univariate_predictor(
        self,
        series,
        distr_output,
        trials,
        loss,
        tune_params,
        max_concurrent,
        eval_samples,
        use_ray,
        seed,
    ):
        train_data = ListDataset(
            [{"start": series.index[0], "target": series.values}], freq=self.freqGluon
        )

        valid_data = ListDataset(
            [
                {
                    "start": series.index[0],
                    "target": series.values[
                        : np.random.randint(
                            max(self.prediction_length, 3), series.size + 1
                        )
                    ],
                }
                for i in range(eval_samples)
            ],
            freq=self.freqGluon,
        )

        def build_estimator(params):
            estimator = None
            if params["name"] == DeepARParams.MODEL_NAME:
                estimator = self.univariate_model_params.get(params["name"]).build(
                    self.mx_ctx,
                    self.freqGluon,
                    self.prediction_length,
                    distr_output,
                    params,
                )
            elif params["name"] == FeedForwardParams.MODEL_NAME:
                estimator = self.univariate_model_params.get(params["name"]).build(
                    self.mx_ctx,
                    self.freqGluon,
                    self.prediction_length,
                    distr_output,
                    params,
                )
            return estimator

        def create_predictor(params):
            if params["name"] == ProphetParams.MODEL_NAME:
                predictor = self.univariate_model_params.get(
                    params["name"]
                ).build_predictor(self.freqGluon, self.prediction_length, params)
            elif params["name"] == RForecastParams.MODEL_NAME:
                predictor = self.univariate_model_params.get(
                    params["name"]
                ).build_predictor(self.freqGluon, self.prediction_length, params)
            else:
                estimator = build_estimator(params)
                predictor = estimator.train(training_data=train_data)
            return predictor

        def trainable(params):
            np.random.seed(params["seed"])
            predictor = create_predictor(params["model"])
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=valid_data, predictor=predictor, num_samples=100
            )
            # print(predictor)
            evaluator = Evaluator(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            agg_metrics, item_metrics = evaluator(
                ts_it, forecast_it, num_series=len(valid_data)
            )

            if not use_ray:
                return {loss: agg_metrics[loss]}

            tune.report(**{loss: agg_metrics[loss]})

        def objective_function(params):
            train_results = trainable(params)

            return {"loss": train_results[loss], "status": "ok"}

        models = []
        for name, p in self.univariate_model_params.items():
            if p is not None:
                params = p.tune_config()
                params["name"] = p.MODEL_NAME
                models.append(params)
        config = {
            "model": hp.choice("model", models),
            "seed": seed,
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
                trainable, search_alg=algo, num_samples=trials, **tune_params
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

        predictor = create_predictor(params["model"])

        time_taken = time.time() - start + time_total_s

        return predictor, time_taken

    def _score_univariate(self, series, num_samples, quantiles, num_workers):
        valid_data = ListDataset(
            [{"start": series.index[0], "target": series}], freq=self.freqGluon
        )

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=valid_data,  # test dataset
            predictor=self.predictor,  # predictor
            num_samples=num_samples,
        )
        forecasts = list(forecast_it)
        evaluator = Evaluator(quantiles=quantiles, num_workers=num_workers)
        agg_metrics, item_metrics = evaluator(
            ts_it, iter(forecasts), num_series=len(valid_data)
        )
        dates = (
            series.index[-self.prediction_length :]
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        return {
            "dates": dates,
            "values": self._quantiles(forecasts),
            "agg_metrics": agg_metrics,
            "item_metrics": item_metrics,
        }
