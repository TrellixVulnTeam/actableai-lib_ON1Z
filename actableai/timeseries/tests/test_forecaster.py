import pytest
import ray
import torch
import numpy as np
import pandas as pd
import mxnet as mx

from actableai.timeseries import params
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.forecaster import AAITimeSeriesForecaster
from actableai.utils.testing import init_ray, generate_forecast_df_dict


@pytest.fixture(scope="module")
def mx_ctx():
    yield mx.cpu()


@pytest.fixture(scope="module")
def torch_device():
    yield torch.device("cpu")


class TestAAITimeSeriesForecaster:
    def _fit_predict_model(self,
                           mx_ctx,
                           torch_device,
                           prediction_length,
                           model_params,
                           freq,
                           target_columns,
                           df_train_dict=None,
                           df_valid_dict=None,
                           df_test_dict=None,
                           real_static_feature_dict=None,
                           cat_static_feature_dict=None,
                           real_dynamic_feature_columns=None,
                           cat_dynamic_feature_columns=None,
                           group_dict=None,
                           trials=1,
                           use_ray=False):
        model = AAITimeSeriesForecaster(
            prediction_length,
            mx_ctx,
            torch_device,
            model_params=model_params
        )

        ray_shutdown = False
        if use_ray and not ray.is_initialized():
            init_ray()
            ray_shutdown = True

        if df_train_dict is not None:
            model.fit(
                df_train_dict,
                freq,
                target_columns,
                real_static_feature_dict=real_static_feature_dict,
                cat_static_feature_dict=cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=cat_dynamic_feature_columns,
                group_dict=group_dict,
                trials=trials,
                loss="mean_wQuantileLoss",
                tune_params={
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0,
                    },
                    "raise_on_failed_trial": False,
                    "max_concurrent_trials": 1 if use_ray else None
                },
                max_concurrent=4 if not use_ray else None,
                tune_samples=3,
                use_ray=use_ray
            )

        if ray_shutdown:
            ray.shutdown()

        validations = None
        if df_valid_dict is not None:
            validations = model.score(df_valid_dict)

            if df_test_dict is not None:
                model.refit(df_valid_dict)

        predictions = None
        if df_test_dict is not None:
            predictions = model.predict(df_test_dict)

        return validations, predictions


    @pytest.mark.parametrize("n_targets", [1, 2])
    @pytest.mark.parametrize("n_groups", [1, 2])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("freq", ["T"])
    @pytest.mark.parametrize("model_type", [
        "prophet",
        "r_forecast",
        "deep_ar",
        "feed_forward",
        "tree_predictor",
    ])
    def test_simple_model(self,
                          np_rng,
                          mx_ctx,
                          torch_device,
                          n_targets,
                          n_groups,
                          use_features,
                          freq,
                          model_type):
        df_dict, target_columns, real_dynamic_feature_columns, cat_dynamic_feature_columns = generate_forecast_df_dict(
            np_rng,
            n_groups,
            n_targets=n_targets,
            freq=freq,
            n_real_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_features=np_rng.integers(1, 10) if use_features else 0
        )
        prediction_length = np_rng.integers(1, 3)

        real_static_feature_dict = {}
        cat_static_feature_dict = {}
        if use_features:
            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                real_static_feature_dict[group] = np_rng.standard_normal(n_features)

            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                cat_static_feature_dict[group] = np_rng.integers(1, 10, n_features)

        group_dict = None
        if n_groups > 1:
            group_dict = {
                group: group_index
                for group_index, group in enumerate(df_dict.keys())
            }

        df_train_dict = {}
        df_valid_dict = {}
        df_test_dict = {}
        for group in df_dict.keys():
            last_valid_index = -prediction_length if use_features else len(df_dict[group])

            df_train_dict[group] = df_dict[group].iloc[:last_valid_index - prediction_length]
            df_valid_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_test_dict[group] = df_dict[group]

        model_param = None
        if model_type == "prophet":
            model_param = params.ProphetParams()
        elif model_type == "r_forecast":
            model_param = params.RForecastParams()
        elif model_type == "deep_ar":
            model_param = params.DeepARParams(
                num_cells=1,
                num_layers=1,
                epochs=2,
                context_length=None,
                use_feat_dynamic_real=use_features
            )
        elif model_type == "feed_forward":
            model_param = params.FeedForwardParams(
                hidden_layer_1_size=1,
                epochs=2,
                context_length=None
            )
        elif model_type == "tree_predictor":
            model_param = params.TreePredictorParams(
                use_feat_dynamic_real=use_features,
                use_feat_dynamic_cat=use_features,
                context_length=None
            )

        validations, predictions = self._fit_predict_model(
            mx_ctx,
            torch_device,
            prediction_length,
            [model_param],
            freq,
            target_columns,
            df_train_dict,
            df_valid_dict,
            df_test_dict,
            real_static_feature_dict,
            cat_static_feature_dict,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
            group_dict
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions


    @pytest.mark.parametrize("freq", ["T"])
    @pytest.mark.parametrize("use_ray", [True, False])
    def test_hyperopt(self, np_rng, mx_ctx, torch_device, use_ray, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng,
            n_groups=1,
            n_targets=1,
            freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        df_train_dict = {}
        df_valid_dict = {}
        df_test_dict = {}
        for group in df_dict.keys():
            last_valid_index = len(df_dict[group])

            df_train_dict[group] = df_dict[group].iloc[:last_valid_index - prediction_length]
            df_valid_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_test_dict[group] = df_dict[group]

        model_params = [params.ConstantValueParams()]

        validations, predictions = self._fit_predict_model(
            mx_ctx,
            torch_device,
            prediction_length,
            model_params,
            freq,
            target_columns,
            df_train_dict,
            df_valid_dict,
            df_test_dict,
            trials=10,
            use_ray=use_ray
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions


    @pytest.mark.parametrize("n_groups", [1, 5])
    @pytest.mark.parametrize("n_targets", [1, 5])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("freq", ["T"])
    def test_score(self,
                   np_rng,
                   mx_ctx,
                   torch_device,
                   n_groups,
                   n_targets,
                   use_features,
                   freq):
        df_dict, target_columns, real_dynamic_feature_columns, cat_dynamic_feature_columns = generate_forecast_df_dict(
            np_rng,
            n_groups,
            n_targets=n_targets,
            freq=freq,
            n_real_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_features=np_rng.integers(1, 10) if use_features else 0
        )
        group_list = list(df_dict.keys())
        prediction_length = np_rng.integers(1, 3)

        real_static_feature_dict = {}
        cat_static_feature_dict = {}
        if use_features:
            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                real_static_feature_dict[group] = np_rng.standard_normal(n_features)

            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                cat_static_feature_dict[group] = np_rng.integers(1, 10, n_features)

        group_dict = None
        if n_groups > 1:
            group_dict = {
                group: group_index
                for group_index, group in enumerate(df_dict.keys())
            }

        df_train_dict = {}
        df_valid_dict = {}
        for group in df_dict.keys():
            df_train_dict[group] = df_dict[group].iloc[:-prediction_length]
            df_valid_dict[group] = df_dict[group]

        model_params = [params.ConstantValueParams()]

        validations, _ = self._fit_predict_model(
            mx_ctx,
            torch_device,
            prediction_length,
            model_params,
            freq,
            target_columns,
            df_train_dict,
            df_valid_dict=df_valid_dict,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
            group_dict=group_dict
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations

        df_predictions_dict = validations["predictions"]
        for group in df_predictions_dict.keys():
            assert group in group_list

        df_item_metrics_dict = validations["item_metrics"]
        for group in df_item_metrics_dict.keys():
            assert group in group_list

        for group in group_list:
            df_predictions = df_predictions_dict[group]
            assert "target" in df_predictions.columns
            assert "date" in df_predictions.columns
            assert "q5" in df_predictions.columns
            assert "q50" in df_predictions.columns
            assert "q95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (df_predictions.groupby("date").first().index == df_valid_dict[group].index[-prediction_length:]).all()

            df_item_metrics = df_item_metrics_dict[group]
            assert "target" in df_item_metrics.columns
            assert len(df_item_metrics) == n_targets

        df_agg_metrics = validations["agg_metrics"]
        assert "target" in df_agg_metrics.columns
        assert len(df_agg_metrics) == n_targets


    @pytest.mark.parametrize("n_groups", [1, 5])
    @pytest.mark.parametrize("n_targets", [1, 5])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("freq", ["T"])
    def test_predict(self,
                     np_rng,
                     mx_ctx,
                     torch_device,
                     n_groups,
                     n_targets,
                     use_features,
                     freq):
        df_dict, target_columns, real_dynamic_feature_columns, cat_dynamic_feature_columns = generate_forecast_df_dict(
            np_rng,
            n_groups,
            n_targets=n_targets,
            freq=freq,
            n_real_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_features=np_rng.integers(1, 10) if use_features else 0
        )
        group_list = list(df_dict.keys())
        prediction_length = np_rng.integers(1, 3)

        real_static_feature_dict = {}
        cat_static_feature_dict = {}
        if use_features:
            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                real_static_feature_dict[group] = np_rng.standard_normal(n_features)

            for group in df_dict.keys():
                n_features = np_rng.integers(2, 10)
                cat_static_feature_dict[group] = np_rng.integers(1, 10, n_features)

        group_dict = None
        if n_groups > 1:
            group_dict = {
                group: group_index
                for group_index, group in enumerate(df_dict.keys())
            }

        df_train_dict = {}
        df_test_dict = {}
        for group in df_dict.keys():
            last_valid_index = -prediction_length if use_features else len(df_dict[group])

            df_train_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_test_dict[group] = df_dict[group]

        model_params = [params.ConstantValueParams()]

        _, predictions = self._fit_predict_model(
            mx_ctx,
            torch_device,
            prediction_length,
            model_params,
            freq,
            target_columns,
            df_train_dict,
            df_test_dict=df_test_dict,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
            group_dict=group_dict
        )

        assert predictions is not None
        assert "predictions" in predictions

        df_predictions_dict = predictions["predictions"]
        for group in df_predictions_dict.keys():
            assert group in group_list

        for group in group_list:
            if len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) <= 0:
                future_dates = pd.date_range(
                    df_test_dict[group].index[-1],
                    periods=prediction_length + 1,
                    freq=freq
                )[1:]
            else:
                future_dates = df_test_dict[group].index[-prediction_length:]

            df_predictions = df_predictions_dict[group]
            assert "target" in df_predictions.columns
            assert "date" in df_predictions.columns
            assert "q5" in df_predictions.columns
            assert "q50" in df_predictions.columns
            assert "q95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (df_predictions.groupby("date").first().index == future_dates).all()


    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_score(self, np_rng, mx_ctx, torch_device, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng,
            n_groups=1,
            n_targets=1,
            freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
             _, _ = self._fit_predict_model(
                 mx_ctx,
                 torch_device,
                 prediction_length,
                 model_params,
                 freq,
                 target_columns,
                 df_train_dict=None,
                 df_valid_dict=df_dict,
                 trials=1,
                 use_ray=False
             )


    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_predict(self, np_rng, mx_ctx, torch_device, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng,
            n_groups=1,
            n_targets=1,
            freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
             _, _ = self._fit_predict_model(
                 mx_ctx,
                 torch_device,
                 prediction_length,
                 model_params,
                 freq,
                 target_columns,
                 df_train_dict=None,
                 df_test_dict=df_dict,
                 trials=1,
                 use_ray=False
             )

