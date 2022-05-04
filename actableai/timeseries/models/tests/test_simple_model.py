import pytest
import ray
import numpy as np
import pandas as pd
import mxnet as mx

from actableai.timeseries.models import params
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.models import AAITimeSeriesSimpleModel
from actableai.utils.testing import init_ray, generate_forecast_df_dict


@pytest.fixture(scope="module")
def mx_ctx():
    yield mx.cpu()

class TestAAITimeSeriesSimpleModel:
    def _fit_predict_model(
        self,
        mx_ctx,
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
        use_ray=False,
    ):
        model = AAITimeSeriesSimpleModel(
            target_columns=target_columns,
            prediction_length=prediction_length,
            freq=freq,
            group_dict=group_dict,
            real_static_feature_dict=real_static_feature_dict,
            cat_static_feature_dict=cat_static_feature_dict,
            real_dynamic_feature_columns=real_dynamic_feature_columns,
            cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        )

        ray_shutdown = False
        if use_ray and not ray.is_initialized():
            init_ray()
            ray_shutdown = True

        if df_train_dict is not None:
            model.fit(
                df_dict=df_train_dict,
                model_params=model_params,
                mx_ctx=mx_ctx,
                loss="mean_wQuantileLoss",
                trials=trials,
                max_concurrent=4 if not use_ray else None,
                use_ray=use_ray,
                tune_samples=3,
                sampling_method="random",
                random_state=0,
                ray_tune_kwargs={
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0,
                    },
                    "raise_on_failed_trial": False,
                    "max_concurrent_trials": 1 if use_ray else None,
                },
                verbose=3,
            )

        if ray_shutdown:
            ray.shutdown()

        validations = None
        if df_valid_dict is not None:
            df_val_predictions_dict, df_item_metrics_dict, df_agg_metrics = model.score(
                df_valid_dict
            )
            validations = {
                "predictions": df_val_predictions_dict,
                "item_metrics": df_item_metrics_dict,
                "agg_metrics": df_agg_metrics,
            }

            if df_test_dict is not None:
                model.refit(df_valid_dict)

        predictions = None
        if df_test_dict is not None:
            df_predictions_dict = model.predict(df_test_dict)
            predictions = {"predictions": df_predictions_dict}

        return validations, predictions

    @pytest.mark.parametrize("n_groups", [1, 2])
    @pytest.mark.parametrize("use_features", [True, False])
    @pytest.mark.parametrize("freq", ["T"])
    @pytest.mark.parametrize(
        "univariate,model_type",
        [
            [True, "prophet"],
            [True, "r_forecast"],
            [True, "deep_ar"],
            [True, "feed_forward"],
            [True, "tree_predictor_qrx"],
            [True, "tree_predictor_quantile_regression"],
            [False, "deep_var"],
        ],
    )
    def test_simple_model(
        self,
        np_rng,
        mx_ctx,
        n_groups,
        use_features,
        freq,
        univariate,
        model_type,
    ):
        n_targets = 1 if univariate else 2
        (
            df_dict,
            target_columns,
            real_dynamic_feature_columns,
            cat_dynamic_feature_columns,
        ) = generate_forecast_df_dict(
            np_rng,
            n_groups,
            n_targets=n_targets,
            freq=freq,
            n_real_features=np_rng.integers(1, 10) if use_features else 0,
            n_cat_features=np_rng.integers(1, 10) if use_features else 0,
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
                group: group_index for group_index, group in enumerate(df_dict.keys())
            }

        df_train_dict = {}
        df_valid_dict = {}
        df_test_dict = {}
        for group in df_dict.keys():
            last_valid_index = (
                -prediction_length if use_features else len(df_dict[group])
            )

            df_train_dict[group] = df_dict[group].iloc[
                : last_valid_index - prediction_length
            ]
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
                use_feat_dynamic_real=use_features,
            )
        elif model_type == "feed_forward":
            model_param = params.FeedForwardParams(
                hidden_layer_1_size=1, epochs=2, context_length=None
            )
        elif model_type == "tree_predictor_qrx":
            model_param = params.TreePredictorParams(
                use_feat_dynamic_real=use_features,
                use_feat_dynamic_cat=use_features,
                context_length=None,
                method="QRX",
            )
        elif model_type == "tree_predictor_quantile_regression":
            model_param = params.TreePredictorParams(
                use_feat_dynamic_real=use_features,
                use_feat_dynamic_cat=use_features,
                context_length=None,
                method="QuantileRegression",
            )
        elif model_type == "deep_var":
            model_param = params.DeepVARParams(
                epochs=2, num_layers=1, num_cells=1, context_length=None
            )

        validations, predictions = self._fit_predict_model(
            mx_ctx,
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
            group_dict,
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions

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
            assert "0.05" in df_predictions.columns
            assert "0.5" in df_predictions.columns
            assert "0.95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (
                df_predictions.groupby("date").first().index
                == df_valid_dict[group].index[-prediction_length:]
            ).all()

            df_item_metrics = df_item_metrics_dict[group]
            assert len(df_item_metrics) == n_targets
            assert "target" in df_item_metrics.columns
            assert "MAPE" in df_item_metrics.columns
            assert "MASE" in df_item_metrics.columns
            assert "RMSE" in df_item_metrics.columns
            assert "sMAPE" in df_item_metrics.columns
            assert (
                not df_item_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]]
                .isna()
                .any(axis=None)
            )

        df_agg_metrics = validations["agg_metrics"]
        assert len(df_agg_metrics) == n_targets
        assert "target" in df_agg_metrics.columns
        assert "MAPE" in df_agg_metrics.columns
        assert "MASE" in df_agg_metrics.columns
        assert "RMSE" in df_agg_metrics.columns
        assert "sMAPE" in df_agg_metrics.columns
        assert (
            not df_agg_metrics[["MAPE", "MASE", "RMSE", "sMAPE"]].isna().any(axis=None)
        )

        df_predictions_dict = predictions["predictions"]
        for group in df_predictions_dict.keys():
            assert group in group_list

        for group in group_list:
            if (
                len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns)
                <= 0
            ):
                future_dates = pd.date_range(
                    df_test_dict[group].index[-1],
                    periods=prediction_length + 1,
                    freq=freq,
                )[1:]
            else:
                future_dates = df_test_dict[group].index[-prediction_length:]

            df_predictions = df_predictions_dict[group]
            assert "target" in df_predictions.columns
            assert "date" in df_predictions.columns
            assert "0.05" in df_predictions.columns
            assert "0.5" in df_predictions.columns
            assert "0.95" in df_predictions.columns
            assert len(df_predictions) == prediction_length * n_targets
            assert (df_predictions.groupby("date").first().index == future_dates).all()

    @pytest.mark.parametrize("freq", ["T"])
    @pytest.mark.parametrize("use_ray", [True, False])
    def test_hyperopt(self, np_rng, mx_ctx, use_ray, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng, n_groups=1, n_targets=1, freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        df_train_dict = {}
        df_valid_dict = {}
        df_test_dict = {}
        for group in df_dict.keys():
            last_valid_index = len(df_dict[group])

            df_train_dict[group] = df_dict[group].iloc[
                : last_valid_index - prediction_length
            ]
            df_valid_dict[group] = df_dict[group].iloc[:last_valid_index]
            df_test_dict[group] = df_dict[group]

        model_params = [params.ConstantValueParams()]

        validations, predictions = self._fit_predict_model(
            mx_ctx,
            prediction_length,
            model_params,
            freq,
            target_columns,
            df_train_dict,
            df_valid_dict,
            df_test_dict,
            trials=10,
            use_ray=use_ray,
        )

        assert validations is not None
        assert "predictions" in validations
        assert "item_metrics" in validations
        assert "agg_metrics" in validations
        assert predictions is not None
        assert "predictions" in predictions

    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_score(self, np_rng, mx_ctx, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng, n_groups=1, n_targets=1, freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
            _, _ = self._fit_predict_model(
                mx_ctx,
                prediction_length,
                model_params,
                freq,
                target_columns,
                df_train_dict=None,
                df_valid_dict=df_dict,
                trials=1,
                use_ray=False,
            )

    @pytest.mark.parametrize("freq", ["T"])
    def test_not_trained_predict(self, np_rng, mx_ctx, freq):
        df_dict, target_columns, _, _ = generate_forecast_df_dict(
            np_rng, n_groups=1, n_targets=1, freq=freq
        )
        prediction_length = np_rng.integers(1, 3)

        model_params = [params.ConstantValueParams()]

        with pytest.raises(UntrainedModelException):
            _, _ = self._fit_predict_model(
                mx_ctx,
                prediction_length,
                model_params,
                freq,
                target_columns,
                df_train_dict=None,
                df_test_dict=df_dict,
                trials=1,
                use_ray=False,
            )
