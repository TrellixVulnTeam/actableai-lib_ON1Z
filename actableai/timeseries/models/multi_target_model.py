import pandas as pd

from actableai.timeseries.models.base import AAITimeSeriesBaseModel
from actableai.timeseries.models.simple_model import AAITimeSeriesSimpleModel
from actableai.timeseries.exceptions import UntrainedModelException


class AAITimeSeriesMultiTargetModel(AAITimeSeriesBaseModel):
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

        self.predictor_dict = {}

        self.shift_target_columns_dict = self._get_shift_target_columns()
        self.shift_target_columns = list(self.shift_target_columns_dict.values())

    def _get_shift_target_columns(self):
        """
        TODO write documentation
        """
        return {
            target_column: f"_{target_column}_shift"
            for target_column in self.target_columns
        }

    def _pre_process_data(self, df_dict, keep_future=True):
        """
        TODO write documentation
        """
        if len(self.target_columns) <= 1:
            return df_dict

        df_dict_new = {}
        for group in df_dict.keys():
            # Create the shifted dataframe
            df_shift = df_dict[group]
            if keep_future and self.has_dynamic_features:
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
    ):
        """
        TODO write documentation
        """
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        total_time = 0

        for target_column in self.target_columns:
            shift_target_column = self.shift_target_columns_dict[target_column]

            real_dynamic_feature_columns = list(
                set(
                    self.real_dynamic_feature_columns + self.shift_target_columns
                ).difference({shift_target_column})
            )

            self.predictor_dict[target_column] = AAITimeSeriesSimpleModel(
                target_columns=[target_column],
                prediction_length=self.prediction_length,
                freq=self.freq,
                group_dict=self.group_dict,
                real_static_feature_dict=self.real_static_feature_dict,
                cat_static_feature_dict=self.cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
            )
            target_total_time = self.predictor_dict[target_column].fit(
                df_dict=df_dict_clean,
                model_params=model_params,
                mx_ctx=mx_ctx,
                torch_device=torch_device,
                loss=loss,
                trials=trials,
                max_concurrent=max_concurrent,
                use_ray=use_ray,
                tune_samples=tune_samples,
                sampling_method=sampling_method,
                random_state=random_state,
                ray_tune_kwargs=ray_tune_kwargs,
                verbose=verbose,
            )

            total_time += target_total_time

        return total_time

    def refit(self, df_dict):
        """
        TODO write documentation
        """
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()
            self.predictor_dict[target_column].refit(df_dict_clean)

    def score(
        self, df_dict, num_samples=100, quantiles=[0.05, 0.5, 0.95], num_workers=None
    ):
        """
        TODO write documentation
        """
        df_dict_clean = self._pre_process_data(df_dict, keep_future=False)

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}
        df_item_metrics_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}
        df_agg_metrics = pd.DataFrame()

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()

            (
                df_target_predictions_dict,
                df_target_item_metrics_dict,
                df_target_agg_metrics,
            ) = self.predictor_dict[target_column].score(
                df_dict_clean,
                num_samples=num_samples,
                quantiles=quantiles,
                num_workers=num_workers,
            )

            for group in df_dict_clean.keys():
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )
                df_item_metrics_dict[group] = pd.concat(
                    [df_item_metrics_dict[group], df_target_item_metrics_dict[group]],
                    ignore_index=True,
                )
            df_agg_metrics = pd.concat(
                [df_agg_metrics, df_target_agg_metrics], ignore_index=True
            )

        return df_predictions_dict, df_item_metrics_dict, df_agg_metrics

    def predict(self, df_dict, quantiles=[0.05, 0.5, 0.95]):
        """
        TODO write documentation
        """
        df_dict_clean = self._pre_process_data(df_dict, keep_future=True)

        df_predictions_dict = {group: pd.DataFrame() for group in df_dict_clean.keys()}

        for target_column in self.target_columns:
            if target_column not in self.predictor_dict:
                raise UntrainedModelException()

            df_target_predictions_dict = self.predictor_dict[target_column].predict(
                df_dict_clean, quantiles=quantiles
            )

            for group in df_dict_clean.keys():
                df_predictions_dict[group] = pd.concat(
                    [df_predictions_dict[group], df_target_predictions_dict[group]],
                    ignore_index=True,
                )

        return df_predictions_dict
