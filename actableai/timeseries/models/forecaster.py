import pandas as pd

from actableai.timeseries.models import AAITimeSeriesMultiTargetModel
from actableai.timeseries.exceptions import UntrainedModelException
from actableai.timeseries.util import handle_datetime_column, find_freq


class AAITimeSeriesForecaster:
    """
    TODO write documentation
    """

    def __init__(
        self,
        date_column,
        target_columns,
        prediction_length,
        group_by=None,
        real_static_feature=None,
        cat_static_feature=None,
        real_dynamic_feature_columns=None,
        cat_dynamic_feature_columns=None,
    ):
        """
        TODO write documentation
        """
        self.date_column = date_column
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.group_by = group_by
        self.real_static_feature_dict = real_static_feature
        self.cat_static_feature_dict = cat_static_feature
        self.real_dynamic_feature_columns = real_dynamic_feature_columns
        self.cat_dynamic_feature_columns = cat_dynamic_feature_columns

        if self.real_static_feature_dict is None:
            self.real_static_feature_dict = {}
        if self.cat_static_feature_dict is None:
            self.cat_static_feature_dict = {}
        if self.real_dynamic_feature_columns is None:
            self.real_dynamic_feature_columns = []
        if self.cat_dynamic_feature_columns is None:
            self.cat_dynamic_feature_columns = []
        if self.group_by is None:
            self.group_by = []

        if isinstance(self.real_static_feature_dict, list):
            self.real_static_feature_dict = {"data": self.real_static_feature_dict}
        if isinstance(self.cat_static_feature_dict, list):
            self.cat_static_feature_dict = {"data": self.cat_static_feature_dict}

        self.model = None

    @staticmethod
    def pre_process_data(df, date_column, target_columns, group_by=None, inplace=True):
        """
        TODO write documentation
        """
        if group_by is None:
            group_by = []

        df_dict = {}
        group_dict = {}
        freq_dict = {}

        # Create groups
        if len(group_by) > 0:
            for group_index, (group, grouped_df) in enumerate(df.groupby(group_by)):
                if len(group_by) == 1:
                    group = (group,)

                group_dict[group] = group_index
                df_dict[group] = grouped_df.reset_index(drop=True)
        else:
            df_dict["data"] = df

        # Process groups
        for group in df_dict.keys():
            # Handle datetime
            pd_date, _ = handle_datetime_column(df_dict[group][date_column])

            # Find frequency
            freq = find_freq(pd_date)
            freq_dict[group] = freq

            if not inplace:
                df_dict[group] = df_dict[group].copy()

            # Sort dataframe
            df_dict[group].index = pd_date
            df_dict[group].name = date_column
            df_dict[group].sort_index(inplace=True)

        return df_dict, group_dict, freq_dict

    def fit(
        self,
        model_params,
        mx_ctx,
        torch_device,
        *,
        df=None,
        df_dict=None,
        freq=None,
        group_dict=None,
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
        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")
        if df is None and freq is None:
            raise Exception("freq cannot be None if df is None")
        if df is None and group_dict is None:
            raise Exception("group_dict cannot be None if df is None")

        if df_dict is None:
            df_dict, group_dict, freq_dict = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                group_by=self.group_by,
                inplace=False,
            )
            first_group = list(df_dict.keys())[0]
            freq = freq_dict[first_group]

        self.model = AAITimeSeriesMultiTargetModel(
            target_columns=self.target_columns,
            prediction_length=self.prediction_length,
            freq=freq,
            group_dict=group_dict,
            real_static_feature_dict=self.real_static_feature_dict,
            cat_static_feature_dict=self.cat_static_feature_dict,
            real_dynamic_feature_columns=self.real_dynamic_feature_columns,
            cat_dynamic_feature_columns=self.cat_dynamic_feature_columns,
        )
        # TODO try multivariate models
        return self.model.fit(
            df_dict=df_dict,
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

    def refit(self, *, df=None, df_dict=None):
        """
        TODO write documentation
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df_dict is None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                group_by=self.group_by,
                inplace=False,
            )

        self.model.refit(df_dict)

    def score(
        self,
        *,
        df=None,
        df_dict=None,
        num_samples=100,
        quantiles=[0.05, 0.5, 0.95],
        num_workers=None,
    ):
        """
        TODO write documentation
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df_dict is None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict, df_item_metrics_dict, df_agg_metrics = self.model.score(
            df_dict,
            num_samples=num_samples,
            quantiles=quantiles,
            num_workers=num_workers,
        )

        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        df_item_metrics = pd.DataFrame()
        for group, df_group in df_item_metrics_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_item_metrics = pd.concat([df_item_metrics, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)
            df_item_metrics[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )
        df_item_metrics = df_item_metrics.drop(columns="_group")

        return df_predictions, df_item_metrics, df_agg_metrics

    def predict(self, *, df=None, df_dict=None, quantiles=[0.05, 0.5, 0.95]):
        """
        TODO write documentation
        """
        if self.model is None:
            raise UntrainedModelException()

        if df is None and df_dict is None:
            raise Exception("df or df_dict must be provided")
        if df is not None and df_dict is not None:
            raise Exception("df or df_dict must be provided")

        if df_dict is None:
            df_dict, _, _ = self.pre_process_data(
                df=df,
                date_column=self.date_column,
                target_columns=self.target_columns,
                group_by=self.group_by,
                inplace=False,
            )

        df_predictions_dict = self.model.predict(df_dict, quantiles=quantiles)

        df_predictions = pd.DataFrame()
        for group, df_group in df_predictions_dict.items():
            df_group["_group"] = [group] * len(df_group)
            df_predictions = pd.concat([df_predictions, df_group], ignore_index=True)

        for group_index, group in enumerate(self.group_by):
            f_group_values = lambda group_values: group_values[group_index]
            df_predictions[group] = df_predictions["_group"].apply(f_group_values)

        df_predictions = df_predictions.rename(columns={"date": self.date_column}).drop(
            columns="_group"
        )

        return df_predictions
