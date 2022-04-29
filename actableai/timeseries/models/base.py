from abc import ABC, abstractmethod

from actableai.timeseries.util import find_gluonts_freq
from actableai.timeseries.exceptions import InvalidFrequencyException


class AAITimeSeriesBaseModel(ABC):
    """
    Time Series Model interface
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
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.freq = freq

        if self.freq is None:
            raise InvalidFrequencyException()

        self.group_dict = group_dict
        self.real_static_feature_dict = real_static_feature_dict
        self.cat_static_feature_dict = cat_static_feature_dict
        self.real_dynamic_feature_columns = real_dynamic_feature_columns
        self.cat_dynamic_feature_columns = cat_dynamic_feature_columns

        if self.group_dict is None:
            self.group_dict = {}
        if self.real_static_feature_dict is None:
            self.real_static_feature_dict = {}
        if self.cat_static_feature_dict is None:
            self.cat_static_feature_dict = {}
        if self.real_dynamic_feature_columns is None:
            self.real_dynamic_feature_columns = []
        if self.cat_dynamic_feature_columns is None:
            self.cat_dynamic_feature_columns = []

        self.freq_gluon = find_gluonts_freq(self.freq)

        self.has_dynamic_features = (
            len(self.real_dynamic_feature_columns)
            + len(self.cat_dynamic_feature_columns)
            > 0
        )

    @abstractmethod
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
        pass

    @abstractmethod
    def refit(self, df_dict):
        """
        TODO write documentation
        """
        pass

    @abstractmethod
    def score(
        self, df_dict, num_samples=100, quantiles=[0.05, 0.5, 0.95], num_workers=None
    ):
        """
        TODO write documentation
        """
        pass

    @abstractmethod
    def predict(self, df_dict):
        """
        TODO write documentation
        """
        pass
