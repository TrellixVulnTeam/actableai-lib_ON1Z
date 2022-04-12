from copy import deepcopy

import pandas as pd

from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.resources.predict.features.common import (
    extract_dataset_features,
    all_dataset_features,
)
from actableai.utils.resources.predict.features.method import MethodFeaturesExtractor


class DataImputationFeaturesExtractor(MethodFeaturesExtractor):
    """
    Data Imputation Features Extractor
    """

    # Dictionary used to filter the features to extract depending on the resource to predict
    resource_predicted_features_filter = {
        ResourcePredictorType.MAX_MEMORY: [*all_dataset_features],
        ResourcePredictorType.MAX_GPU_MEMORY: [*all_dataset_features],
    }

    def _filter_features(self, features: dict) -> dict:
        """
        Filter Data Imputation features

        Parameters
        ----------
        features:
            The features to filter

        Returns
        -------
        The filtered features
        """
        features_filter = self.resource_predicted_features_filter.get(
            self.resource_predicted, []
        )
        return {key: value for key, value in features.items() if key in features_filter}

    @staticmethod
    def _extract_all_features(arguments: dict) -> dict:
        """
        Extract all features for the Data Imputation task

        Parameters
        ----------
        arguments:
            The arguments used to call the Data Imputation task and to extract the features from

        Returns
        -------
        The extracted features
        """
        features = {
            **extract_dataset_features(arguments.get("df", pd.DataFrame())),
            **deepcopy(arguments),
        }

        features.pop("df", None)

        return features
