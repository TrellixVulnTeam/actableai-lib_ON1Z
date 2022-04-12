from copy import deepcopy

import pandas as pd

from actableai.utils.resources.predict import ResourcePredictorType
from actableai.utils.resources.predict.features.common import (
    extract_dataset_features,
    all_dataset_features,
)
from actableai.utils.resources.predict.features.method import MethodFeaturesExtractor


class ClusteringFeaturesExtractor(MethodFeaturesExtractor):
    """
    Clustering Features Extractor
    """

    # Dictionary used to filter the features to extract depending on the resource to predict
    resource_predicted_features_filter = {
        ResourcePredictorType.MAX_MEMORY: [*all_dataset_features],
        ResourcePredictorType.MAX_GPU_MEMORY: [*all_dataset_features],
    }

    def _filter_features(self, features: dict) -> dict:
        """
        Filter Clustering features

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
        Extract all features for the Clustering task

        Parameters
        ----------
        arguments:
            The arguments used to call the Clustering task and to extract the features from

        Returns
        -------
        The extracted features
        """

        arguments.pop("explainer_task_params", None)
        features = {
            **extract_dataset_features(arguments.get("df", pd.DataFrame())),
            **deepcopy(arguments),
        }

        features.pop("df", None)

        return features


class DECAnchorClusteringFeaturesExtractor(MethodFeaturesExtractor):
    """
    DEC Anchor Clustering Features Extractor
    """

    # Dictionary used to filter the features to extract depending on the resource to predict
    resource_predicted_features_filter = {
        ResourcePredictorType.MAX_MEMORY: [*all_dataset_features],
        ResourcePredictorType.MAX_GPU_MEMORY: [*all_dataset_features],
    }

    def _filter_features(self, features: dict) -> dict:
        """
        Filter Clustering features

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
        Extract all features for the Clustering task

        Parameters
        ----------
        arguments:
            The arguments used to call the Clustering task and to extract the features from

        Returns
        -------
        The extracted features
        """

        arguments.pop("explainer", None)

        features = {
            **extract_dataset_features(arguments.get("df", pd.DataFrame())),
            **deepcopy(arguments),
        }

        features.pop("df", None)

        return features
