import shap
import numpy as np
import pandas as pd
from typing import Union, List, Any, Dict

from autogluon.tabular.predictor import TabularPredictor

from actableai.utils.autogluon import (
    transform_features,
    get_feature_links,
    get_final_features,
)


class AutoGluonShapTreeExplainer:
    """AutoGluon Shap Tree Explainer Wrapper."""

    def __init__(self, autogluon_predictor: TabularPredictor):
        """AutoGluonShapTreeExplainer Constructor.

        Args:
            autogluon_predictor: The AutoGluon predictor to use for explanations.
        """
        self.autogluon_predictor = autogluon_predictor
        self.model_name = self.autogluon_predictor.get_model_best()
        self.autogluon_model = self.autogluon_predictor._trainer.load_model(
            self.model_name
        )

        self.explainer = shap.TreeExplainer(self.autogluon_model.model)

    @staticmethod
    def is_predictor_compatible(autogluon_predictor: TabularPredictor) -> bool:
        """Check if an AutoGluon predictor is compatible with the Shap Tree Explainer.

        Args:
            autogluon_predictor: The AutoGluon predictor to check.

        Returns:
            True if the predictor is compatible.
        """
        model_name = autogluon_predictor.get_model_best()
        try:
            shap.TreeExplainer(
                autogluon_predictor._trainer.load_model(model_name).model
            )
        except:
            return False
        return True

    def shap_values(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Compute shap values.

        Args:
            data: The data to create the explanations for.
            args: The arguments to pass to the underlying shap_values function.
            kwargs: The named arguments to pass to the underlying shap_values function.

        Returns:
            The shap values.
        """
        # Transform features to use the model directly
        df_transformed_data, transformed_data = transform_features(
            self.autogluon_predictor, self.model_name, data
        )

        # Extract features information
        feature_links = get_feature_links(self.autogluon_predictor, self.model_name)
        final_features = get_final_features(self.autogluon_predictor, self.model_name)

        if len(final_features) != transformed_data.shape[1]:
            raise Exception("Model not supported")

        # Compute shap values
        shap_values = self.explainer.shap_values(transformed_data, *args, **kwargs)
        shap_values = np.array(shap_values)

        # Select correct shap values in case the model returned probabilities
        if len(shap_values.shape) == 3:
            shap_values = np.array(shap_values)

            pred = self.autogluon_model.predict_proba(df_transformed_data)

            if len(pred.shape) == 1:
                pred = np.concatenate(
                    [(1 - pred).reshape(-1, 1), pred.reshape(-1, 1)], axis=1
                )

            row_index, column_index = np.meshgrid(
                np.arange(shap_values.shape[1]), np.arange(shap_values.shape[2])
            )
            shap_values = shap_values[
                pred.argmax(axis=1), row_index, column_index
            ].transpose(1, 0)

        df_shap_values = pd.DataFrame(shap_values, columns=final_features)

        df_final_shap_values = pd.DataFrame(
            0, index=np.arange(len(data)), columns=data.columns
        )
        # Compute final shap values (grouping features)
        for column in df_final_shap_values.columns:
            column_links = feature_links.get(column, [])

            if len(column_links) <= 0:
                continue

            df_final_shap_values[column] = df_shap_values[column_links].sum(axis=1)

        return df_final_shap_values
