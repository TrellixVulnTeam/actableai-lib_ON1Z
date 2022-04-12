import os
from uuid import uuid4
from copy import deepcopy

import pandas as pd

from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.core.constants import REGRESSION, QUANTILE
from autogluon.tabular import TabularPredictor

from actableai.utils import memory_efficient_hyperparameters, debiasing_hyperparameters
from actableai.debiasing.residuals_model import ResidualsModel


class DebiasingModel(AbstractModel):
    """
    TODO write documentation
    """

    def __init__(self, **kwargs):
        """
        TODO write documentation
        """
        super().__init__(**kwargs)

        self.models_fit = False

        # Initialize Parameters
        self.drop_duplicates = self.params_aux["drop_duplicates"]
        self.label = self.params_aux["label"]
        self.features = self.params_aux["features"]
        self.biased_groups = self.params_aux["biased_groups"]
        self.debiased_features = self.params_aux["debiased_features"]
        self.non_residuals_features = list(
            set(self.features).difference(set(self.debiased_features))
        )

        # Initialize Models
        # Residuals Model is computing the residuals of the debiased_features
        self.hyperparameters_residuals = self.params_aux["hyperparameters_residuals"]
        self.presets_residuals = self.params_aux["presets_residuals"]
        self.residuals_model = ResidualsModel(
            path=os.path.join(self.path, "residuals"),
            biased_groups=self.biased_groups,
            debiased_features=self.debiased_features,
        )

        # Non-residuals Model is handling all the features which are not debiased
        # If there is no non-residuals features we do not use this model
        self.hyperparameters_non_residuals = self.params_aux[
            "hyperparameters_non_residuals"
        ]
        self.presets_non_residuals = self.params_aux["presets_non_residuals"]
        self.non_residuals_model = None
        if len(self.non_residuals_features) > 0:
            self.non_residuals_model = TabularPredictor(
                label=self.label,
                path=os.path.join(self.path, "non_residuals"),
                problem_type=self.problem_type,
            )

        # Final Model to handle Residuals + Non-residuals
        self.presets_final = self.params_aux["presets_final"]
        self.final_model = TabularPredictor(
            label=self.label,
            path=os.path.join(self.path, "final"),
            problem_type=self.problem_type,
        )

    def is_fit(self):
        """
        TODO write documentation
        """
        return self.models_fit

    def _get_default_auxiliary_params(self):
        """
        TODO write documentation
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()

        extra_auxiliary_params = {
            "drop_duplicates": True,
            "label": uuid4(),
            "features": [],
            "biased_groups": [],
            "debiased_features": [],
            "hyperparameters_residuals": "default",
            "presets_residuals": "medium_quality_faster_train",
            "hyperparameters_non_residuals": "default",
            "presets_non_residuals": "medium_quality_faster_train",
            "presets_final": "medium_quality_faster_train",
        }

        default_auxiliary_params.update(extra_auxiliary_params)

        return default_auxiliary_params

    def _fit_residuals(self, train_data):
        """
        TODO write documentation
        """
        # Prepare data
        train_data = train_data[self.biased_groups + self.debiased_features]

        # Train model
        residuals_model = self._get_residuals_model()
        residuals_model.fit(
            train_data,
            hyperparameters=self.hyperparameters_residuals,
            presets=self.presets_residuals,
        )

    def _fit_non_residuals(self, train_data, tuning_data=None):
        """
        TODO write documentation
        """
        if self.non_residuals_model is None:
            return

        # Prepare data
        train_data = train_data[self.non_residuals_features + [self.label]]
        if tuning_data is not None:
            tuning_data = tuning_data[self.non_residuals_features + [self.label]]

        # Train model
        non_residuals_model = self._get_non_residuals_model()
        non_residuals_model.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=self.hyperparameters_non_residuals,
            presets=self.presets_non_residuals,
        )

    def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """
        TODO write documentation
        """
        # Set up train_data
        train_data = X.copy()
        train_data[self.label] = y

        # Set up tuning data
        tuning_data = None
        if X_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.label] = y_val

        # Train sub models (can be done simulateneously)
        self._fit_residuals(train_data)
        self._fit_non_residuals(train_data, tuning_data)

        self._persist_residuals_model()
        self._persist_non_residuals_model()

        # Prepare training data
        final_train_data, categorical_residuals_count = self._predict_residuals(
            train_data
        )
        if self.non_residuals_model is not None:
            final_train_data = pd.concat(
                [final_train_data, self._predict_non_residuals(train_data)], axis=1
            )

            final_train_data.rename(
                columns={self.label: "non_residuals_pred"},
                inplace=True,
                errors="ignore",
            )

        final_train_data[self.label] = train_data[self.label]

        # Prepare tuning data
        final_tuning_data = None
        if tuning_data is not None:
            final_tuning_data, _ = self._predict_residuals(tuning_data)
            if self.non_residuals_model is not None:
                final_tuning_data = pd.concat(
                    [final_tuning_data, self._predict_non_residuals(tuning_data)],
                    axis=1,
                )

                final_tuning_data.rename(
                    columns={self.label: "non_residuals_pred"},
                    inplace=True,
                    errors="ignore",
                )

            final_tuning_data[self.label] = tuning_data[self.label]

        if self.drop_duplicates:
            final_train_data = final_train_data.drop_duplicates()

        hyperparameters_final = debiasing_hyperparameters()

        # Train final model
        self.final_model.fit(
            train_data=final_train_data,
            tuning_data=final_tuning_data,
            hyperparameters=hyperparameters_final,
            presets=self.presets_final,
        )

        self.models_fit = True

    def _predict_residuals(self, data):
        """
        TODO write documentation
        """
        # Prepare data
        data = data[self.biased_groups + self.debiased_features]

        # Predict
        residuals_model = self._get_residuals_model()
        (
            df_residuals,
            residuals_features_dict,
            categorical_residuals_count,
        ) = residuals_model.predict(data)

        residuals_features = list(residuals_features_dict.keys())
        return df_residuals[residuals_features], categorical_residuals_count

    def _predict_non_residuals(self, data):
        """
        TODO write documentation
        """
        if self.non_residuals_model is None:
            return None

        # Prepare data
        data = data[self.non_residuals_features]

        # Predict
        non_residuals_model = self._get_non_residuals_model()
        return non_residuals_model.predict_proba(data, as_multiclass=True)

    def _predict_proba(self, X, **kwargs):
        """
        TODO write documentation
        """
        # Prepare data
        final_data, _ = self._predict_residuals(X)
        if self.non_residuals_model is not None:
            final_data = pd.concat([final_data, self._predict_non_residuals(X)], axis=1)

            final_data.rename(
                columns={self.label: "non_residuals_pred"},
                inplace=True,
                errors="ignore",
            )

        # Predict
        final_model = self._get_final_model()
        if self.problem_type in [REGRESSION, QUANTILE]:
            return final_model.predict(final_data)

        final_pred_proba = final_model.predict_proba(final_data, as_pandas=False)
        return self._convert_proba_to_unified_form(final_pred_proba)

    def convert_to_template(self):
        """
        TODO write documentation
        """
        residuals_model = self.residuals_model
        non_residuals_model = self.non_residuals_model
        final_model = self.final_model

        self.non_residuals_model = None
        self.residuals_model = None
        self.final_model = None

        template = deepcopy(self)
        template.reset_metrics()

        self.residuals_model = residuals_model
        self.non_residuals_models = non_residuals_model
        self.final_model = final_model

        return template

    def _get_residuals_model(self):
        """
        TODO write documentation
        """
        residuals_model = self.residuals_model
        if isinstance(residuals_model, str):
            return ResidualsModel.load(path=residuals_model)
        return residuals_model

    def _get_non_residuals_model(self):
        """
        TODO write documentation
        """
        non_residuals_model = self.non_residuals_model
        if non_residuals_model is not None and isinstance(non_residuals_model, str):
            return TabularPredictor.load(path=non_residuals_model)
        return non_residuals_model

    def _get_final_model(self):
        """
        TODO write documentation
        """
        final_model = self.final_model
        if isinstance(final_model, str):
            return TabularPredictor.load(path=final_model)
        return final_model

    def _persist_residuals_model(self):
        """
        TODO write documentation
        """
        if isinstance(self.residuals_model, str):
            self.residuals_model = self._get_residuals_model()
        self.residuals_model.persist_models()

    def _persist_non_residuals_model(self):
        """
        TODO write documentation
        """
        if self.non_residuals_model is None:
            return

        if isinstance(self.non_residuals_model, str):
            self.non_residuals_model = self._get_non_residuals_model()
        self.non_residuals_model.persist_models()

    def _persist_final_model(self):
        """
        TODO write documentation
        """
        if isinstance(self.final_model, str):
            self.final_model = self._get_final_model()
        self.final_model.persist_models()

    def persist_models(self):
        """
        TODO write documentation
        """
        self._persist_residuals_model()
        self._persist_non_residuals_model()
        self._persist_final_model()

    def unpersist_models(self):
        """
        TODO write documentation
        """
        # Unpersist residuals model
        if not isinstance(self.residuals_model, str):
            self.residuals_model.unpersist_models()
            self.residuals_model = self.residuals_model.path

        # Unpersist non residuals model
        if self.non_residuals_model is not None and not isinstance(
            self.non_residuals_model, str
        ):
            self.non_residuals_model.unpersist_models()
            self.non_residuals_model = self.non_residuals_model.path

        # Unpersist final model
        if not isinstance(self.final_model, str):
            self.final_model.unpersist_models()
            self.final_model = self.final_model.path

    def save(self, path=None, verbose=True):
        """
        TODO write documentation
        """
        self.unpersist_models()
        return super().save(path=path, verbose=verbose)

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        TODO write documentation
        """
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        model.persist_models()
        return model
