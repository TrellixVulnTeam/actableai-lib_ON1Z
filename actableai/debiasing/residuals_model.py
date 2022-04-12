import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, check_consistent_length
from pandas.api.types import is_numeric_dtype

from autogluon.core.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

from actableai.utils.multilabel_predictor import MultilabelPredictor


class ResidualsModel:
    """
    TODO write documentation
    """

    residuals_file = "residuals_predictor.pkl"

    def __init__(self, path, biased_groups, debiased_features):
        """
        TODO write documentation
        """
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.biased_groups = biased_groups
        self.debiased_features = debiased_features

        self.model = MultilabelPredictor(
            labels=self.debiased_features,
            path=self.path,
            consider_labels_correlation=False,
            problem_types=["regression"] * len(self.debiased_features),
        )

    def _get_model(self):
        """
        TODO write documentation
        """
        model = self.model
        if isinstance(model, str):
            return MultilabelPredictor.load(path=model)
        return model

    def _preprocess(self, X):
        """
        TODO write documentation
        """
        X_clean = X.copy()

        for column in X_clean.columns:
            if is_numeric_dtype(X_clean[column].dtype):
                continue

            # Replace NaN values in categorical columns
            if X_clean[column].isna().sum() > 0:
                if X_clean[column].dtype.name == "category":
                    X_clean[column] = X_clean[column].cat.add_categories(["NaN"])
                    X_clean[column] = X_clean[column].cat.rename_categories(
                        {
                            category: str(category)
                            for category in X_clean[column].cat.categories
                        }
                    )

                X_clean[column] = X_clean[column].fillna("NaN")

            # Transform categories into numbers
            if X_clean[column].dtype.name == "category":
                X_clean[column] = X_clean[column].cat.codes

        return X_clean

    def fit(self, X, hyperparameters=None, presets=None):
        """
        TODO write documentation
        """
        X = self._preprocess(X)
        model = self._get_model()

        model.fit(
            X[self.biased_groups + self.debiased_features],
            hyperparameters=hyperparameters,
            presets=presets,
            refit_full="all",
            keep_only_best=True,
        )
        pd.set_option("chained_assignment", "warn")

        self.save()

    def predict(self, X):
        """
        TODO write documentation
        """
        X = self._preprocess(X)

        model = self._get_model()
        pred_proba = model.predict_proba(X[self.biased_groups])

        df_residuals = pd.DataFrame()
        residuals_features_dict = {}
        categorical_residuals_count = 0

        model = self._get_model()

        for debiased_feature in self.debiased_features:
            residuals_feature = f"{debiased_feature}_residuals"
            problem_type = model.get_predictor(debiased_feature).problem_type

            nan_mask = X[debiased_feature].isna()

            if problem_type == "regression":
                df_residuals[residuals_feature] = (
                    X[debiased_feature] - pred_proba[debiased_feature]
                )
                df_residuals[residuals_feature][nan_mask] = 0

                residuals_features_dict[residuals_feature] = debiased_feature
            else:
                for class_label in model.get_predictor(debiased_feature).class_labels:
                    class_residuals_feature = f"{residuals_feature}_{class_label}"
                    df_residuals[class_residuals_feature] = (
                        X[debiased_feature] == class_label
                    ).apply(lambda row: 1 if row else 0)
                    df_residuals[class_residuals_feature] -= pred_proba[
                        debiased_feature
                    ][class_label]

                    residuals_features_dict[class_residuals_feature] = debiased_feature
                    categorical_residuals_count += 1

        residuals_features_list = list(residuals_features_dict.keys())
        df_residuals[residuals_features_list] = df_residuals[
            residuals_features_list
        ].astype(float)

        return df_residuals, residuals_features_dict, categorical_residuals_count

    @staticmethod
    def _per_sample_log_loss(y_true, y_pred, eps=1e-15, labels=None):
        """
        TODO write documentation
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py#L2309
        Exactly the same function but not with the weighted sum at the end
        Actable : return loss
        SKLearn : return _weighted_sum(loss, sample_weight, normalize)
        """
        y_pred = check_array(y_pred, ensure_2d=False)
        check_consistent_length(y_pred, y_true)

        lb = LabelBinarizer()
        if labels is not None:
            lb.fit(labels)
        else:
            lb.fit(y_true)

        if len(lb.classes_) == 1:
            if labels is None:
                raise ValueError(
                    "y_true contains only one label ({0}). Please "
                    "provide the true labels explicitly through the "
                    "labels argument.".format(lb.classes_[0])
                )
            else:
                raise ValueError(
                    "The labels array needs to contain at least two "
                    "labels for log_loss, "
                    "got {0}.".format(lb.classes_)
                )

        transformed_labels = lb.transform(y_true)

        if transformed_labels.shape[1] == 1:
            transformed_labels = np.append(
                1 - transformed_labels, transformed_labels, axis=1
            )

        # Clipping
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # If y_pred is of single dimension, assume y_true to be binary
        # and then check.
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)

        # Check if dimensions are consistent.
        transformed_labels = check_array(transformed_labels)
        if len(lb.classes_) != y_pred.shape[1]:
            if labels is None:
                raise ValueError(
                    "y_true and y_pred contain different number of "
                    "classes {0}, {1}. Please provide the true "
                    "labels explicitly through the labels argument. "
                    "Classes found in "
                    "y_true: {2}".format(
                        transformed_labels.shape[1], y_pred.shape[1], lb.classes_
                    )
                )
            else:
                raise ValueError(
                    "The number of classes in labels is different "
                    "from that in y_pred. Classes found in "
                    "labels: {0}".format(lb.classes_)
                )

        # Renormalize
        y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
        loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

        return loss

    def persist_models(self):
        """
        TODO write documentation
        """
        if isinstance(self.model, str):
            self.model = self._get_model()
        self.model.persist_models()

    def unpersist_models(self):
        """
        TODO write documentation
        """
        if not isinstance(self.model, str):
            self.model.unpersist_models()
            self.model = self.model.path

    def save(self, path=None):
        """
        TODO write documentation
        """
        self.unpersist_models()

        if path is None:
            path = self.path

        file_path = path + self.residuals_file
        save_pkl.save(path=file_path, object=self)

        return path

    @classmethod
    def load(cls, path):
        """
        TOO write documentation
        """
        path = os.path.expanduser(path)
        if path[-1] != os.path.sep:
            path = path + os.path.sp

        model = load_pkl.load(path=path + cls.residuals_file)
        model.persist_models()

        return model
