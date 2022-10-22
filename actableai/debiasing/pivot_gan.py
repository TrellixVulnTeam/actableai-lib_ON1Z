import numpy as np
import pandas as pd
from typing import List, Optional

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display


class PivotGAN:
    def __init__(
        self,
        path: str,
        biased_features: List[str],
        label: str,
        lambdas: List[float],
        debiased_features: Optional[List[str]] = None,
        problem_type: str = "auto",
    ):
        """
        Debiasing models with GAN.

        Currently supports only binary targets, sensitive groups with categorical
        values.

        Args:
            path: where to store trained models
            biased_features: columns for features that need to be protected from biases
            debiased_features: columns that need to be debiased to prevent biases from
                protected groups
            label: predicted target column
            lambdas: weights of bias losses for categories in biases features. 1 is
                as important as target loss.
        """
        self.model_fit = False
        self.path = path
        self.label = label
        self.biased_features = biased_features
        self.debiased_features = debiased_features
        self.features = biased_features + (debiased_features or [])
        self.lambdas = lambdas
        self.problem_type = problem_type
        self._n_sensitive = len(self.biased_features)

        assert len(self.lambdas) == len(self.biased_features)

    def fit(self, X, ag_args_fit=None, verbose=0, validation_data=None):
        if ag_args_fit is None:
            ag_args_fit = {}
        epochs = ag_args_fit.get("epochs", 50)
        pretrain_epochs = ag_args_fit.get("pretrain_epochs", 10)
        batch_size = ag_args_fit.get("batch_size", 128)

        if self.debiased_features is None:
            self.debiased_features = X.columns[
                ~X.columns.isin([self.label] + self.biased_features)
            ].tolist()
            self.features = self.biased_features + self.debiased_features

        X = self._preprocess(X, is_train=True)

        self._init(X[self._transformed_debiased_features])
        self._pretrain(X, epochs=pretrain_epochs, verbose=verbose)

        x, y, z = (
            X[self._transformed_debiased_features].values,
            X[self.label].values,
            X[self._transformed_biased_features].values,
        )

        if validation_data is not None:
            validation_data = self._preprocess(validation_data, is_train=False)

        # class_weight_gen = [{0: 1.0, 1: 1.0}]
        # class_weight_dis = self._compute_class_weights(z)
        # class_weight_gan_gen = class_weight_gen + class_weight_dis
        self._fairness_metrics = pd.DataFrame()
        self._val_metrics = pd.DataFrame()
        for idx in range(epochs):
            if validation_data is not None:
                x_val, y_val, z_val = (
                    validation_data[self._transformed_debiased_features].values,
                    validation_data[self.label].values,
                    validation_data[self._transformed_biased_features].values,
                )

                y_pred = self._gen_net.predict(x_val).ravel()
                if self.problem_type == "binary_classification":
                    self._val_metrics.loc[idx, "ROC AUC"] = roc_auc_score(y_val, y_pred)
                self._val_metrics.loc[idx, "Accuracy"] = (
                    accuracy_score(y_val, (y_pred > 0.5)) * 100
                )
                for i, sensitive_attr in enumerate(self._transformed_biased_features):
                    self._fairness_metrics.loc[idx, sensitive_attr] = self._p_rule(
                        y_pred, z_val[:, i]
                    )
                display.clear_output(wait=True)
                self._plot_distributions(
                    y_pred,
                    z_val,
                    idx + 1,
                    self._val_metrics.loc[idx],
                    self._fairness_metrics.loc[idx],
                )
                plt.show(plt.gcf())

            # train discriminator
            self._set_trainable_gen_net(False)
            self._set_trainable_dis_net(True)
            self._gan_dis.fit(
                x,
                np.hsplit(z, z.shape[1]),
                batch_size=batch_size,
                # class_weight=class_weight_dis,
                epochs=1,
                verbose=verbose,
            )

            # train generator
            self._set_trainable_gen_net(True)
            self._set_trainable_dis_net(False)
            self._gan_gen.fit(
                x,
                [y] + np.hsplit(z, z.shape[1]),
                batch_size=x.shape[0],
                # class_weight=class_weight_gan_gen,
                epochs=5,
                verbose=verbose,
            )

    def predict(self, X):
        X = self._preprocess(X, is_train=False)
        y = self._gen_net.predict(X[self._transformed_debiased_features])
        if self._label_encoder is not None:
            y = self._label_encoder.inverse_transform(y)
        return y

    def _preprocess(self, X: pd.DataFrame, is_train=False):
        if is_train:
            if self.problem_type == "auto":
                if X[self.label].dtype == "object":
                    if len(X[self.label].unique()) > 2:
                        self.problem_type = "multiclass_classification"
                    else:
                        self.problem_type = "binary_classification"
                else:
                    self.problem_type = "regression"

            y = X[[self.label]]
            self._label_encoder = None
            if self.problem_type in [
                "multiclass_classification",
                "binary_classification",
            ]:
                self._label_encoder = LabelEncoder()
                y = pd.DataFrame(
                    {self.label: self._label_encoder.fit_transform(X[self.label])}
                )

            self._biased_categorical_columns = (
                X[self.biased_features]
                .dtypes[X[self.biased_features].dtypes == "object"]
                .index.tolist()
            )
            self._biased_numerical_columns = list(
                set(self.biased_features).difference(self._biased_categorical_columns)
            )
            self._biased_one_hot_encoder = OneHotEncoder(
                handle_unknown="ignore", drop="first"
            )
            self._biased_scaler = StandardScaler()
            X_biased, self._transformed_biased_features = [], []
            if len(self._biased_categorical_columns) > 0:
                X_biased.append(
                    pd.DataFrame(
                        self._biased_one_hot_encoder.fit_transform(
                            X[self._biased_categorical_columns]
                        ).todense(),
                        columns=self._biased_one_hot_encoder.get_feature_names_out(),
                    )
                )
                self._transformed_biased_features += (
                    self._biased_one_hot_encoder.get_feature_names_out().tolist()
                )

            if len(self._biased_numerical_columns) > 0:
                X_biased.append(
                    pd.DataFrame(
                        self._biased_scaler.fit_transform(
                            X[self._biased_numerical_columns]
                        ),
                        columns=self._biased_numerical_columns,
                    ),
                )
                self._transformed_biased_features += self._biased_numerical_columns
            self._n_sensitive = len(self._transformed_biased_features)

            self._debiased_categorical_columns = (
                X[self.debiased_features]
                .dtypes[X[self.debiased_features].dtypes == "object"]
                .index.tolist()
            )
            self._debiased_numerical_columns = list(
                set(self.debiased_features).difference(
                    self._debiased_categorical_columns
                )
            )
            self._debiased_one_hot_encoder = OneHotEncoder(
                handle_unknown="ignore", drop="first"
            )
            self._debiased_scaler = StandardScaler()
            X_debiased, self._transformed_debiased_features = [], []
            if len(self._debiased_categorical_columns) > 0:
                X_debiased.append(
                    pd.DataFrame(
                        self._debiased_one_hot_encoder.fit_transform(
                            X[self._debiased_categorical_columns]
                        ).todense(),
                        columns=self._debiased_one_hot_encoder.get_feature_names_out(),
                    )
                )
                self._transformed_debiased_features += (
                    self._debiased_one_hot_encoder.get_feature_names_out().tolist()
                )
            if len(self._debiased_numerical_columns) > 0:
                X_debiased.append(
                    pd.DataFrame(
                        self._debiased_scaler.fit_transform(
                            X[self._debiased_numerical_columns]
                        ),
                        columns=self._debiased_numerical_columns,
                    ),
                )
                self._transformed_debiased_features += self._debiased_numerical_columns
        else:
            y = X[[self.label]]
            if self._label_encoder is not None:
                y = pd.DataFrame(
                    {self.label: self._label_encoder.transform(X[self.label])}
                )

            X_biased = []
            if len(self._biased_categorical_columns) > 0:
                X_biased.append(
                    pd.DataFrame(
                        self._biased_one_hot_encoder.transform(
                            X[self._biased_categorical_columns]
                        ).todense(),
                        columns=self._biased_one_hot_encoder.get_feature_names_out(),
                    )
                )

            if len(self._biased_numerical_columns) > 0:
                X_biased.append(
                    pd.DataFrame(
                        self._biased_scaler.transform(
                            X[self._biased_numerical_columns]
                        ),
                        columns=self._biased_numerical_columns,
                    ),
                )

            X_debiased = []
            if len(self._debiased_categorical_columns) > 0:
                X_debiased.append(
                    pd.DataFrame(
                        self._debiased_one_hot_encoder.transform(
                            X[self._debiased_categorical_columns]
                        ).todense(),
                        columns=self._debiased_one_hot_encoder.get_feature_names_out(),
                    )
                )
            if len(self._debiased_numerical_columns) > 0:
                X_debiased.append(
                    pd.DataFrame(
                        self._debiased_scaler.transform(
                            X[self._debiased_numerical_columns]
                        ),
                        columns=self._debiased_numerical_columns,
                    ),
                )

        return pd.concat(X_biased + X_debiased + [y], axis=1)

    def _pretrain(self, X, epochs=10, verbose=0):
        x, y, z = (
            X[self._transformed_debiased_features].values,
            X[self.label].values,
            X[self._transformed_biased_features].values,
        )
        self._set_trainable_gen_net(True)

        self._gen_net.fit(x, y, epochs=epochs, verbose=verbose)

        self._set_trainable_gen_net(False)
        self._set_trainable_dis_net(True)
        # class_weight_adv = self._compute_class_weights(z)
        self._gan_dis.fit(
            x,
            np.hsplit(z, z.shape[1]),
            # class_weight=class_weight_adv,
            epochs=epochs,
            verbose=verbose,
        )

    def _init(
        self,
        X: pd.DataFrame,
    ):
        inputs = Input(shape=(X.shape[1],))
        dis_inputs = Input(shape=(1,))

        self._gen_net = self._create_gen_net(inputs)
        self._dis_net = self._create_dis_net(dis_inputs)

        self._set_trainable_gen_net = self._make_trainable(self._gen_net)
        self._set_trainable_dis_net = self._make_trainable(self._dis_net)

        self._set_trainable_gen_net(True)
        self._gen_net.compile(loss="binary_crossentropy", optimizer="adam")

        self._gan_gen = self._compile_gan_gen(inputs)
        self._gan_dis = self._compile_gan_dis(inputs)

        self._val_metrics = None
        self._fairness_metrics = None

    def _create_adv_net(self, inputs):
        dense1 = Dense(32, activation="relu")(inputs)
        dense2 = Dense(32, activation="relu")(dense1)
        dense3 = Dense(32, activation="relu")(dense2)
        outputs = [
            Dense(1, activation="sigmoid")(dense3) for _ in range(self._n_sensitive)
        ]
        return Model(inputs=[inputs], outputs=outputs)

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    def _create_gen_net(self, inputs):
        dense1 = Dense(32, activation="relu")(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation="relu")(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        activation = (
            "sigmoid" if self.problem_type == "binary_classification" else "softmax"
        )
        import pdb

        pdb.set_trace()
        outputs = Dense(1, activation=activation, name="y")(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])

    def _create_dis_net(self, inputs):
        dense1 = Dense(32, activation="relu")(inputs)
        dense2 = Dense(32, activation="relu")(dense1)
        dense3 = Dense(32, activation="relu")(dense2)
        outputs = [
            Dense(1, activation="sigmoid")(dense3) for _ in range(self._n_sensitive)
        ]
        return Model(inputs=[inputs], outputs=outputs)

    def _compile_gan_gen(self, inputs):
        gan = Model(
            inputs=[inputs],
            outputs=[self._gen_net(inputs)] + self._dis_net(self._gen_net(inputs)),
        )
        self._set_trainable_gen_net(True)
        self._set_trainable_dis_net(False)
        loss_weights = [1.0] + [-lambda_param for lambda_param in self.lambdas]
        gan.compile(
            loss=[self._target_loss()] * (len(loss_weights)),
            loss_weights=loss_weights,
            optimizer="adam",
        )
        return gan

    def _compile_gan_dis(self, inputs):
        model = Model(inputs=[inputs], outputs=self._dis_net(self._gen_net(inputs)))
        self._set_trainable_gen_net(False)
        self._set_trainable_dis_net(True)
        model.compile(
            loss=["binary_crossentropy"] * self._n_sensitive,
            loss_weights=self.lambdas,
            optimizer="adam",
        )
        return model

    def _compute_class_weights(self, y):
        class_weights = []
        if len(y.shape) == 1:
            classes = np.unique(y)
            balanced_weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weights.append(dict(zip(classes, balanced_weights)))
        else:
            n_attr = y.shape[1]
            for attr_idx in range(n_attr):
                classes = np.unique(y[:, attr_idx])
                balanced_weights = compute_class_weight(
                    "balanced", classes=classes, y=y[:, attr_idx]
                )
                class_weights.append(dict(zip(classes, balanced_weights)))
        return class_weights

    def _compute_target_class_weights(self, y):
        balanced_weights = compute_class_weight("balanced", y=y)
        class_weights = {"y": dict(zip(y.unique(), balanced_weights))}
        return class_weights

    def _p_rule(self, y_pred, z_values, threshold=0.5):
        y_z_1 = (
            y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
        )
        y_z_0 = (
            y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
        )
        odds = y_z_1.mean() / y_z_0.mean()
        return np.min([odds, 1 / odds]) * 100

    def _target_loss(self):
        return (
            "binary_crossentropy"
            if self.problem_type == "binary_classification"
            else "categorical_crossentropy"
            if self.problem_type == "multilabel_classification"
            else None
        )

    def _plot_distributions(
        self, y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        legend = {"race": ["black", "white"], "sex": ["female", "male"]}
        for idx, attr in enumerate(self.biased_features):
            for attr_val in [0, 1]:
                ax = sns.kdeplot(
                    data=y[Z[:, idx] == attr_val],
                    label="{}".format(legend[attr][attr_val]),
                    ax=axes[idx],
                    fill=True,
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 7)
            ax.set_yticks([])
            ax.set_title("sensitive attibute: {}".format(attr))
            if idx == 0:
                ax.set_ylabel("prediction distribution")
            ax.set_xlabel(r"$P({{income>50K}}|z_{{{}}})$".format(attr))
        if iteration:
            fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize="16")
        # if val_metrics is not None:
        #     fig.text(
        #         1.0,
        #         0.65,
        #         "\n".join(
        #             [
        #                 "Prediction performance:",
        #                 f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
        #                 f"- Accuracy: {val_metrics['Accuracy']:.1f}",
        #             ]
        #         ),
        #         fontsize="16",
        #     )
        if p_rules is not None:
            fig.text(
                1.0,
                0.4,
                "\n".join(
                    ["Satisfied p%-rules:"]
                    + [
                        f"- {attr}: {p_rules[attr]:.0f}%-rule"
                        for attr in p_rules.keys()
                    ]
                ),
                fontsize="16",
            )
        fig.tight_layout()
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", dpi=300)
        return fig
