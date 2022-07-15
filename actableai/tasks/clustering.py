import pandas as pd
from typing import Dict, List, Optional, Union

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIClusteringTask(AAITask):
    """Clustering Task

    Args:
        AAITask: Base class for every tasks
    """

    @AAITask.run_with_ray_remote(TaskType.CLUSTERING)
    def run(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        num_clusters: Union[int, str] = "auto",
        explain_samples: bool = False,
        auto_num_clusters_min: int = 2,
        auto_num_clusters_max: int = 20,
        init: str = "glorot_uniform",
        pretrain_optimizer: str = "adam",
        update_interval: int = 30,
        pretrain_epochs: int = 300,
        alpha_k: float = 0.01,
        cluster_explain_max_depth=20,
        cluster_explain_min_impurity_decrease=0.001,
        cluster_explain_min_samples_leaf=0.001,
        cluster_explain_min_precision=0.8,
        max_train_samples: Optional[int] = None,
    ) -> Dict:
        """Runs a clustering analysis on df

        Args:
            df: Input DataFrame
            features: Features used in Input DataFrame. Defaults to None.
            num_clusters: Number of different clusters assignable to each row.
                "auto" automatically finds the optimal number of clusters.
                Defaults to "auto".
            explain_samples: If the result contains a human readable explanation of
                the clustering. Defaults to False.
            auto_num_clusters_min: Minimum number of clusters when num_clusters is
                _auto_. Defaults to 2.
            auto_num_clusters_max: Maximum number of clusters when num_clusters is
                _auto_. Defaults to 20.
            init: ?. Defaults to "glorot_uniform".
            pretrain_optimizer: Optimizer for pretaining phase of autoencoder.
                Defaults to "adam".
            update_interval: The interval to check the stopping criterion and update the
                cluster centers. Default to 140.
            pretrain_epochs: Number of epochs for pretraining DEC. Defaults to 300.
            alpha_k: The factor to control the penalty term of the number of clusters.
                Default to 0.01.
            max_train_samples: Number of randomly selected rows to train the DEC.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> AAIClusteringTask().run(df, ["feature1", "feature2", "feature3"])

        Returns:
            Dict: Dictionnary of results
        """
        import tensorflow as tf

        # This needs to be done to make the shap library compatible
        tf.compat.v1.disable_v2_behavior()

        import time
        import shap
        import pandas as pd
        import numpy as np
        from collections import defaultdict
        from tensorflow.keras.optimizers import SGD
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.manifold import TSNE
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.tree import DecisionTreeClassifier
        from actableai.clustering.dec_keras import DEC
        from actableai.data_validation.params import ClusteringDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.preprocessors.preprocessing import impute_df
        from actableai.clustering import ClusteringDataTransformer
        from actableai.clustering.explain import generate_cluster_descriptions

        from actableai.utils.sanitize import sanitize_timezone

        pd.set_option("chained_assignment", "warn")
        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        if features is None:
            features = list(df.columns)
        if max_train_samples is None:
            max_train_samples = len(df)

        df_train = df[features]

        data_validation_results = ClusteringDataValidator().validate(
            features,
            df_train,
            n_cluster=num_clusters,
            explain_samples=explain_samples,
            max_train_samples=max_train_samples,
        )
        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
                "data": {},
            }

        df_train = df_train.dropna(how="all", axis=1)
        features = list(df_train.columns)

        numeric_columns = list(df_train.select_dtypes(include=["number"]).columns)
        if len(numeric_columns):
            df_train[numeric_columns] = pd.DataFrame(
                SimpleImputer(strategy="median").fit_transform(
                    df_train[numeric_columns]
                ),
                columns=numeric_columns,
            )

        category_map = {}
        label_encoder = {}
        for i, c in enumerate(features):
            if df_train[c].dtype in ["object", "bool"]:
                df_train[c] = df_train[c].fillna("NA")
                le = LabelEncoder()
                df_train[c] = le.fit_transform(df_train[c])
                category_map[i] = le.classes_
                label_encoder[c] = le

        # Process data
        categorical_features = list(category_map.keys())
        preprocessor = ClusteringDataTransformer()
        transformed_values = preprocessor.fit_transform(
            df_train.values, categorical_cols=categorical_features
        )
        if num_clusters == "auto" and max_train_samples is not None:
            auto_num_clusters_max = min(auto_num_clusters_max, max_train_samples)
        dec = DEC(
            dims=[transformed_values.shape[-1], 500, 500, 2000, 10],
            init=init,
            n_clusters=num_clusters,
            auto_num_clusters_min=auto_num_clusters_min,
            auto_num_clusters_max=auto_num_clusters_max,
            alpha_k=alpha_k,
        )
        sampled_transformed_values = transformed_values
        if max_train_samples is not None:
            max_train_samples = min(max_train_samples, transformed_values.shape[0])
            sampled_transformed_values = (
                pd.DataFrame(transformed_values).sample(max_train_samples).values
            )
        dec.pretrain(
            x=sampled_transformed_values,
            optimizer=pretrain_optimizer,
            epochs=pretrain_epochs,
        )
        dec.compile(optimizer=SGD(0.01, 0.9), loss="kld")

        dec.fit(sampled_transformed_values, update_interval=update_interval)

        probs = dec.predict_proba(transformed_values)
        cluster_ids = probs.argmax(axis=1)
        sample_ids_nearest_to_centroids = np.asarray(
            [probs[:, c].argmax(axis=0) for c in range(dec.n_clusters)]
        )
        z = dec.project(transformed_values)

        df_shap_values = None
        if explain_samples:
            background_samples = 100
            if len(transformed_values) < 100:
                background_samples = int(len(transformed_values) * 0.1)

            background = transformed_values[
                np.random.choice(
                    transformed_values.shape[0],
                    background_samples,
                    replace=False,
                )
            ]

            explainer = shap.DeepExplainer(dec.model, np.array(background))

            shap_values = explainer.shap_values(
                np.array(transformed_values), check_additivity=False
            )

            # Extract only the shap values for the predicted values
            shap_values = np.array(shap_values)
            row_index, column_index = np.meshgrid(
                np.arange(shap_values.shape[1]), np.arange(shap_values.shape[2])
            )
            shap_values = shap_values[cluster_ids, row_index, column_index].transpose(
                1, 0
            )

            df_final_shap_values = pd.DataFrame(
                0, index=np.arange(transformed_values.shape[0]), columns=features
            )

            for i, feature in enumerate(features):
                df_final_shap_values[feature] = shap_values[
                    :, preprocessor.feature_links[i]
                ].sum(axis=1)

            df_shap_values = df_final_shap_values

        try:
            lda = LinearDiscriminantAnalysis(n_components=2)
            x_embedded = lda.fit_transform(z, cluster_ids)
            projected_cluster_centers = lda.transform(dec.encoded_cluster_centers)
        except Exception:
            tsne = TSNE(n_components=2)
            embedded = tsne.fit_transform(np.vstack([z, dec.encoded_cluster_centers]))
            x_embedded, projected_cluster_centers = (
                embedded[: z.shape[0], :],
                embedded[z.shape[0] :, :],
            )

        # Return data
        data = []
        points_x = x_embedded[:, 0]
        points_y = x_embedded[:, 1]

        for c in label_encoder:
            df_train[c] = label_encoder[c].inverse_transform(df_train[c])

        origin_dict = df_train.to_dict("record")
        for idx, (i, j, k, l) in enumerate(
            zip(points_x.tolist(), points_y.tolist(), cluster_ids, origin_dict)
        ):
            data.append((k, {"x": i, "y": j}, l))

        res = defaultdict(list)
        for idx, (k, v, s) in enumerate(data):
            res[k].append({"train": v, "column": s, "index": df_train.index[idx]})

        # Explain clusters
        clusters = [{"cluster_id": int(k), "value": v} for k, v in res.items()]
        rows, cluster_id = [], []
        for c in clusters:
            for row in c["value"]:
                rows.append(row["column"])
                cluster_id.append(c["cluster_id"])
        df_ = pd.DataFrame(rows)
        impute_df(df_)
        df_dummies = pd.get_dummies(df_)
        dummy_columns = set(df_dummies.columns) - set(df_.columns)
        clf = DecisionTreeClassifier(
            max_depth=cluster_explain_max_depth,
            min_impurity_decrease=cluster_explain_min_impurity_decrease,
            min_samples_leaf=cluster_explain_min_samples_leaf / len(clusters),
        )
        clf.fit(df_dummies, cluster_id)
        cluster_explanations = generate_cluster_descriptions(
            clf.tree_,
            df_dummies.columns,
            dummy_columns,
            min_precision=cluster_explain_min_precision,
        )

        for cluster in clusters:
            cid = cluster["cluster_id"]
            cluster["explanation"] = "\n".join(cluster_explanations[cid])
            cluster["encoded_value"] = dec.encoded_cluster_centers[cid]
            cluster["projected_value"] = projected_cluster_centers[cid]
            cluster["projected_nearest_point"] = x_embedded[
                sample_ids_nearest_to_centroids[cid]
            ]

        runtime = time.time() - start

        return {
            "data_v2": {
                "clusters": clusters,
                "shap_values": df_shap_values,
            },
            "data": clusters,
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
