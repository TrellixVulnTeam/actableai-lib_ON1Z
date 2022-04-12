from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIClusteringTask(AAITask):
    """
    Clustering Task
    """

    @AAITask.run_with_ray_remote(TaskType.CLUSTERING)
    def run(
        self,
        df,
        features=None,
        num_clusters="auto",
        explain_samples=False,
        explainer_task_params=None,
        auto_num_clusters_min=2,
        auto_num_clusters_max=20,
        init="glorot_uniform",
        pretrain_optimizer="adam",
        update_interval=30,
        pretrain_epochs=300,
        explain_max_concurrent=1,
        explain_precision_threshold=0.8,
        alpha_k=0.01,
        max_train_samples=None,
    ):
        """
        TODO write documentation
        """
        import time
        import ray
        import pandas as pd
        import numpy as np
        import psutil
        from multiprocessing.pool import ThreadPool
        from collections import defaultdict
        from tensorflow.keras.optimizers import SGD
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
        from sklearn.manifold import TSNE
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier
        from actableai.clustering.dec_keras import DEC, DECAnchor
        from actableai.data_validation.params import ClusteringDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.utils import gen_anchor_explanation
        from actableai.utils.preprocessing import impute_df
        from actableai.clustering import ClusteringDataTransformer
        from actableai.clustering.explain import generate_cluster_descriptions
        from alibi.explainers import AnchorTabular

        from actableai.utils.sanitize import sanitize_timezone

        pd.set_option("chained_assignment", "warn")
        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        if features is None:
            features = list(df.columns)

        df_train = df[features]

        data_validation_results = ClusteringDataValidator().validate(
            features, df_train, n_cluster=num_clusters, explain_samples=explain_samples
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

        category_map = {}
        for i, c in enumerate(features):
            if df_train[c].dtype == "object":
                df_train[c] = df_train[c].fillna("NA")
                le = LabelEncoder()
                df_train[c] = le.fit_transform(df_train[c])
                category_map[i] = le.classes_
        df_train = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(df_train),
            columns=df_train.columns,
        )

        # Process data
        ordinal_features = [
            i
            for i in range(len(df_train.columns))
            if i not in list(category_map.keys())
        ]
        categorical_features = list(category_map.keys())
        preprocessor = ClusteringDataTransformer()
        transformed_values = preprocessor.fit_transform(
            df_train.values, categorical_cols=categorical_features
        )

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

        anchors = [None] * df_train.shape[0]
        if explain_samples:
            if explainer_task_params is None:
                explainer_task_params = {}

            predict_fn = lambda x: dec.predict(preprocessor.transform(x))
            explainer = AnchorTabular(
                predict_fn, df_train.columns, categorical_names=category_map, seed=1
            )
            explainer.fit(df_train.values)
            # Remove Tensorflow model here as it's not serializable
            explainer.predictor = None
            for sampler in explainer.samplers:
                sampler.predictor = None

            dec_anchor_task = DECAnchor(**explainer_task_params)

            dec_anchor_pool = ThreadPool(processes=explain_max_concurrent)

            chunks = np.array_split(df_train.values, explain_max_concurrent)
            dec_anchor_async_results = [
                dec_anchor_pool.apply_async(
                    dec_anchor_task.run,
                    kwds={
                        "n_clusters": dec.n_clusters,
                        "input_dim": transformed_values.shape[1],
                        "explainer": explainer,
                        "dec_encoder_weights": dec.encoder.get_weights(),
                        "dec_weights": dec.model.get_weights(),
                        "init": init,
                        "preprocessor": preprocessor,
                        "df": chunk,
                        "threshold": explain_precision_threshold,
                    },
                )
                for chunk in chunks
            ]

            anchors = [
                anchor
                for anchors in dec_anchor_async_results
                for anchor in anchors.get()
            ]

            df["explanation"] = [
                gen_anchor_explanation(a, df.shape[0]) for a in anchors
            ]

        try:
            lda = LinearDiscriminantAnalysis(n_components=2)
            x_embedded = lda.fit_transform(z, cluster_ids)
            projected_cluster_centers = lda.transform(dec.encoded_cluster_centers)
        except:
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

        origin_dict = df.to_dict("record")
        for idx, (i, j, k, l, e) in enumerate(
            zip(points_x.tolist(), points_y.tolist(), cluster_ids, origin_dict, anchors)
        ):
            data.append((k, {"x": i, "y": j}, l, e))

        res = defaultdict(list)
        for idx, (k, v, s, e) in enumerate(data):
            res[k].append({"train": v, "column": s, "index": df.index[idx]})

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
            max_depth=20,
            min_impurity_decrease=0.01,
            min_samples_leaf=0.1 / len(clusters),
        )
        clf.fit(df_dummies, cluster_id)
        cluster_explanations = generate_cluster_descriptions(
            clf.tree_, df_dummies.columns, dummy_columns
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
            "data": clusters,
            "status": "SUCCESS",
            "messenger": "",
            "runtime": runtime,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
