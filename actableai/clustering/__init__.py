import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class ClusteringDataTransformer(TransformerMixin, BaseEstimator):
    """Transform numeric columns using StandardScaler and categorical columns
    using OneHotEncoder."""

    def fit_transform(self, X, categorical_cols):
        self.transformers = []
        self.categorical_cols = categorical_cols
        result = []
        for i in range(X.shape[1]):
            if i not in categorical_cols:
                t = StandardScaler()
                result.append(t.fit_transform(X[:, i : i + 1]))
            else:
                t = OneHotEncoder()
                result.append(t.fit_transform(X[:, i : i + 1]).todense())
            self.transformers.append(t)
        return np.hstack(result)

    def transform(self, X):
        result = []
        for i in range(X.shape[1]):
            x = self.transformers[i].transform(X[:, i : i + 1])
            if i in self.categorical_cols:
                x = x.todense()
            result.append(x)
        return np.hstack(result)

    def inverse_transform(self, X):
        c0 = 0
        result = []
        for i in range(len(self.transformers)):
            if type(self.transformers[i]) is StandardScaler:
                result.append(self.transformers[i].inverse_transform(X[:, c0 : c0 + 1]))
                c0 += 1
            else:
                t = self.transformers[i]
                result.append(
                    t.inverse_transform(X[:, c0 : c0 + len(t.categories_[0])])
                )
                c0 += len(t.categories_[0])
        return np.hstack(result)


def KMeans_scaled_inertia(scaled_data, k, alpha_k, *KMeans_args, **KMeans_kwargs):
    """
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns
    -------
    scaled_inertia: float
        scaled inertia value for current k
    """

    # fit k-means
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    kmeans = KMeans(n_clusters=k, *KMeans_args, **KMeans_kwargs).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def KMeans_pick_k(scaled_data, alpha_k, k_range, *KMeans_args, **KMeans_kwargs):
    # https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
    ans = []
    for k in k_range:
        scaled_inertia = KMeans_scaled_inertia(
            scaled_data, k, alpha_k, *KMeans_args, **KMeans_kwargs
        )
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k


def KMeans_pick_k_sil(X, k_range, *KMeans_args, **KMeans_kwargs):
    # https://newbedev.com/scikit-learn-k-means-elbow-criterion
    max_sil_coeff, best_k = 0, 2
    for k in k_range:
        kmeans = KMeans(n_clusters=k).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric="euclidean")
        print("Cluster: ", k, ", Silhouette coeff: ", sil_coeff)
        if max_sil_coeff < sil_coeff:
            max_sil_coeff = sil_coeff
            best_k = k
    return best_k
