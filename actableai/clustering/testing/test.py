import numpy as np
import unittest
from actableai.clustering import ClusteringDataTransformer


class TestClusteringDataTransformer:
    def test_clustering_data_transformer(self):
        X = np.asarray(
            [[1.0, "a", "@", 7.0], [2.0, "b", "#", 8.0], [3.0, "c", "$", 9.0]]
        )
        t = ClusteringDataTransformer()
        transformed_x = t.fit_transform(X, categorical_cols=[1, 2])
        assert np.all(
            np.round(transformed_x, decimals=2)
            == np.array(
                [
                    [-1.22, 1, 0, 0, 0, 0, 1, -1.22],
                    [0, 0, 1, 0, 1, 0, 0, 0],
                    [1.22, 0, 0, 1, 0, 1, 0, 1.22],
                ]
            )
        )
        inversed_x = t.inverse_transform(transformed_x)
        assert np.all(inversed_x == X)
