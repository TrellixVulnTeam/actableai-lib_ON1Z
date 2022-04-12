import pytest

from actableai.data_validation.base import *
import pandas as pd
from actableai.tasks.clustering import AAIClusteringTask
from actableai.utils.dataset_generator import DatasetGenerator


@pytest.fixture(scope="function")
def clustering_task():
    yield AAIClusteringTask(use_ray=False)


class TestRemoteSegmentation:
    def test_segment_1_col(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "SUCCESS"
        assert "data" in r
        assert "explanation" in r["data"][0]

    def test_auto_cluster(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters="auto")

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_segment_mutiple_cols(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, features=["x", "y"], num_clusters=3)

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_segment_categorical_cols(self, clustering_task):
        df = pd.DataFrame({"x": ["a", "a", "c", "c", "c", "b", "b", "b"] * 3})

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_segment_mixed_type(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, "b", "a"] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainMixedChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_insufficent_data(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10],
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsSufficientDataChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_invalid_n_cluster_data(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=40)

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "IsValidNumberOfClusterChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_segment_with_explanations(self, clustering_task, init_ray):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, None, None, 8, 9, 10] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=3, explain_samples=True)

        assert r["status"] == "SUCCESS"
        assert "data" in r
        for cluster in r["data"]:
            for row in cluster["value"]:
                assert "explanation" in row["column"]

    def test_segment_empty_col(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_max_train_samples(self, clustering_task):
        df = pd.DataFrame(
            {
                "x": [1, None, None, None, None, None, None, None, None, None] * 2,
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            }
        )

        r = clustering_task.run(df, num_clusters=3, max_train_samples=10)

        assert r["status"] == "SUCCESS"
        assert "data" in r

    def test_text_column(self, clustering_task):
        df = DatasetGenerator.generate(
            columns_parameters=[
                {"type": "text", "word_range": (5, 10)},
                {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2},
            ],
            rows=20,
            random_state=0,
        )

        r = clustering_task.run(df, num_clusters=3)

        assert r["status"] == "FAILURE"
        assert "validations" in r
        assert len(r["validations"]) > 0
        assert r["validations"][0]["name"] == "DoNotContainTextChecker"
        assert r["validations"][0]["level"] == CheckLevels.CRITICAL
