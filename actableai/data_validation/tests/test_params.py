import pandas as pd
from pandas._testing import rands_array
import numpy as np

from actableai.data_validation.params import BayesianRegressionDataValidator
from actableai.data_validation.params import CausalDataValidator


class TestBayesianRegressionDataValidator:
    def test_validate_CheckNUnique(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 200),
                "y": ["a" for _ in range(200)],
                "z": [i for i in range(200)],
            }
        )

        validation_results = BayesianRegressionDataValidator().validate(
            "x", ["y", "z"], df, 1
        )
        validation_results = [x for x in validation_results if x is not None]

        assert "CheckNUnique" in [x.name for x in validation_results]

    def test_validate_not_CheckNUnique(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": ["a" for _ in range(5)],
                "z": ["b" for i in range(5)],
            }
        )

        validation_results = BayesianRegressionDataValidator().validate(
            "x", ["y", "z"], df, 1
        )
        validation_results = [x for x in validation_results if x is not None]

        assert "CheckNUnique" not in [x.name for x in validation_results]


class TestCausalDataValidator:
    def test_validate(self):
        df = pd.DataFrame(
            {
                "x": rands_array(10, 5),
                "y": rands_array(10, 5),
                "z": rands_array(10, 5),
                "t": rands_array(10, 5),
            }
        )

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])
        validation_results = [x for x in validation_results if x is not None]

        assert "IsSufficientDataChecker" in [x.name for x in validation_results]

    def test_validate_nan_treatment(self):
        df = pd.DataFrame(
            {
                "x": rands_array(100, 5),
                "y": rands_array(100, 5),
                "z": rands_array(100, 5),
                "t": rands_array(100, 5),
            }
        )
        df["x"] = np.nan

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])
        validation_results = [x for x in validation_results if x is not None]

        assert "IsSufficientDataChecker" in [x.name for x in validation_results]

    def test_validate_nan_outcome(self):
        df = pd.DataFrame(
            {
                "x": rands_array(100, 5),
                "y": rands_array(100, 5),
                "z": rands_array(100, 5),
                "t": rands_array(100, 5),
            }
        )
        df["y"] = np.nan

        validation_results = CausalDataValidator().validate(["x"], ["y"], df, [], [])
        validation_results = [x for x in validation_results if x is not None]

        assert "IsSufficientDataChecker" in [x.name for x in validation_results]
