from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.correlation_calculator import (
    CorrelationCalculator,
)


class TestNormalizeColumn:
    def setup(self):
        self._calculator = CorrelationCalculator()

    @pytest.mark.parametrize(
        "df, column, expected",
        [
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, 81, 93, 74],
                    }
                ),
                "Maths",
                pd.Series([78, 85, 67, 69, 53, 81, 93, 74]),
            ),
            (
                pd.DataFrame({"Maths": ["b", "b", "a", "c", "b"]}),
                "Maths",
                pd.Series([0, 0, 1, 2, 0]),
            ),
            (pd.DataFrame({"Time": [datetime.now()]}), "Time", pd.Series()),
        ],
    )
    def test_calc(self, df, column, expected):
        result = self._calculator._normalize_column(df, column)
        assert result.equals(expected)


class TestCalculateCorrelation:
    def setup(self):
        self._calculator = CorrelationCalculator()

    @pytest.mark.parametrize(
        "df, source_column, expected",
        [
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, 93, 74],
                        "Physics": [78, 85, 67, 69, 53, 93, 74],
                        "History": [-78, -85, -67, -69, -53, -93, -74],
                    }
                ),
                "Maths",
                {"History": -0.9999999999999999, "Physics": 0.9999999999999999},
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, 81, 93, 74],
                        "Physics": [81, 77, 63, 74, 46, 72, 88, 76],
                        "History": [53, 65, 95, 87, 63, 58, 73, 42],
                    }
                ),
                "Maths",
                {
                    "Physics": 0.9063395113712818,
                    "History": -0.15906252133255808,
                },
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, np.nan, 93, 74],
                        "Physics": [81, 77, 63, 74, 46, 72, 88, 76],
                        "History": [53, 65, 95, 87, 63, 58, 73, 42],
                    }
                ),
                "Maths",
                {
                    "Physics": 0.925369627022846,
                    "History": -0.12305046257577017,
                },
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, np.nan, 93, 74],
                        "Physics": [81, 77, 63, 74, 46, 72, 88, 76],
                        "History": ["a", "b", "c", "a", "d", "e", "g", "h"],
                    }
                ),
                "Maths",
                {
                    "History": 0.14379170227946525,
                    "Physics": 0.925369627022846,
                },
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, np.nan, 93, 74],
                        "Physics": [81, np.nan, np.nan, 74, 46, 72, 88, 76],
                        "History": ["a", np.nan, "c", "a", "d", "e", "g", "h"],
                    }
                ),
                "Maths",
                {
                    "History": 0.08316997840317032,
                    "Physics": 0.9452412933181921,
                },
            ),
            (
                pd.DataFrame(
                    {
                        "Time": [
                            datetime.now(),
                            datetime.now(),
                        ],
                        "Physics": [
                            1,
                            2,
                        ],
                    }
                ),
                "Time",
                {},
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [
                            np.nan,
                            np.nan,
                        ],
                        "Physics": [
                            np.nan,
                            np.nan,
                        ],
                    }
                ),
                "Maths",
                {},
            ),
        ],
    )
    def test_calc(self, df, source_column, expected):
        result = self._calculator._calculate_correlation(df, source_column)
        assert result == expected


class TestMostCorrelatedColumns:
    def setup(self):
        self._calculator = CorrelationCalculator()

    def test_calc(self):
        mock_correlation = {"Math": 0.6, "Physics": 0.5, "History": -0.8}
        mock_correlation_func = MagicMock()
        self._calculator._calculate_correlation = mock_correlation_func
        mock_correlation_func.return_value = mock_correlation
        result = self._calculator.most_correlate_columns(MagicMock(), "anything", top=2)
        assert result == ["History", "Math"]


class TestCalculateCorrelationsForAllColumnPairs:
    def setup(self):
        self._calculator = CorrelationCalculator()

    @pytest.mark.parametrize(
        "df, mock_correlation, expected",
        [
            (
                pd.DataFrame(
                    {
                        "Maths": [78, 85, 67, 69, 53, np.nan, 93, 74],
                        "Physics": [81, 77, 63, 74, 46, 72, 88, 76],
                        "History": ["a", "b", "c", "a", "d", "e", "g", "h"],
                    }
                ),
                {
                    "Maths": {
                        "Physics": 0.1,
                        "History": 0.2,
                    },
                    "Physics": {
                        "Maths": 0.1,
                        "History": 0.4,
                    },
                    "History": {
                        "Maths": 0.2,
                        "Physics": 0.4,
                    },
                },
                pd.DataFrame(
                    data={
                        "Maths": [1, 0.1, 0.2],
                        "Physics": [0.1, 1, 0.4],
                        "History": [0.2, 0.4, 1],
                    },
                    index=["Maths", "Physics", "History"],
                ),
            ),
        ],
    )
    def test_calc(self, df, mock_correlation, expected):
        self._calculator._correlation_scores = mock_correlation

        self._calculator._calculate_correlation = MagicMock()

        result = self._calculator.calculate_correlations_for_all_column_pairs(df)

        assert result.equals(expected)

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame(
                {
                    "Maths": [78, 85, 67, 69, 53, np.nan, 93, 74],
                    "Physics": [81, 77, 63, 74, 46, 72, 88, 76],
                    "History": ["a", "b", "c", "a", "d", "e", "g", "h"],
                }
            ),
        ],
    )
    def test_calc_integration(self, df):
        result = self._calculator.calculate_correlations_for_all_column_pairs(df)

        for col1 in df.columns:
            for col2 in df.columns:
                assert round(result.at[col1, col2], 3) == round(
                    result.at[col2, col1], 3
                )
                assert 0 <= result.at[col1, col2] <= 1
