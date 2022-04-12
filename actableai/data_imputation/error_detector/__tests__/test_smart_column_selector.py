from math import inf
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation import ColumnType
from actableai.data_imputation.error_detector.cell_erros import (
    ErrorCandidate,
    CellError,
    ErrorType,
)
from actableai.data_imputation.error_detector.smart_column_selector import (
    SmartErrorCellSelector,
)
from actableai.data_imputation.type_recon.type_detector import DfTypes

stub_path = "actableai.data_imputation.error_detector.smart_column_selector"


class TestDistanceToCenter:
    def setup(self):
        self._selector = SmartErrorCellSelector()

    @pytest.mark.parametrize(
        "df,dftypes, column, index",
        [
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, 90, 91, 100, 90],
                        "Physics": [78, 69, 54, 91, 92, 90],
                        "History": [69, 78, 56, 90, 100, 80],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.Integer),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, np.nan, 91, 100, 90],
                        "Physics": [78, 69, 54, 91, 92, 90],
                        "History": [69, 78, 56, 90, 100, 80],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.Integer),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
        ],
    )
    @patch(f"{stub_path}.cosine_distances")
    @patch(f"{stub_path}.KMeans")
    def test_calc_should_user_correct_parameters(
        self, mock_kmeans_mod, mock_distance_func, df, dftypes, column, index
    ):
        mock_kmeans = MagicMock()
        mock_kmeans_mod.return_value = mock_kmeans
        mock_center = MagicMock()
        mock_kmeans.cluster_centers_ = [mock_center]

        self._selector._distance_to_center(df, dftypes, column, index)
        assert (
            mock_distance_func.call_args_list[0][0][0][0] == np.array([90, 69, 78])
        ).all()
        assert mock_distance_func.call_args_list[0][0][1] == [mock_center]

    @pytest.mark.parametrize(
        "df, dftypes, column, index",
        [
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, 90, 91, 100, 90],
                        "Physics": [78, 69, 54, 91, 92, 90],
                        "History": [69, 78, 56, 90, 100, 80],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.Integer),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, np.nan, 91, 100, 90],
                        "Physics": [78, 69, np.nan, 91, 92, np.nan],
                        "History": [69, 78, 56, 90, 100, np.nan],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.Integer),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, np.nan, 90, 90, 90],
                        "Physics": [
                            "callahan eye foundation hospital",
                            np.nan,
                            "callahan eye foundation hospital",
                            "shelen keller memorial hospital",
                            "outheast alabama medical center",
                            "helen keller memorial hospital",
                        ],
                        "History": [69, 78, 56, 90, 100, np.nan],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.String),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
            (
                pd.DataFrame(
                    {
                        "Maths": [100, 90, np.nan, 90, 90, 90],
                        "Physics": [
                            "callahan eye foundation hospital",
                            np.nan,
                            "callahan eye foundation hospital",
                            "shelen keller memorial hospital",
                            "outheast alabama medical center",
                            "helen keller memorial hospital",
                        ],
                        "History": [69, np.nan, 56, 90, 100, np.nan],
                    }
                ),
                DfTypes(
                    [
                        ("Maths", ColumnType.Integer),
                        ("Physics", ColumnType.String),
                        ("History", ColumnType.Integer),
                    ]
                ),
                "Maths",
                1,
            ),
        ],
    )
    def test_calc(self, df, dftypes, column, index):
        result = self._selector._distance_to_center(df, dftypes, column, index)
        assert isinstance(result, float)


class TestFindColumnsWithUniqValue:
    def setup(self):
        self._selector = SmartErrorCellSelector()

    @pytest.mark.parametrize(
        "df,index,expect",
        [
            (
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4, 1, 2, 3],
                        "b": ["a", "b", "c", "a", "b", "c", "d"],
                    }
                ),
                3,
                {"a"},
            ),
            (
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4, 1, 2, 3],
                        "b": ["a", "b", "c", "a", "b", "c", "d"],
                    }
                ),
                6,
                {"b"},
            ),
            (
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4, 1, 2, 3],
                        "b": ["a", "b", "c", "a", "b", "c", "d"],
                    }
                ),
                1,
                set(),
            ),
            (
                pd.DataFrame(),
                1,
                set(),
            ),
        ],
    )
    def test_find(self, df, index, expect):
        assert self._selector._find_columns_with_uniq_value(df, index) == expect


class TestFindActualErrorColumn:
    @patch(f"{stub_path}.CorrelationCalculator")
    def setup(self, mock_correlation_calculator_mod):
        mock_most_correlate_columns = MagicMock()
        mock_distance_to_center = MagicMock()
        mock_correlation_calculator = MagicMock()
        mock_correlation_calculator_mod.return_value = mock_correlation_calculator

        self._selector = SmartErrorCellSelector()
        self._selector.reset()
        mock_correlation_calculator.most_correlate_columns = mock_most_correlate_columns
        self._selector._distance_to_center = mock_distance_to_center
        self._most_correlate_columns = mock_most_correlate_columns
        self._distance_to_center = mock_distance_to_center

    def test_find_when_one_cell_is_empty(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [1, 2, 3]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 2
        candidates.potential_columns = {"a", "b"}
        result = self._selector._find_actual_error_column(
            df, MagicMock(), candidates, {"a", "b"}
        )
        return list(result) == [CellError("a", 2, ErrorType.INVALID)]

    def test_find_when_multiple_cells_are_empty(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [1, 2, np.nan]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 2
        candidates.potential_columns = {"a", "b"}
        result = list(
            self._selector._find_actual_error_column(
                df, MagicMock(), candidates, {"a", "b"}
            )
        )
        assert sorted(result, key=lambda x: x.column) == [
            CellError("a", 2, ErrorType.INVALID),
            CellError("b", 2, ErrorType.INVALID),
        ]

    def test_find_distance_column_all_columns_get_distance_to_center_are_inf(
        self,
    ):
        mock_find_columns_with_uniq_value = MagicMock()
        mock_find_columns_with_uniq_value.return_value = set("a")
        self._selector._find_columns_with_uniq_value = mock_find_columns_with_uniq_value
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 0
        candidates.potential_columns = {"a", "b"}
        self._most_correlate_columns.return_value = ["b"]
        self._distance_to_center.return_value = inf
        result = list(
            self._selector._find_actual_error_column(
                df, MagicMock(), candidates, {"a", "b"}
            )
        )
        assert self._distance_to_center.call_count == 2
        assert result == [
            CellError("a", 0, ErrorType.INVALID),
            CellError("b", 0, ErrorType.INVALID),
        ]

    def test_find_distance_column_only_one_column_get_distance_to_center_is_inf(
        self,
    ):
        mock_find_columns_with_uniq_value = MagicMock()
        mock_find_columns_with_uniq_value.return_value = set("a")
        self._selector._find_columns_with_uniq_value = mock_find_columns_with_uniq_value
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 0
        candidates.potential_columns = {"a", "b"}
        self._most_correlate_columns.return_value = ["b"]
        self._distance_to_center.side_effect = [inf, 1]
        result = list(
            self._selector._find_actual_error_column(
                df, MagicMock(), candidates, {"a", "b"}
            )
        )
        assert self._distance_to_center.call_count == 2
        assert result == [
            CellError("a", 0, ErrorType.INVALID),
        ]

    def test_find_when_most_correlate_columns_is_empty(self):
        mock_find_columns_with_uniq_value = MagicMock()
        mock_find_columns_with_uniq_value.return_value = set()
        self._selector._find_columns_with_uniq_value = mock_find_columns_with_uniq_value
        self._most_correlate_columns.side_effect = [["b"], []]
        self._distance_to_center.side_effect = [1]
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [1, 2, 3]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 2
        candidates.potential_columns = {"a", "b"}
        result = list(
            self._selector._find_actual_error_column(
                df, MagicMock(), candidates, {"a", "b"}
            )
        )
        assert self._distance_to_center.call_count == 1
        assert result == [
            CellError("a", 2, ErrorType.INVALID),
        ]

    @pytest.mark.parametrize(
        "distances, expect_errors",
        [
            (
                [0.1, 0.3, 0.3],
                [
                    CellError("b", 2, ErrorType.INVALID),
                    CellError("c", 2, ErrorType.INVALID),
                ],
            )
        ],
    )
    def test_find(self, distances, expect_errors):
        mock_find_columns_with_uniq_value = MagicMock()
        mock_find_columns_with_uniq_value.return_value = set()
        self._selector._find_columns_with_uniq_value = mock_find_columns_with_uniq_value
        self._most_correlate_columns.return_value = ["b"]
        self._distance_to_center.side_effect = distances
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 4, 5]})
        candidates = MagicMock(ErrorCandidate)
        candidates.index = 2
        candidates.potential_columns = {"a", "b", "c"}
        result = list(
            self._selector._find_actual_error_column(
                df, MagicMock(), candidates, {"a", "b"}
            )
        )
        assert self._distance_to_center.call_count == 3
        assert sorted(result, key=lambda x: x.column) == expect_errors
