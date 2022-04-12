from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.auto_fixer import FixStrategy
from actableai.data_imputation.auto_fixer.strategy import (
    determine_fix_strategy,
    determine_refine_strategy,
)
from actableai.data_imputation.error_detector import ColumnErrors
from actableai.data_imputation.error_detector.cell_erros import (
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.types import ColumnType

stub = "actableai.data_imputation.auto_fixer.strategy"


class TestDetermineFixStrategy:
    @patch(f"{stub}.Counter")
    @patch(f"{stub}.config")
    @pytest.mark.parametrize(
        "series, column_type, column_errors, expect_fix_strategy",
        [
            (
                pd.Series(data=["a", "a", "a", "a"]),
                ColumnType.Category,
                ColumnErrors(set()),
                FixStrategy.SINGLE_CATEGORY,
            ),
            (
                pd.Series(data=[None, np.nan, np.nan, "a", None, np.nan]),
                ColumnType.Category,
                ColumnErrors(set()),
                FixStrategy.SINGLE_CATEGORY,
            ),
            (
                pd.Series(data=[None, np.nan, np.nan, None, np.nan]),
                ColumnType.Category,
                ColumnErrors(set()),
                FixStrategy.UNABLE_TO_FIX,
            ),
            (
                pd.Series(data=[1, 2, 3, 4]),
                ColumnType.NULL,
                ColumnErrors(set()),
                FixStrategy.UNABLE_TO_FIX,
            ),
            (
                pd.Series(data=[1, 2, 3]),
                ColumnType.Integer,
                ColumnErrors(
                    {
                        CellError("a", 0, ErrorType.INVALID),
                        CellError("a", 1, ErrorType.INVALID),
                        CellError("a", 2, ErrorType.INVALID),
                    }
                ),
                FixStrategy.UNABLE_TO_FIX,
            ),
            (
                pd.Series(data=[1, 2, 3]),
                ColumnType.Integer,
                ColumnErrors(
                    {
                        CellError("a", 0, ErrorType.INVALID),
                        CellError("a", 1, ErrorType.INVALID),
                        CellError("a", 2, ErrorType.INVALID),
                    }
                ),
                FixStrategy.UNABLE_TO_FIX,
            ),
            (
                pd.Series(data=["a", "b", "a"]),
                ColumnType.Category,
                ColumnErrors(
                    {
                        CellError("a", 1, ErrorType.INVALID),
                    }
                ),
                FixStrategy.SINGLE_CATEGORY,
            ),
        ],
    )
    def test_should_return_correct_strategy(
        self,
        mock_config,
        mock_counter,
        series,
        column_type,
        column_errors,
        expect_fix_strategy,
    ):
        mock_config.UNABLE_TO_FIX_DISTINCT_SIZE_THRESHOLD = 1
        mock_counter.return_value.most_common.return_value = [("", 2)]
        mock_counter.return_value.__len__.return_value = 1
        assert (
            determine_fix_strategy(series, column_type, column_errors)
            == expect_fix_strategy
        )

    @patch(f"{stub}.config")
    def test_should_return_single_category_strategy_when_distinct_value_count_lt_threshold_and_type_is_category(
        self,
        mock_config,
    ):
        mock_config.UNABLE_TO_FIX_DISTINCT_SIZE_THRESHOLD = 100
        series = pd.Series(data=["a", "b", "a"])
        assert (
            determine_fix_strategy(series, ColumnType.Category, ColumnErrors(set()))
            == FixStrategy.SINGLE_CATEGORY
        )

    @patch(f"{stub}.config")
    def test_should_return_unable_to_fix_strategy_when_there_are_no_correct_rows_remain_after_error_filter(
        self,
        mock_config,
    ):
        series = pd.Series(data=["a", "b", "a"])
        errors = ColumnErrors(
            {
                CellError("a", 0, ErrorType.MISPLACED),
                CellError("a", 1, ErrorType.MISPLACED),
                CellError("a", 2, ErrorType.INVALID),
            }
        )
        assert (
            determine_fix_strategy(series, ColumnType.Category, errors)
            == FixStrategy.UNABLE_TO_FIX
        )


class TestDetermineRefineStrategy:
    @pytest.mark.parametrize(
        "series, errors, expect_fix_strategy",
        [
            (
                pd.Series(data=["a", "a", "a", "a"]),
                ColumnErrors(set()),
                FixStrategy.SINGLE_CATEGORY,
            ),
            (
                pd.Series(data=["a", "b", "a", "a"]),
                ColumnErrors({CellError("a", 1, ErrorType.INVALID)}),
                FixStrategy.SINGLE_CATEGORY,
            ),
            (
                pd.Series(data=["a", "b", "a", "a"]),
                ColumnErrors({CellError("a", 0, ErrorType.INVALID)}),
                FixStrategy.AUTOGLUON,
            ),
        ],
    )
    def test_should_return_correct_strategy(self, series, errors, expect_fix_strategy):
        assert determine_refine_strategy(series, errors) == expect_fix_strategy
