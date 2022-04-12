import pandas as pd
import numpy as np

np.random.seed(1)
from unittest.mock import MagicMock

from actableai.data_imputation.auto_fixer.datetime_fixer import DatetimeFixer
from actableai.data_imputation.meta.column import RichColumnMeta
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
    CellError,
    ErrorType,
)
from datetime import date, datetime


def test_integration_should_fix_to_correct_value_when_indices_are_in_middle():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2015-02-24", periods=20, freq="T"),
            "x": ["a", "a", "a", "c", "b", "b", "c", "b", None, "b"] * 2,
        }
    )
    drop_indices = np.random.randint(1, 18, 10)
    df.iloc[drop_indices, :] = None

    errors = [CellError("Date", index, ErrorType.NULL) for index in drop_indices]

    all_errors = CellErrors(
        MagicMock(),
        errors,
    )

    fixer = DatetimeFixer()
    fix_info = fixer.fix(df, all_errors, RichColumnMeta("Date", ColumnType.Timestamp))

    for index in sorted(set(drop_indices)):
        fix = fix_info.get_item(index, "Date")
        assert len(fix.options.options) == 1
        assert fix.options.options[0].value.to_pydatetime() == datetime.strptime(
            f"2015-02-24 00:{index:02}:00", "%Y-%m-%d %H:%M:%S"
        )


def test_integration_should_fix_to_correct_value_when_indices_are_in_edge():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2015-02-24", periods=20, freq="T"),
            "x": ["a", "a", "a", "c", "b", "b", "c", "b", None, "b"] * 2,
        }
    )
    drop_indices = [0, 19]
    df.iloc[drop_indices, :] = None

    errors = [CellError("Date", index, ErrorType.NULL) for index in drop_indices]

    all_errors = CellErrors(
        MagicMock(),
        errors,
    )

    fixer = DatetimeFixer()
    fix_info = fixer.fix(df, all_errors, RichColumnMeta("Date", ColumnType.Timestamp))

    for index in sorted(set(drop_indices)):
        fix = fix_info.get_item(index, "Date")
        assert len(fix.options.options) == 1
        assert fix.options.options[0].value.to_pydatetime() == datetime.strptime(
            f"2015-02-24 00:{index:02}:00", "%Y-%m-%d %H:%M:%S"
        )
