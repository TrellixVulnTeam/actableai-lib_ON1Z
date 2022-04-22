import numpy as np
import pytest

from actableai.data_imputation.auto_fixer.__tests__.helper import (
    assert_fix_info_list,
)
from actableai.data_imputation.auto_fixer.auto_gluon_fixer import AutoGluonFixer
from actableai.data_imputation.auto_fixer.errors import EmptyTrainDataException
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.auto_fixer.neighbor_fixer import NeighborFixer
from actableai.data_imputation.data import DataFrame
from actableai.data_imputation.meta import ColumnType
from actableai.data_imputation.meta.column import RichColumnMeta
from actableai.utils.testing import unittest_hyperparameters


@pytest.mark.parametrize(
    "broken_df, column, expect_fix_info_list",
    [
        (
            DataFrame.from_dict(
                {
                    "B": [2, 2, None, 2, np.nan, 2, 2],
                    "C": [1, 1, 1, 1, 1, 1, 1],
                    "A": [30, None, 30, None, 30, np.nan, None],
                    "D": [5, 5, 5, 5, 5, 5, 5],
                }
            ),
            "A",
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value=30.0, confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=3,
                        options=FixValueOptions(
                            options=[FixValue(value=30.0, confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=5,
                        options=FixValueOptions(
                            options=[FixValue(value=30.0, confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=6,
                        options=FixValueOptions(
                            options=[FixValue(value=30.0, confidence=1)]
                        ),
                    ),
                ]
            ),
        ),
    ],
)
def test_neighbor_fixer(broken_df, column, expect_fix_info_list):
    df_origin = broken_df.copy()

    errors = broken_df.detect_error()

    fixer = NeighborFixer()
    fix_info_list = fixer.fix(
        broken_df, errors, RichColumnMeta("A", ColumnType.Integer)
    )

    assert_fix_info_list(fix_info_list, expect_fix_info_list)
    assert broken_df.equals(df_origin)


@pytest.mark.parametrize(
    "broken_df",
    [
        DataFrame.from_dict({"A": [None, None, None, None], "B": [30, 30, 30, 30]}),
        DataFrame.from_dict(
            {"A": [np.nan, None, np.nan, None], "B": [30, 30, 30, None]}
        ),
    ],
)
def test_neighbor_fixer_should_throw_error_when_there_is_no_sufficient_data_to_train(
    broken_df,
):
    errors = broken_df.detect_error()

    fixer = NeighborFixer()

    with pytest.raises(EmptyTrainDataException):
        fixer.fix(broken_df, errors, RichColumnMeta("A", ColumnType.Integer))


@pytest.mark.parametrize(
    "broken_df, current_column, expect_fix_info_list",
    [
        (
            DataFrame.from_dict(
                {
                    "a": [
                        "a",
                        "b",
                        "b",
                        "b",
                        "b",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                    ],
                    "b": [1, 2, None, 2, 2, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "c": [None, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                }
            ),
            RichColumnMeta("b", ColumnType.Integer),
            FixInfoList(
                [
                    FixInfo(
                        col="b",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value=2.0, confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="b",
                        index=5,
                        options=FixValueOptions(
                            options=[FixValue(value=1.0, confidence=1)]
                        ),
                    ),
                ]
            ),
        ),
    ],
)
def test_auto_gluon_fix(broken_df, current_column, expect_fix_info_list):
    df_origin = broken_df.copy()
    errors = broken_df.detect_error()

    fixer = AutoGluonFixer()
    fix_info_list = fixer.fix(
        broken_df, errors, current_column, ag_hyperparameters=unittest_hyperparameters()
    )

    assert_fix_info_list(fix_info_list, expect_fix_info_list)
    assert broken_df.equals(df_origin)
