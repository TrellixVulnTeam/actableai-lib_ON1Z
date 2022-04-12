import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.auto_fixer.helper import (
    get_df_without_error,
    get_df_with_only_error,
    finalize_columns,
    fulfil_fix_back,
    merge_num_with_tag_columns,
)
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.column import NumWithTagColumnMeta
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


@pytest.mark.parametrize(
    "df,errors,expect_df",
    [
        (
            pd.DataFrame.from_dict({"A": [1, 2, 3]}),
            CellErrors(
                DfTypes([("A", ColumnType.Integer)]),
                [CellError("A", 0, ErrorType.NULL)],
            ),
            pd.DataFrame(data={"A": [2, 3]}, index=[1, 2]),
        ),
        (
            pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]}),
            CellErrors(
                DfTypes([("A", ColumnType.Integer), ("B", ColumnType.Integer)]),
                [
                    CellError("A", 0, ErrorType.NULL),
                    CellError("B", 2, ErrorType.INVALID),
                ],
            ),
            pd.DataFrame(data={"A": [2], "B": ["b"]}, index=[1]),
        ),
    ],
)
def test_get_df_without_error(df, errors, expect_df):
    result = get_df_without_error(df, errors)
    assert result.equals(expect_df)


@pytest.mark.parametrize(
    "df,errors,expect_df",
    [
        (
            pd.DataFrame.from_dict({"A": [1, 2, 3]}),
            CellErrors(
                DfTypes([("A", ColumnType.Integer)]),
                [CellError("A", 0, ErrorType.NULL)],
            ),
            pd.DataFrame(data={"A": [1]}, index=[0]),
        ),
        (
            pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]}),
            CellErrors(
                DfTypes([("A", ColumnType.Integer), ("B", ColumnType.Integer)]),
                [
                    CellError("A", 0, ErrorType.NULL),
                    CellError("B", 2, ErrorType.INVALID),
                ],
            ),
            pd.DataFrame(data={"A": [1, 3], "B": ["a", "c"]}, index=[0, 2]),
        ),
    ],
)
def test_get_df_without_error(df, errors, expect_df):
    result = get_df_with_only_error(df, errors)
    assert result.equals(expect_df)


@pytest.mark.parametrize(
    "df, fix_info_list, expect_df",
    [
        (
            pd.DataFrame(
                data={
                    "a": ["a", "b", "c"],
                    "b": [1.0, 2.0, 3.0],
                    "c": ["e", None, None],
                }
            ),
            FixInfoList(
                [
                    FixInfo(
                        "b",
                        2,
                        options=FixValueOptions(
                            options=[FixValue(value=4.0, confidence=1)]
                        ),
                    ),
                    FixInfo(
                        "c",
                        0,
                        options=FixValueOptions(
                            options=[FixValue(value="n", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        "c",
                        1,
                        options=FixValueOptions(
                            options=[FixValue(value="m", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        "c",
                        2,
                        options=FixValueOptions(
                            options=[FixValue(value="l", confidence=1)]
                        ),
                    ),
                ]
            ),
            pd.DataFrame(
                data={
                    "a": ["a", "b", "c"],
                    "b": [1.0, 2.0, 4.0],
                    "c": ["n", "m", "l"],
                }
            ),
        )
    ],
)
def test_fulfil_fix_back(df, fix_info_list, expect_df):
    assert fulfil_fix_back(df, fix_info_list).equals(expect_df)


@pytest.mark.parametrize(
    "df, column_types, expect_df",
    [
        (
            pd.DataFrame(
                {
                    "__a_num__": [10, 15, 11.1],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": ["%", "%", "%"],
                }
            ),
            DfTypes([("a", ColumnType.Percentage)]),
            pd.DataFrame({"a": ["10.0%", "15.0%", "11.1%"]}),
        ),
        (
            pd.DataFrame(
                {
                    "__a_num__": [10, None, 11.1, np.nan],
                    "__a_ltag__": ["", "", "", ""],
                    "__a_rtag__": ["%", "%", "%", "%"],
                }
            ),
            DfTypes([("a", ColumnType.Percentage)]),
            pd.DataFrame({"a": ["10.0%", np.nan, "11.1%", np.nan]}),
        ),
        (
            pd.DataFrame(
                {
                    "__a_num__": [np.nan, 1],
                    "__a_ltag__": [np.nan, ""],
                    "__a_rtag__": [np.nan, "%"],
                }
            ),
            DfTypes([("a", ColumnType.Percentage)]),
            pd.DataFrame({"a": [np.nan, "1.0%"]}),
        ),
        (
            pd.DataFrame(
                data={
                    "__a_num__": [-10, 20, 10],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": ["°C", "°C", "°C"],
                }
            ),
            DfTypes([("a", ColumnType.Percentage)]),
            pd.DataFrame(data={"a": ["-10°C", "20°C", "10°C"]}),
        ),
        (
            pd.DataFrame(
                data={
                    "__a_num__": [1300, 200],
                    "__a_ltag__": ["", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        " med center drive",
                    ],
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            pd.DataFrame(
                data={
                    "a": [
                        "1300 south montgomery avenue",
                        "200 med center drive",
                    ]
                }
            ),
        ),
    ],
)
def test_merge_num_with_tag_columns(df, column_types, expect_df):
    actual_df = merge_num_with_tag_columns(df, column_types)
    assert actual_df.equals(expect_df)


@pytest.mark.parametrize(
    "df, column_types, meta_column, num_type, expect_df",
    [
        (
            pd.DataFrame(
                data={
                    "__a_num__": [1300.0, 200.1],
                    "__a_ltag__": ["", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        " med center drive",
                    ],
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            "a",
            "int",
            pd.DataFrame(
                data={
                    "a": [
                        "1300 south montgomery avenue",
                        "200 med center drive",
                    ]
                }
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "__a_num__": [1300.0, np.nan, 200.1],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        np.nan,
                        " med center drive",
                    ],
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            "a",
            "int",
            pd.DataFrame(
                data={
                    "a": [
                        "1300 south montgomery avenue",
                        np.nan,
                        "200 med center drive",
                    ]
                }
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "__a_num__": [1300.0, np.nan, 200.1],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        np.nan,
                        " med center drive",
                    ],
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            "a",
            "float",
            pd.DataFrame(
                data={
                    "a": [
                        "1300.0 south montgomery avenue",
                        np.nan,
                        "200.1 med center drive",
                    ]
                }
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "__a_num__": [1300.0, np.nan, np.nan],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        np.nan,
                        " med center drive",
                    ],
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            "a",
            "int",
            pd.DataFrame(data={"a": ["1300 south montgomery avenue", np.nan, np.nan]}),
        ),
    ],
)
def test_merge_num_with_tag_columns_should_respect_num_type(
    df, column_types, meta_column, num_type, expect_df
):
    meta = column_types.get_meta(meta_column)
    assert isinstance(meta, NumWithTagColumnMeta)
    meta.set_num_type(num_type)
    actual_df = merge_num_with_tag_columns(df, column_types)
    assert actual_df.equals(expect_df)


@pytest.mark.parametrize(
    "original_df, fixed_df, expect_df",
    [
        (
            pd.DataFrame(
                data={
                    "a": ["a", "b", "c"],
                    "b": [1.0, 2.0, 3.0],
                    "c": [None, None, None],
                }
            ),
            pd.DataFrame(data={"c": [3.0, 4.0, 5.0]}),
            pd.DataFrame(
                data={
                    "a": ["a", "b", "c"],
                    "b": [1.0, 2.0, 3.0],
                    "c": [3.0, 4.0, 5.0],
                }
            ),
        )
    ],
)
def test_finalize_columns(original_df, fixed_df, expect_df):
    original_df_copy = original_df.copy()
    assert finalize_columns(original_df, fixed_df).equals(expect_df)
    assert original_df_copy.equals(original_df)
