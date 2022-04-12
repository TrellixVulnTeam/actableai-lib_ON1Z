from datetime import datetime
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.error_detector.cell_erros import (
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.column import NumWithTagColumnMeta
from actableai.data_imputation.type_recon.type_detector import DfTypes
from actableai.data_imputation.processor import ProcessOps, Processor
from actableai.data_imputation.meta.types import ColumnType

stub_path = "actableai.data_imputation.processor.processor"


@patch(f"{stub_path}.CellErrors")
@pytest.mark.parametrize(
    "df, dftypes, df_expect, dftypes_expect",
    [
        (
            pd.DataFrame(data={"a": ["10%", "20 %  ", "-0.4  %"]}),
            DfTypes([("a", ColumnType.Percentage)]),
            pd.DataFrame(
                data={
                    "__a_num__": [10.0, 20.0, -0.4],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": ["%", "%", "%"],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.Percentage),
                ]
            ),
        ),
        (
            pd.DataFrame(data={"a": [" -10 °C", "  +20.2 °C  ", "50  °F"]}),
            DfTypes([("a", ColumnType.Temperature)]),
            pd.DataFrame(
                data={
                    "__a_num__": [-10.0, 20.2, 10.0],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": ["°C", "°C", "°C"],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.Temperature),
                ]
            ),
        ),
        (
            pd.DataFrame(data={"a": ["", "  +20.2 °C  ", "50  °F"]}),
            DfTypes([("a", ColumnType.Temperature)]),
            pd.DataFrame(
                data={
                    "__a_num__": [np.nan, 20.2, 10.0],
                    "__a_ltag__": ["", "", ""],
                    "__a_rtag__": ["", "°C", "°C"],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.Temperature),
                ]
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "a": [
                        "Tag: -10 Suffix",
                        "What 20 °C  ",
                        "Whatever -0.4  °F",
                    ]
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            pd.DataFrame(
                data={
                    "__a_num__": [-10.0, 20.0, -0.4],
                    "__a_ltag__": ["Tag: ", "What ", "Whatever "],
                    "__a_rtag__": [" Suffix", " °C", " °F"],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.NumWithTag),
                ]
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "a": [
                        "Tag: -10 Suffix",
                        "What 20 °C  ",
                        "Whatever -3  °F",
                        "Only left -10",
                    ]
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            pd.DataFrame(
                data={
                    "__a_num__": [-10.0, 20.0, -3.0, -10],
                    "__a_ltag__": ["Tag: ", "What ", "Whatever ", "Only left "],
                    "__a_rtag__": [" Suffix", " °C", " °F", ""],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.NumWithTag),
                ]
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "a": [
                        "1300   south montgomery avenue",
                        "200 med center drive",
                    ]
                }
            ),
            DfTypes([("a", ColumnType.NumWithTag)]),
            pd.DataFrame(
                data={
                    "__a_num__": [1300.0, 200.0],
                    "__a_ltag__": ["", ""],
                    "__a_rtag__": [
                        " south montgomery avenue",
                        " med center drive",
                    ],
                }
            ),
            DfTypes(
                [
                    ("a", ColumnType.NumWithTag),
                ]
            ),
        ),
    ],
)
def test_expand_num_with_tag(
    mock_cell_errors_mod, df, dftypes, df_expect, dftypes_expect
):
    mock_cell_errors = MagicMock()
    mock_cell_errors_mod.return_value = mock_cell_errors

    processor = Processor(df, dftypes)
    processed_df = processor.expand_num_with_tag()
    assert processed_df.equals(df_expect)
    assert processor.get_column_types() == dftypes_expect


@pytest.mark.parametrize(
    "df, dftypes, target_column_name, num_column_name, expect_arg",
    [
        (
            pd.DataFrame(data={"a": ["10%", "20 %  ", "-3  %"]}),
            DfTypes([("a", ColumnType.Percentage)]),
            "a",
            "__a_num__",
            "int",
        ),
        (
            pd.DataFrame(data={"a": ["10%", "20 %  ", "-3.2 %"]}),
            DfTypes([("a", ColumnType.Percentage)]),
            "a",
            "__a_num__",
            "float",
        ),
        (
            pd.DataFrame(data={"a": ["10%", np.nan, "-3 %"]}),
            DfTypes([("a", ColumnType.Percentage)]),
            "a",
            "__a_num__",
            "int",
        ),
        (
            pd.DataFrame(data={"a": ["10%", np.nan, "-3.2 %"]}),
            DfTypes([("a", ColumnType.Percentage)]),
            "a",
            "__a_num__",
            "float",
        ),
    ],
)
def test_expand_num_with_tag_should_set_num_type_in_meta(
    df, dftypes, target_column_name, num_column_name, expect_arg
):
    mock_get_meta = MagicMock()
    mock_meta = MagicMock(NumWithTagColumnMeta)
    mock_meta.get_num_column_name.return_value = num_column_name
    mock_meta.original_type = dftypes[target_column_name]
    mock_get_meta.return_value = mock_meta
    dftypes.get_meta = mock_get_meta
    processor = Processor(df, dftypes)
    processor.expand_num_with_tag()

    assert mock_meta.set_num_type.call_args_list == [call(expect_arg)]


@patch(f"{stub_path}.ColumnTypeUnsupported", [ColumnType.Text])
@pytest.mark.parametrize(
    "df, dftypes, expect_unsupported_columns, expect_df",
    [
        (
            pd.DataFrame(
                data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]},
            ),
            DfTypes([("a", ColumnType.Float)]),
            [],
            pd.DataFrame(
                data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]},
            ),
        ),
        (
            pd.DataFrame(
                data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]},
            ),
            DfTypes([("a", ColumnType.Text)]),
            ["a"],
            pd.DataFrame(),
        ),
        (
            pd.DataFrame(
                data={
                    "a": [1.0, 2.0, 3.0, np.nan, 3.1],
                    "b": [1.0, 2.0, 3.0, np.nan, 3.1],
                    "c": [1.0, 2.0, 3.0, np.nan, 3.1],
                },
            ),
            DfTypes(
                [
                    ("a", ColumnType.Text),
                    ("b", ColumnType.Float),
                    ("c", ColumnType.Text),
                ]
            ),
            ["a", "c"],
            pd.DataFrame(data={"b": [1.0, 2.0, 3.0, np.nan, 3.1]}),
        ),
    ],
)
def test_exclude_unsupported_columns(
    df, dftypes, expect_unsupported_columns, expect_df
):
    mock_mark_column_unsupported = MagicMock()
    dftypes.mark_column_unsupported = mock_mark_column_unsupported
    processor = Processor(df, dftypes)
    result = processor._exclude_unsupported_columns()
    if expect_unsupported_columns:
        assert mock_mark_column_unsupported.call_args_list == [
            call(col) for col in expect_unsupported_columns
        ]
    assert result.equals(expect_df)


@pytest.mark.parametrize(
    "df, dftypes, df_expect, use_cell_per_cell_compare",
    [
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]}),
            DfTypes([("a", ColumnType.Category)]),
            pd.DataFrame(
                data={"a": ["1.0", "2.0", "3.0", np.nan, "3.1"]},
                dtype="str",
            ),
            False,
        ),
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]}, dtype="category"),
            DfTypes([("a", ColumnType.Category)]),
            pd.DataFrame(
                data={"a": ["1.0", "2.0", "3.0", np.nan, "3.1"]},
                dtype="str",
            ),
            False,
        ),
        (
            pd.DataFrame(data={"a": [1.0, "a", np.nan, 3.1]}),
            DfTypes([("a", ColumnType.String)]),
            pd.DataFrame(
                data={"a": ["1.0", "a", np.nan, "3.1"]},
            ),
            False,
        ),
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]}),
            DfTypes([("a", ColumnType.Integer)]),
            pd.DataFrame(data={"a": [1, 2, 3, np.nan, 3]}),
            False,
        ),
        (
            pd.DataFrame(data={"a": [1, 2, 3, np.nan, 3.1]}),
            DfTypes([("a", ColumnType.Float)]),
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, np.nan, 3.1]}),
            False,
        ),
        (
            pd.DataFrame(data={"a": [1, 2j, 3.1, "3i"]}),
            DfTypes([("a", ColumnType.Complex)]),
            pd.DataFrame(
                data={"a": ["1+0j", "2j", "3.1+0j", "3j"]}, dtype="complex128"
            ),
            True,  # there is a bug in pandas equal when compare with complex number
        ),
        (
            pd.DataFrame(
                data={
                    "a": [
                        "2021-01-02",
                        "02/01/2021",
                        None,
                        "2021-01-02T10:01:01Z",
                    ]
                }
            ),
            DfTypes([("a", ColumnType.Timestamp)]),
            pd.DataFrame(
                data={
                    "a": [
                        datetime.strptime("2021-01-02", "%Y-%m-%d"),
                        datetime.strptime("02/01/2021", "%d/%m/%Y"),
                        None,
                        datetime.strptime("2021-01-02T10:01:01Z", "%Y-%m-%dT%H:%M:%SZ"),
                    ]
                }
            ),
            True,
        ),
    ],
)
def test_convert_df_for_fix(df, dftypes, df_expect, use_cell_per_cell_compare):
    processor = Processor(df, dftypes)
    processed_df = processor._convert_df_for_fix()
    if use_cell_per_cell_compare:
        for col in df.columns:
            for i in range(len(df)):
                if (processed_df[col][i] is None and pd.isna(df_expect[col][i])) or (
                    pd.isna(processed_df[col][i]) and pd.isna(df_expect[col][i])
                ):
                    pass
                else:
                    assert (
                        processed_df[col][i] == df_expect[col][i]
                    ), f"col: {col}, index: {i} not equal"
    else:
        assert processed_df.equals(df_expect)


@patch(f"{stub_path}.CategoriesDataProcessor")
def test_convert_category_to_label(mock_categories_processor_mod):
    df = MagicMock()
    dftypes = MagicMock()
    mock_categories_processor = MagicMock(name="aaaa")
    mock_categories_processor_mod.return_value = mock_categories_processor

    processor = Processor(df, dftypes)
    processor._convert_category_to_label()
    assert mock_categories_processor.encode.called


@pytest.mark.parametrize(
    "df, dftypes, errors, df_expect",
    [
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, 3.1]}),
            DfTypes([("a", ColumnType.Float)]),
            CellErrors(
                DfTypes([("a", ColumnType.Float)]),
                [CellError(column="a", index=2, error_type=ErrorType.NULL)],
            ),
            pd.DataFrame(data={"a": [1.0, 2.0, np.nan, 3.1]}),
        ),
        (
            pd.DataFrame(data={"a": [1, 2, 3, 3]}),
            DfTypes([("a", ColumnType.Integer)]),
            CellErrors(
                DfTypes([("a", ColumnType.Integer)]),
                [CellError(column="a", index=2, error_type=ErrorType.INVALID)],
            ),
            pd.DataFrame(data={"a": [1.0, 2.0, np.nan, 3.0]}),
        ),
    ],
)
def test_replace_all_error_to_na(df, dftypes, errors, df_expect):
    processor = Processor(df, dftypes)
    processed_df = processor._replace_all_error_to_na(errors)
    assert processed_df.equals(df_expect)


def test_chain():
    mock_expand_num_with_tag = MagicMock()
    mock_exclude_unsupported_columns = MagicMock()
    mock_convert_df_for_fix = MagicMock()
    mock_convert_category_to_label = MagicMock()
    mock_replace_all_error_to_na = MagicMock()
    errors = MagicMock()
    processes = [
        ProcessOps.EXPEND_NUM_WITH_TAG,
        ProcessOps.EXCLUDE_UNSUPPORTED_COLUMNS,
        ProcessOps.COLUMN_AS_DETECTED_TYPE_TO_TRAIN,
        ProcessOps.CATEGORY_TO_LABEL_NUMBER,
        ProcessOps.REPLACE_ALL_ERROR_TO_NA,
    ]
    processor = Processor(MagicMock(), MagicMock())
    setattr(
        processor,
        "expand_num_with_tag",
        mock_expand_num_with_tag,
    )
    setattr(
        processor,
        "_exclude_unsupported_columns",
        mock_exclude_unsupported_columns,
    )
    setattr(
        processor,
        "_convert_df_for_fix",
        mock_convert_df_for_fix,
    )
    setattr(
        processor,
        "_convert_category_to_label",
        mock_convert_category_to_label,
    )
    setattr(
        processor,
        "_replace_all_error_to_na",
        mock_replace_all_error_to_na,
    )
    processor.chain(processes, errors)
    assert mock_expand_num_with_tag.called
    assert mock_exclude_unsupported_columns.called
    assert mock_convert_df_for_fix.called
    assert mock_convert_category_to_label.called
    assert mock_replace_all_error_to_na.called


@pytest.mark.parametrize(
    "df, dftypes, df_expect",
    [
        (
            pd.DataFrame(data={"a": [1, 2, 3]}),
            DfTypes([("a", ColumnType.Text)]),
            pd.DataFrame(data={"a": ["1", "2", "3"]}),
        ),
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.0, 8.9999, 4.111111]}),
            DfTypes([("a", ColumnType.Integer)]),
            pd.DataFrame(data={"a": ["1", "2", "3", "9", "4"]}),
        ),
        (
            pd.DataFrame(data={"a": [1.0, 2.0, 3.9999, np.NaN]}),
            DfTypes([("a", ColumnType.Integer)]),
            pd.DataFrame(
                data={
                    "a": [
                        "1",
                        "2",
                        "4",
                        "",
                    ]
                }
            ),
        ),
        (
            pd.DataFrame(
                data={
                    "a": [
                        datetime.strptime("2021-01-01", "%Y-%m-%d").timestamp(),
                        datetime.strptime(
                            "2021-01-01 10:11:12", "%Y-%m-%d %H:%M:%S"
                        ).timestamp(),
                    ]
                }
            ),
            DfTypes([("a", ColumnType.Timestamp)]),
            pd.DataFrame(
                data={
                    "a": [
                        datetime.strptime("2021-01-01", "%Y-%m-%d"),
                        datetime.strptime("2021-01-01 10:11:12", "%Y-%m-%d %H:%M:%S"),
                    ]
                }
            ),
        ),
    ],
)
def test_restore_types(df, dftypes, df_expect):
    processor = Processor(df, dftypes)
    df_result = processor._restore_types(df)
    assert df_result.equals(df_expect)
