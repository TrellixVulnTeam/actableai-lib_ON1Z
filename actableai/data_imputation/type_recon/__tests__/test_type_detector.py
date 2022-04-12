from unittest.mock import patch, MagicMock

import pytest

from actableai.data_imputation.meta.column import (
    SingleValueColumnMeta,
    NumWithTagColumnMeta,
)
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon import TypeDetector
from actableai.data_imputation.type_recon.type_detector import DfTypes

stub_path = "actableai.data_imputation.type_recon.type_detector"


@pytest.fixture(autouse=True)
def type_detector():
    return TypeDetector()


@patch(f"{stub_path}.build_dtype_detector")
def test_detect_column(mock_build_detector, manual_data, type_detector):
    mock_detector = MagicMock()
    mock_build_detector.return_value = mock_detector
    type_detector.detect_column(manual_data["string"])
    mock_detector.get_column_type.assert_called()


@patch(f"{stub_path}.detect_possible_type_for_column")
def test_detect_possible_types(mock_detector, manual_data, type_detector):
    type_detector.detect_possible_types(manual_data)
    mock_detector.assert_called()
    assert mock_detector.call_count == len(manual_data.columns)


@patch(f"{stub_path}.build_dtype_detector")
def test_detect(mock_build_detector, manual_data, type_detector):
    mock_detector_1 = MagicMock()
    mock_detector_1.get_column_type.return_value = ColumnType.Integer
    mock_detector_2 = MagicMock()
    mock_detector_2.get_column_type.return_value = ColumnType.String
    mock_detector_3 = MagicMock()
    mock_detector_3.get_column_type.return_value = ColumnType.Category
    mock_detector_4 = MagicMock()
    mock_detector_4.get_column_type.return_value = ColumnType.Percentage
    mock_build_detector.side_effect = [
        mock_detector_1,
        mock_detector_2,
        mock_detector_3,
        mock_detector_4,
    ]
    detect_result = type_detector.detect(
        manual_data[["id", "string", "enum", "percent"]]
    )
    mock_detector_1.get_column_type.assert_called()
    mock_detector_2.get_column_type.assert_called()
    mock_detector_3.get_column_type.assert_called()
    mock_detector_4.get_column_type.assert_called()
    assert detect_result == DfTypes(
        [
            ("id", ColumnType.Integer),
            ("string", ColumnType.String),
            ("enum", ColumnType.Category),
            ("percent", ColumnType.Float),
        ]
    )


def test_df_types():
    df_types = DfTypes([("integer", ColumnType.Integer), ("string", ColumnType.String)])
    assert df_types["integer"] == ColumnType.Integer
    assert df_types["string"] == ColumnType.String


def test_df_types_get_item_when_column_is_expanded():
    df_types = DfTypes(
        [("integer", ColumnType.Integer), ("percentage", ColumnType.Percentage)]
    )
    assert df_types["integer"] == ColumnType.Integer
    assert df_types["percentage"] == ColumnType.Percentage
    assert df_types["__percentage_num__"] == ColumnType.Float
    assert df_types["__percentage_ltag__"] == ColumnType.Category
    assert df_types["__percentage_rtag__"] == ColumnType.Category


def test_df_types_get_meta():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    assert type(df_types.get_meta("percent")) == NumWithTagColumnMeta
    assert type(df_types.get_meta("__percent_ltag__")) == SingleValueColumnMeta
    assert type(df_types.get_meta("string")) == SingleValueColumnMeta


def test_df_types_columns_original():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    assert list(df_types.columns_original) == [
        "percent",
        "string",
    ]


def test_df_types_columns_original_should_ignore_columns_unsupported():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    df_types.mark_column_unsupported("string")
    assert list(df_types.columns_original) == [
        "percent",
    ]


def test_df_types_columns_after_expand():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    assert list(df_types._columns_after_expand) == [
        "__percent_ltag__",
        "__percent_num__",
        "__percent_rtag__",
        "string",
    ]


def test_df_types_columns_after_expand_should_not_ignore_columns_unsupported():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    df_types.mark_column_unsupported("__percent_ltag__")
    assert list(df_types._columns_after_expand) == [
        "__percent_ltag__",
        "__percent_num__",
        "__percent_rtag__",
        "string",
    ]


def test_mark_column_unsupported_should_mark_expanded_columns_as_unsupported_when_mark_original_column():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    df_types.mark_column_unsupported("percent")
    assert df_types.is_support_fix("percent") is False
    assert df_types.is_support_fix("__percent_ltag__") is False
    assert df_types.is_support_fix("__percent_num__") is False
    assert df_types.is_support_fix("__percent_rtag__") is False


def test_df_types_columns_to_fix():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    assert list(df_types.columns_to_fix) == [
        "__percent_ltag__",
        "__percent_num__",
        "__percent_rtag__",
        "string",
    ]


def test_df_types_columns_to_fix_should_ignore_columns_unsupported():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    df_types.mark_column_unsupported("string")
    assert list(df_types.columns_to_fix) == [
        "__percent_ltag__",
        "__percent_num__",
        "__percent_rtag__",
    ]


def test_df_types_is_support_fix():
    df_types = DfTypes(
        [("percent", ColumnType.Percentage), ("string", ColumnType.String)]
    )
    assert df_types.is_support_fix("string") is True
    df_types.mark_column_unsupported("string")
    assert df_types.is_support_fix("string") is False
    assert df_types.is_support_fix("percent") is True
    assert df_types.is_support_fix("__percent_ltag__") is True
    assert df_types.is_support_fix("__percent_num__") is True
    assert df_types.is_support_fix("__percent_rtag__") is True


def test_df_types_override():
    df_types = DfTypes([("integer", ColumnType.Integer), ("string", ColumnType.String)])
    df_types.override("integer", ColumnType.Float)
    assert df_types["integer"] == ColumnType.Float
    assert df_types["string"] == ColumnType.String


def test_df_types_override_should_raise_error_when_column_not_present_in_original():
    df_types = DfTypes([])
    with pytest.raises(KeyError):
        df_types.override("a", ColumnType.Integer)


def test_df_types_should_raise_key_error_when_column_not_found():
    df_types = DfTypes([("integer", ColumnType.Integer), ("string", ColumnType.String)])
    with pytest.raises(KeyError):
        _ = df_types["non_exist"]
