from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.dtype_detector import (
    build_dtype_detector,
    ObjectDetector,
    Int64Detector,
    Float64Detector,
    CategoryDetector,
    NullDetector,
    detect_possible_type_for_column,
)


@pytest.mark.parametrize(
    "column,expect_detector,expect_type",
    [
        ("id", Int64Detector, ColumnType.Id),
        ("enum", ObjectDetector, ColumnType.Category),
        ("percent", ObjectDetector, ColumnType.Percentage),
        ("timestamp", ObjectDetector, ColumnType.Timestamp),
        ("temperature", ObjectDetector, ColumnType.Temperature),
        ("integer", Float64Detector, ColumnType.Integer),
        ("float", Float64Detector, ColumnType.Float),
        ("complex", ObjectDetector, ColumnType.Complex),
        ("number_with_tag", ObjectDetector, ColumnType.NumWithTag),
        ("long_string", ObjectDetector, ColumnType.Text),
        ("null", NullDetector, ColumnType.NULL),
        ("mix", ObjectDetector, ColumnType.String),
    ],
)
def test_dtype_detector(manual_data, column, expect_detector, expect_type):
    detector = build_dtype_detector(manual_data[column])
    assert type(detector) == expect_detector
    assert detector.get_column_type() == expect_type


def test_null_detector():
    series = pd.Series(data=[None, np.nan, np.nan, None], dtype="category")
    detector = build_dtype_detector(series)
    assert type(detector) == NullDetector


def test_category_detector():
    series = pd.Series(data=["a", "b", "a"], dtype="category")
    detector = build_dtype_detector(series)
    assert type(detector) == CategoryDetector


@pytest.mark.parametrize(
    "series, expect_type",
    [
        [pd.Series([3, 5, 6, 2]), ColumnType.Integer],
        [pd.Series([None, None, 3, 4]), ColumnType.Integer],
        [pd.Series([None, np.NaN, 3, 4]), ColumnType.Integer],
        [pd.Series([None, np.NaN, 3.1, 4]), ColumnType.Float],
        [pd.Series([1.0, 2.0, 3.0, 4.0]), ColumnType.Id],
        [pd.Series([1, 2, 3, 4]), ColumnType.Id],
        [pd.Series([1.1, 2.1, 3.1, 4.1]), ColumnType.Id],
    ],
)
def test_float64_detector(series, expect_type):
    detector = Float64Detector(series)
    assert detector.get_column_type() == expect_type


@pytest.mark.parametrize(
    "series, expect_types",
    [
        [pd.Series([None, None]), {ColumnType.NULL}],
        [pd.Series([3, 5, 6, 2]), {ColumnType.Integer, ColumnType.Category}],
        [
            pd.Series([None, None, 3, 4]),
            {ColumnType.Integer, ColumnType.Float, ColumnType.Category},
        ],
        [
            pd.Series([None, np.NaN, 3, 4]),
            {ColumnType.Integer, ColumnType.Float, ColumnType.Category},
        ],
        [
            pd.Series([None, np.NaN, 3.1, 4]),
            {ColumnType.Category, ColumnType.Float},
        ],
        [
            pd.Series([1.0, 2.0, 3.0, 4.0]),
            {
                ColumnType.Category,
                ColumnType.Id,
                ColumnType.Float,
            },
        ],
        [
            pd.Series([1, 2, 3, 4]),
            {ColumnType.Integer, ColumnType.Category, ColumnType.Id},
        ],
        [
            pd.Series([1.1, 2.1, 3.1, 4.1]),
            {ColumnType.Float, ColumnType.Category, ColumnType.Id},
        ],
        [
            pd.Series(
                [
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                ],
                dtype="datetime64[ns]",
            ),
            {ColumnType.Category, ColumnType.Timestamp},
        ],
    ],
)
def test_detect_possible_type_for_column(series, expect_types):
    assert detect_possible_type_for_column(series) == expect_types


@pytest.mark.parametrize(
    "value, match",
    [
        ["yes:22", True],
        ["yes : 22", True],
        ["22yes", True],
        ["22 yes", True],
        ["yes     :   1234567998765.1234", True],
        ["    -12123123yes", True],
        ["bibou:123", True],
        ["mango            123", True],
        ["22.2memo", True],
        ["22", False],
        ["     -123 : pear", False],
        ["22.swing", False],
        ["", False],
    ],
)
def test_regex_num_with_tag(value, match):
    objd = ObjectDetector(pd.Series(value))
    actual = objd._ObjectDetector__is_num_with_tag()
    assert actual == match
