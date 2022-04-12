import math
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Text, Set

import numpy as np
import pandas as pd

from actableai.data_imputation.config import (
    MATCH_ROW_NUM_THRESHOLD,
)
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.regex_consts import (
    REGEX_CONSTS,
)


def _is_constantly_incremental(series: pd.Series) -> bool:
    expect_offset = Decimal(math.inf)
    if len(series) <= 1:
        return False

    series = series.sort_index()
    prev_index = series.index[0]
    prev_v = Decimal(str(series[prev_index]))
    for cur_index in series.index[1:]:
        cur_v = series[cur_index]
        if np.isnan(cur_v):
            return False
        cur_v = Decimal(str(cur_v))
        index_offset = cur_index - prev_index
        v_offset = cur_v - prev_v
        if expect_offset == math.inf:
            expect_offset = v_offset / index_offset
        else:
            if v_offset / index_offset != expect_offset:
                return False
        prev_v = cur_v
        prev_index = cur_index
    return True


def build_dtype_detector(series: pd.Series) -> "DtypeDetector":
    if sum(pd.isnull(series)) == len(series):
        return NullDetector(series)
    if series.dtype.name in ["category", "bool"]:
        return CategoryDetector(series)
    elif series.dtype == np.object or "time" in series.dtype.name:
        return ObjectDetector(series)
    elif series.dtype == np.int64:
        return Int64Detector(series)
    elif series.dtype == np.float64:
        return Float64Detector(series)
    else:
        raise TypeError(f"unsupported dtype {series.dtype}")


def detect_possible_type_for_column(series: pd.Series) -> Set[ColumnType]:
    detector = build_dtype_detector(series)
    if isinstance(detector, NullDetector):
        return {ColumnType.NULL}
    elif isinstance(detector, CategoryDetector):
        return {ColumnType.Category}
    elif isinstance(detector, Int64Detector):
        types = {ColumnType.Category, ColumnType.Integer}
        types.add(detector.get_column_type())
        return types
    elif isinstance(detector, Float64Detector):
        types = {ColumnType.Category, ColumnType.Float}
        types.add(detector.get_column_type())
        return types
    elif isinstance(detector, ObjectDetector):
        types = {ColumnType.Category}
        types.add(detector.get_column_type())
        return types
    else:
        raise NotImplementedError


class DtypeDetector(ABC):
    def __init__(self, series: pd.Series):
        self._series_full = series
        self._series = series[~series.isnull()]
        self._full_series = series

    @abstractmethod
    def get_column_type(self) -> ColumnType:
        raise NotImplementedError

    def _match_row_pass_threshold(self, match: pd.Series) -> bool:
        if len(self._series) * MATCH_ROW_NUM_THRESHOLD <= len(self._series[match]):
            return True
        return False


class NullDetector(DtypeDetector):
    def get_column_type(self) -> ColumnType:
        return ColumnType.NULL


class CategoryDetector(DtypeDetector):
    def get_column_type(self) -> ColumnType:
        return ColumnType.Category


class ObjectDetector(DtypeDetector):
    def __init__(self, series: pd.Series):
        super().__init__(series.astype(str).str.strip())

    def __is_integer(self):
        try:
            return np.array_equal(
                self._series.dropna().values,
                self._series.dropna().values.astype(int),
            )
        except ValueError:
            return False

    def __get_type_special(self) -> str:
        from autogluon.core.features.infer_types import (
            check_if_datetime_as_object_feature,
            check_if_nlp_feature,
        )
        from pandas.api.types import (
            is_numeric_dtype,
            is_datetime64_any_dtype,
            infer_dtype,
        )

        x = self._series
        type_special = "unknown"
        if len(x) > 0:
            # only 1 distinct value OR
            # at least 2 distinct values appear more than once
            if (
                len(self._series.value_counts()) == 1
                or sum(self._series.value_counts() > 1) > 1
            ):
                return "category"
            elif "mixed" in infer_dtype(x):
                type_special = "mixed"
            elif is_datetime64_any_dtype(x):
                type_special = "datetime"
            elif check_if_datetime_as_object_feature(x):
                type_special = "datetime"
            elif check_if_nlp_feature(x):
                type_special = "text"
            elif self.__is_integer():
                type_special = "integer"
            elif is_numeric_dtype(x):
                type_special = "numeric"
        elif len(x) == 0:
            type_special = "empty"
        return type_special

    def __is_category(self) -> bool:
        return self.__get_type_special() in ["category", "integer"]

    def __is_match(self, regex: Text) -> bool:
        match = self._series.str.match(regex)
        return self._match_row_pass_threshold(match)

    def __is_timestamp(self) -> bool:
        return self.__get_type_special() == "datetime"

    def __is_temperature(self) -> bool:
        regex = REGEX_CONSTS[ColumnType.Temperature]
        return self.__is_match(regex)

    def __is_percent(self) -> bool:
        regex = REGEX_CONSTS[ColumnType.Percentage]
        return self.__is_match(regex)

    def __is_num_with_tag(self) -> bool:
        regex = REGEX_CONSTS[ColumnType.NumWithTag]
        return self.__is_match(regex)

    def __is_complex(self) -> bool:
        try:
            self._series.astype("complex")
            return True
        except:
            return False

    def __is_text(self) -> bool:
        return self.__get_type_special() == "text"

    def get_column_type(self) -> ColumnType:
        if self.__is_timestamp():
            return ColumnType.Timestamp
        elif self.__is_temperature():
            return ColumnType.Temperature
        elif self.__is_percent():
            return ColumnType.Percentage
        elif self.__is_num_with_tag():
            return ColumnType.NumWithTag
        elif self.__is_complex():
            return ColumnType.Complex
        elif self.__is_category():
            return ColumnType.Category
        elif self.__is_text():
            return ColumnType.Text
        else:
            return ColumnType.String


class Int64Detector(DtypeDetector):
    def get_column_type(self) -> ColumnType:
        if _is_constantly_incremental(self._series_full):
            return ColumnType.Id
        return ColumnType.Integer


class Float64Detector(DtypeDetector):
    def get_column_type(self) -> ColumnType:
        if _is_constantly_incremental(self._series_full):
            return ColumnType.Id

        match = self._series % 1 == 0
        if self._match_row_pass_threshold(match):
            return ColumnType.Integer
        else:
            return ColumnType.Float
