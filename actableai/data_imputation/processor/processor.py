import numpy as np
import pandas as pd
import re
from datetime import datetime
from enum import Enum, auto
from functools import partial
from typing import List

from actableai.data_imputation.config import NAN_INTEGER
from actableai.data_imputation.config import logger
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import NumWithTagColumnMeta
from actableai.data_imputation.meta.types import (
    ColumnTypeUnsupported,
    ColumnType,
)
from actableai.data_imputation.processor.categories_data_processor import (
    CategoriesDataProcessor,
)
from actableai.data_imputation.type_recon.helper import as_datetime
from actableai.data_imputation.type_recon.type_detector import DfTypes


class ProcessOps(Enum):
    EXPEND_NUM_WITH_TAG = auto()  # expand num_with_tag column to separate columns
    EXCLUDE_UNSUPPORTED_COLUMNS = auto()
    COLUMN_AS_DETECTED_TYPE_TO_TRAIN = (
        auto()
    )  # convert column as value in detected type
    CATEGORY_TO_LABEL_NUMBER = auto()
    REPLACE_ALL_ERROR_TO_NA = auto()


class Processor:
    def __init__(self, df: pd.DataFrame, dftypes: DfTypes):
        self.__original_df = df
        self.__current_df = df
        self.__dftypes = dftypes
        self.__categories_data_processor = CategoriesDataProcessor()

    def expand_num_with_tag(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for col in self.__current_df.columns:
            col_meta = self.__dftypes.get_meta(col)
            if isinstance(col_meta, NumWithTagColumnMeta):
                v = self.__current_df[col].str.extract(
                    r"\s*([^0-9\-\+]*\s*)([+-]?\d+\.?\d*)(\s*[\S\s]*)\s*"
                )
                num_col_name = col_meta.get_num_column_name()
                if v[1].str.contains(r"\.").any():
                    col_meta.set_num_type("float")
                else:
                    col_meta.set_num_type("int")
                df[num_col_name] = v[1].apply(float)
                left_tag_name = col_meta.get_left_tag_column_name()
                right_tag_name = col_meta.get_right_tag_column_name()
                if col_meta.original_type in [
                    ColumnType.Temperature,
                    ColumnType.Percentage,
                ]:
                    left_tag = v[0].astype(str).apply(lambda x: x.strip())
                    right_tag = v[2].astype(str).apply(lambda x: x.strip())
                elif col_meta.original_type == ColumnType.NumWithTag:
                    left_tag = v[0].astype(str).apply(lambda x: re.sub(r"\s+", " ", x))

                    right_tag = (
                        v[2]
                        .astype(str)
                        .apply(lambda x: re.sub(r"\s+", " ", x))
                        .apply(lambda x: re.sub(r"\s+$", "", x))
                    )
                else:
                    raise NotImplementedError

                # np.nan to str will become nan
                left_tag = left_tag.apply(lambda x: "" if x == "nan" else x)
                right_tag = right_tag.apply(lambda x: "" if x == "nan" else x)

                df[left_tag_name] = left_tag
                df[right_tag_name] = right_tag

                # additional step: convert fahrenheit value
                if col_meta.original_type == ColumnType.Temperature:
                    is_fahrenheit = df[right_tag_name] == "°F"
                    df.loc[is_fahrenheit, num_col_name] = pd.Series(
                        (df[num_col_name][is_fahrenheit] - 32) * 5 / 9
                    )
                    df.loc[is_fahrenheit, right_tag_name] = "°C"
            else:
                df[col] = self.__current_df[col]

        return df

    def _exclude_unsupported_columns(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for col in self.__current_df.columns:
            t = self.__dftypes[col]
            if t not in ColumnTypeUnsupported:
                df[col] = self.__current_df[col]
            else:
                self.__dftypes.mark_column_unsupported(col)
                logger.warning(
                    f"{col} is ignored for data fix due to the type {self.__dftypes[col]} is not supported"
                )

        return df

    def _convert_df_for_fix(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for col in self.__current_df.columns:
            column_type = self.__dftypes[col]
            column = self.__current_df[col].copy()
            if column_type == ColumnType.Category:
                column = np.where(pd.isnull(column), column, column.astype(str))
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype("str")
            elif column_type == ColumnType.String:
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype("str")
            elif column_type == ColumnType.Integer:
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype("int64")
            elif column_type == ColumnType.Float:
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype(
                    "float64"
                )
            elif column_type == ColumnType.Complex:
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype("str")
                column[~pd.isnull(column)] = column[~pd.isnull(column)].str.replace(
                    "i", "j"
                )
                column[~pd.isnull(column)] = column[~pd.isnull(column)].str.replace(
                    " ", ""
                )
                column[~pd.isnull(column)] = column[~pd.isnull(column)].astype(
                    "complex128"
                )
            elif column_type == ColumnType.Timestamp:
                column[~pd.isnull(column)] = as_datetime(
                    column[~pd.isnull(column)].astype(str),
                )

            df[col] = column
        return df

    def _convert_category_to_label(self) -> pd.DataFrame:
        return self.__categories_data_processor.encode(
            self.__current_df, self.__dftypes
        )

    def _replace_all_error_to_na(self, errors: CellErrors) -> pd.DataFrame:
        df = self.__current_df
        for err in errors:
            if df[err.column].dtype in [np.int32, np.int64]:
                df[err.column] = df[err.column].astype(np.float64)
            df.at[err.index, err.column] = np.nan
        return df

    def chain(self, processes: List[ProcessOps], errors: CellErrors):
        for process in processes:
            if process == ProcessOps.EXPEND_NUM_WITH_TAG:
                op = self.expand_num_with_tag
            elif process == ProcessOps.EXCLUDE_UNSUPPORTED_COLUMNS:
                op = self._exclude_unsupported_columns
            elif process == ProcessOps.COLUMN_AS_DETECTED_TYPE_TO_TRAIN:
                op = self._convert_df_for_fix
            elif process == ProcessOps.CATEGORY_TO_LABEL_NUMBER:
                op = self._convert_category_to_label
            elif process == ProcessOps.REPLACE_ALL_ERROR_TO_NA:
                op = partial(self._replace_all_error_to_na, errors)
            else:
                raise NotImplementedError(f"{process} is not supported")
            self.__current_df = op()

    def _restore_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = pd.DataFrame()
        for col in df.columns:
            expect_type = self.__dftypes[col]
            logger.info(f"Restore {col} to type {expect_type}")
            if expect_type in [
                ColumnType.Category,
                ColumnType.String,
                ColumnType.Text,
            ]:
                df_result[col] = list(map(str, df[col]))
            elif expect_type == ColumnType.Integer:
                values = df[col]
                nan_indexes = np.isnan(values)
                values[nan_indexes] = NAN_INTEGER
                df_result[col] = values.apply(
                    lambda x: int(round(x)) if not np.isnan(x) else np.nan
                )
            elif expect_type == ColumnType.Float:
                df_result[col] = df[col].apply(float)
            elif expect_type == ColumnType.Complex:
                df_result[col] = df[col].apply(complex)
            elif expect_type == ColumnType.Timestamp:
                if df[col].dtype == float:
                    df_result[col] = df[col].apply(datetime.fromtimestamp)
                else:
                    df_result[col] = df[col]
            else:
                df_result[col] = df[col]
        return df_result

    def get_processed_df(self) -> pd.DataFrame:
        return self.__current_df

    def get_column_types(self) -> DfTypes:
        return self.__dftypes

    def restore(self, df: pd.DataFrame):
        df = self.__categories_data_processor.decode(df)
        return self._restore_types(df)
